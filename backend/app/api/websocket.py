"""Realtime WebSocket channel.

Carries structured state only - detections, tracks, recognition changes,
proximity transitions, alarm state, source state, recording state - so the
UI never has to poll. Video pixels go over the separate MJPEG endpoint.
"""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.alerts.engine import get_alert_engine
from app.core.logging import get_logger
from app.db.session import session_scope
from app.events.bus import get_bus
from app.repositories.event_repository import AlertRepository
from app.repositories.identity_repository import FaceRepository
from app.services.surveillance import get_manager

log = get_logger(__name__)
router = APIRouter(tags=["realtime"])

HEARTBEAT_SECONDS = 20.0


def _initial_snapshot() -> dict:
    manager = get_manager()
    payload = {
        "type": "snapshot",
        "system": manager.system_state(),
        "sources": list(manager.all_runtime_status().values()),
        "active_alerts": [],
        "pending_review": 0,
        # The backend owns the mute state so every console agrees on it.
        "alarm_sound": get_alert_engine().sound_enabled,
    }
    try:
        with session_scope() as session:
            payload["active_alerts"] = [
                {
                    "id": a.id,
                    "source_id": a.source_id,
                    "alert_type": str(getattr(a.alert_type, "value", a.alert_type)),
                    "reason": a.reason,
                    "started_at": a.started_at.isoformat(),
                    "track_id": a.track_id,
                    "identity_id": a.identity_id,
                    "expires_at": (a.alert_metadata or {}).get("expires_at"),
                }
                for a in AlertRepository(session).active_alerts()
            ]
            payload["pending_review"] = FaceRepository(session).pending_review_count()
    except Exception as exc:  # pragma: no cover - DB may be unavailable
        log.warning("ws_snapshot_failed", error=str(exc))
    return payload


@router.websocket("/ws")
async def realtime(
    websocket: WebSocket,
    topics: str | None = Query(None, description="comma-separated event types"),
    sources: str | None = Query(None, description="comma-separated source uids"),
):
    await websocket.accept()
    bus = get_bus()
    subscriber = bus.subscribe(
        topics=[t.strip() for t in topics.split(",") if t.strip()] if topics else None,
        sources=[s.strip() for s in sources.split(",") if s.strip()] if sources else None,
    )

    try:
        await websocket.send_json(_initial_snapshot())
    except Exception:
        bus.unsubscribe(subscriber)
        return

    async def pump() -> None:
        while True:
            try:
                message = await asyncio.wait_for(
                    subscriber.queue.get(), timeout=HEARTBEAT_SECONDS
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            await websocket.send_json(message.to_dict())

    async def receive() -> None:
        # Drain client messages so the socket stays healthy; a client may send
        # {"type": "ping"} or a resubscribe request.
        while True:
            data = await websocket.receive_json()
            if isinstance(data, dict) and data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    pump_task = asyncio.create_task(pump())
    receive_task = asyncio.create_task(receive())
    try:
        done, pending = await asyncio.wait(
            {pump_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                raise exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - one bad socket must not matter
        log.info("ws_closed", reason=str(exc))
    finally:
        for task in (pump_task, receive_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        bus.unsubscribe(subscriber)
        with contextlib.suppress(Exception):
            await websocket.close()


@router.get("/api/events/stream-buffer")
def recent_bus_messages(limit: int = Query(50, ge=1, le=200)):
    """Last messages published on the bus (debugging aid)."""
    return get_bus().recent(limit)
