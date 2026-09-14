"""Events, alerts and the Local-AI-friendly read endpoints."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from app.alerts.engine import get_alert_engine
from app.api.deps import DbSession, Manager, PaginationDep, window_since
from app.core.exceptions import NotFoundError
from app.models.enums import AlertState, EventSeverity, EventType
from app.repositories.event_repository import AlertRepository, EventRepository
from app.repositories.source_repository import SourceRepository
from app.schemas.common import OperationResult
from app.schemas.intelligence import AlertRead, EventRead

router = APIRouter(prefix="/api", tags=["events"])


# ------------------------------------------------------------------ events
@router.get("/events", response_model=list[EventRead])
def list_events(
    session: DbSession,
    page: PaginationDep,
    source_id: int | None = None,
    event_type: list[EventType] | None = Query(None),
    severity: EventSeverity | None = None,
    identity_id: int | None = None,
    hours: int = Query(24, ge=1, le=8760),
):
    return EventRepository(session).list_events(
        source_id=source_id,
        event_types=event_type,
        severity=severity,
        identity_id=identity_id,
        since=window_since(hours),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/events/recent", response_model=list[EventRead])
def recent_events(
    session: DbSession,
    limit: int = Query(50, ge=1, le=500),
    since: datetime | None = None,
    min_severity: EventSeverity | None = None,
):
    """Structured feed intended for downstream consumers (e.g. a local LLM).

    Stable, self-describing event objects; no UI-specific shaping.
    """
    return EventRepository(session).list_events(
        since=since, severity=min_severity, limit=limit
    )


@router.get("/events/{event_id}", response_model=EventRead)
def get_event(event_id: int, session: DbSession):
    event = EventRepository(session).get(event_id)
    if event is None:
        raise NotFoundError(f"event {event_id} not found")
    return event


# ------------------------------------------------------------------ alerts
@router.get("/alerts", response_model=list[AlertRead])
def list_alerts(
    session: DbSession,
    page: PaginationDep,
    source_id: int | None = None,
    state: AlertState | None = None,
    hours: int = Query(168, ge=1, le=8760),
):
    return AlertRepository(session).list_alerts(
        source_id=source_id,
        state=state,
        since=window_since(hours),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/alerts/active", response_model=list[AlertRead])
def active_alerts(session: DbSession, source_id: int | None = None):
    return AlertRepository(session).active_alerts(source_id=source_id)


@router.get("/security/active", response_model=dict)
def active_security_state(session: DbSession, manager: Manager):
    """Consolidated 'what is happening right now' view."""
    alerts = AlertRepository(session).active_alerts()
    open_events = EventRepository(session).list_events(limit=100)
    sources = {s.id: s for s in SourceRepository(session).list_sources()}
    return {
        "system": manager.system_state(),
        "active_alerts": [
            {
                "id": a.id,
                "source_id": a.source_id,
                "source_uid": sources[a.source_id].uid if a.source_id in sources else None,
                "alert_type": str(getattr(a.alert_type, "value", a.alert_type)),
                "reason": a.reason,
                "started_at": a.started_at.isoformat(),
                "track_id": a.track_id,
                "identity_id": a.identity_id,
            }
            for a in alerts
        ],
        "open_events": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "event_type": str(getattr(e.event_type, "value", e.event_type)),
                "label": e.label,
                "started_at": e.started_at.isoformat(),
                "duration_seconds": e.duration_seconds,
            }
            for e in open_events if e.is_open
        ],
    }


@router.post("/alerts/{alert_id}/stop", response_model=OperationResult)
def stop_alert(alert_id: int, session: DbSession, stopped_by: str = "operator"):
    """Manual STOP ALARM. This is the only thing that ends a continuous alarm."""
    repo = AlertRepository(session)
    existing = repo.get(alert_id)
    if existing is None:
        raise NotFoundError(f"alert {alert_id} not found")
    source = SourceRepository(session).get(existing.source_id)
    alert = get_alert_engine().stop(
        session, alert_id, source_uid=source.uid if source else None, stopped_by=stopped_by
    )
    session.commit()
    return OperationResult(
        data={
            "alert_id": alert_id,
            "state": str(getattr(alert.state, "value", alert.state)) if alert else "UNKNOWN",
        }
    )


@router.post("/alerts/stop-all", response_model=OperationResult)
def stop_all_alerts(session: DbSession, source_id: int | None = None,
                    stopped_by: str = "operator"):
    count = get_alert_engine().stop_all(
        session, source_id=source_id, stopped_by=stopped_by
    )
    session.commit()
    return OperationResult(data={"stopped": count})
