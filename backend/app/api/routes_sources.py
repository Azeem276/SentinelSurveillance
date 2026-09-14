"""Video source CRUD, control, streaming and per-source analysis."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import DbSession, Manager, PaginationDep, window_since
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.enums import EventType, SourceType
from app.models.source import VideoSource
from app.repositories.detection_repository import (
    DetectionRepository, MotionRepository, TrackRepository,
)
from app.repositories.event_repository import AlertRepository, EventRepository
from app.repositories.identity_repository import FaceRepository, IdentityRepository
from app.repositories.source_repository import SourceRepository
from app.schemas.common import OperationResult
from app.schemas.intelligence import (
    DetectionRead, EventRead, MotionEventRead, SourceAnalysis, TrackRead,
)
from app.schemas.source import (
    SourceCreate, SourceDetail, SourceListItem, SourceRuntime, SourceUpdate,
)
from app.services.analysis_service import build_source_analysis
from app.services.frame_hub import get_frame_hub
from app.storage.paths import (
    ALLOWED_VIDEO_SUFFIXES, resolve_video, validate_video_upload, video_root,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/sources", tags=["sources"])

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def _runtime(manager, source_id: int) -> SourceRuntime:
    data = manager.runtime_status(source_id)
    return SourceRuntime(
        running=bool(data.get("running", False)),
        status=str(data.get("status", "IDLE")),
        recording=bool(data.get("recording", False)),
        recording_id=data.get("recording_id"),
        intelligence=bool(data.get("intelligence", False)),
        motion=bool(data.get("motion", False)),
        active_tracks=int(data.get("active_tracks", 0) or 0),
        frames_read=int(data.get("frames_read", 0) or 0),
        error=data.get("error"),
        stats=data.get("stats") or {},
        capabilities=data.get("capabilities"),
    )


def _get_source(session, source_id: int) -> VideoSource:
    source = SourceRepository(session).get(source_id)
    if source is None:
        raise NotFoundError(f"source {source_id} not found")
    return source


# ------------------------------------------------------------------- CRUD
@router.get("", response_model=list[SourceListItem])
def list_sources(session: DbSession, manager: Manager, enabled_only: bool = False):
    repo = SourceRepository(session)
    alerts = AlertRepository(session)
    active = alerts.active_alerts()
    alert_counts: dict[int, int] = {}
    for alert in active:
        alert_counts[alert.source_id] = alert_counts.get(alert.source_id, 0) + 1

    out: list[SourceListItem] = []
    for source in repo.list_sources(enabled_only=enabled_only):
        item = SourceListItem.model_validate(source)
        item.runtime = _runtime(manager, source.id)
        item.unread_alerts = alert_counts.get(source.id, 0)
        out.append(item)
    return out


@router.post("", response_model=SourceDetail, status_code=status.HTTP_201_CREATED)
def create_source(payload: SourceCreate, session: DbSession, manager: Manager):
    repo = SourceRepository(session)
    if repo.get_by_uid(payload.uid) is not None:
        raise ValidationError(f"a source with uid {payload.uid!r} already exists")

    # File sources must resolve inside the configured video root.
    if payload.type is SourceType.FILE:
        path = resolve_video(payload.uri)
        if not path.exists():
            raise ValidationError(
                f"video file not found under {video_root()}: {payload.uri}"
            )
    elif payload.type is SourceType.RTSP and not payload.uri.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    ):
        raise ValidationError("RTSP sources require an rtsp:// or http(s):// URI")

    source = VideoSource(
        uid=payload.uid,
        name=payload.name,
        type=payload.type,
        uri=payload.uri,
        location=payload.location,
        description=payload.description,
        enabled=payload.enabled,
        surveillance_enabled=payload.surveillance_enabled,
        intelligence_enabled=payload.intelligence_enabled,
        recognition_enabled=payload.recognition_enabled,
        recording_enabled=payload.recording_enabled,
        loop_playback=payload.loop_playback,
        proximity_a=payload.proximity_a,
        proximity_b=payload.proximity_b,
        recognition_threshold=payload.recognition_threshold,
        detection_confidence=payload.detection_confidence,
        temporary_retention_days=payload.temporary_retention_days,
        alert_policy=payload.alert_policy.model_dump() if payload.alert_policy else {},
        calibration=payload.calibration.model_dump() if payload.calibration else {},
        display_order=repo.next_display_order(),
    )
    repo.add(source)
    session.commit()
    session.refresh(source)

    detail = SourceDetail.model_validate(source)
    detail.runtime = _runtime(manager, source.id)
    detail.stream_url = f"/api/sources/{source.id}/stream"
    return detail


@router.get("/{source_id}", response_model=SourceDetail)
def get_source(source_id: int, session: DbSession, manager: Manager):
    source = _get_source(session, source_id)
    detail = SourceDetail.model_validate(source)
    detail.runtime = _runtime(manager, source_id)
    detail.stream_url = f"/api/sources/{source_id}/stream"
    return detail


@router.patch("/{source_id}", response_model=SourceDetail)
def update_source(
    source_id: int, payload: SourceUpdate, session: DbSession, manager: Manager
):
    source = _get_source(session, source_id)
    data = payload.model_dump(exclude_unset=True)

    # Validate the A > B invariant against the merged result, not just input.
    new_a = data.get("proximity_a", source.proximity_a)
    new_b = data.get("proximity_b", source.proximity_b)
    if new_a is not None and new_b is not None and new_a <= new_b:
        raise ValidationError(
            "Proximity A (recognition zone) must be greater than Proximity B (alarm zone)"
        )
    if (new_a is not None and new_a <= 0) or (new_b is not None and new_b <= 0):
        raise ValidationError("Proximity distances must be greater than 0")

    for key, value in data.items():
        if key in ("alert_policy", "calibration") and value is not None:
            setattr(source, key, dict(value))
        else:
            setattr(source, key, value)

    session.commit()
    session.refresh(source)

    # A running worker picks the new settings up immediately.
    manager.reload_source(source_id)

    detail = SourceDetail.model_validate(source)
    detail.runtime = _runtime(manager, source_id)
    detail.stream_url = f"/api/sources/{source_id}/stream"
    return detail


@router.delete("/{source_id}", response_model=OperationResult)
def delete_source(source_id: int, session: DbSession, manager: Manager):
    source = _get_source(session, source_id)
    manager.stop_source(source_id)
    session.delete(source)
    session.commit()
    return OperationResult(data={"source_id": source_id, "deleted": True})


# ---------------------------------------------------------------- control
@router.post("/{source_id}/start", response_model=OperationResult)
def start_source(source_id: int, manager: Manager):
    return OperationResult(data=manager.start_source(source_id))


@router.post("/{source_id}/stop", response_model=OperationResult)
def stop_source(source_id: int, manager: Manager):
    return OperationResult(data=manager.stop_source(source_id))


@router.post("/{source_id}/restart", response_model=OperationResult)
def restart_source(source_id: int, manager: Manager):
    return OperationResult(data=manager.restart_source(source_id))


@router.post("/{source_id}/intelligence/start", response_model=OperationResult)
def start_source_intelligence(source_id: int, manager: Manager):
    return OperationResult(data=manager.set_source_intelligence(source_id, True))


@router.post("/{source_id}/intelligence/stop", response_model=OperationResult)
def stop_source_intelligence(source_id: int, manager: Manager):
    return OperationResult(data=manager.set_source_intelligence(source_id, False))


# --------------------------------------------------------------- streaming
@router.get("/{source_id}/stream")
async def stream_source(source_id: int, session: DbSession):
    """MJPEG preview.

    Video pixels are served over HTTP; detection overlays arrive separately
    over the WebSocket and are drawn by the client. That keeps the realtime
    channel small and lets the UI render crisp vector overlays.
    """
    source = _get_source(session, source_id)
    uid = source.uid
    hub = get_frame_hub()

    async def generate():
        boundary = b"--frame\r\n"
        last_frame_number = -1
        idle = 0
        while True:
            slot = hub.latest(uid)
            if slot is None or slot.frame_number == last_frame_number:
                idle += 1
                if idle > 200:  # ~20 s with no new frames
                    break
                await asyncio.sleep(0.1)
                continue
            idle = 0
            last_frame_number = slot.frame_number
            yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " + str(
                len(slot.jpeg)
            ).encode() + b"\r\n\r\n" + slot.jpeg + b"\r\n"
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/{source_id}/snapshot")
def source_snapshot(source_id: int, session: DbSession):
    source = _get_source(session, source_id)
    slot = get_frame_hub().latest(source.uid)
    if slot is None:
        raise HTTPException(status_code=404, detail="no frame available for this source")
    return Response(content=slot.jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------- upload
@router.post("/upload", response_model=OperationResult)
async def upload_video(file: UploadFile):
    """Add a clip to the sample-video directory.

    The filename is sanitised and the extension allow-listed; the file is
    written only inside VIDEO_STORAGE_PATH.
    """
    if not file.filename:
        raise ValidationError("no filename provided")
    safe_name = validate_video_upload(file.filename)
    target = video_root() / safe_name
    counter = 1
    while target.exists():
        counter += 1
        stem = safe_name.rsplit(".", 1)[0]
        suffix = safe_name.rsplit(".", 1)[1]
        target = video_root() / f"{stem}_{counter}.{suffix}"

    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValidationError("file exceeds the 2 GB upload limit")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    return OperationResult(
        data={"filename": target.name, "bytes": written, "uri": target.name}
    )


@router.get("/available/files", response_model=list[dict])
def list_available_videos():
    """Clips present in VIDEO_STORAGE_PATH that can back a source."""
    root = video_root()
    if not root.exists():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_VIDEO_SUFFIXES:
            out.append(
                {"filename": path.name, "size_bytes": path.stat().st_size}
            )
    return out


# --------------------------------------------------------------- analytics
@router.get("/{source_id}/events", response_model=list[EventRead])
def source_events(
    source_id: int,
    session: DbSession,
    page: PaginationDep,
    hours: int = Query(24, ge=1, le=8760),
    event_type: list[EventType] | None = Query(None),
):
    _get_source(session, source_id)
    return EventRepository(session).list_events(
        source_id=source_id,
        event_types=event_type,
        since=window_since(hours),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{source_id}/detections", response_model=list[DetectionRead])
def source_detections(
    source_id: int,
    session: DbSession,
    page: PaginationDep,
    hours: int = Query(24, ge=1, le=8760),
    object_class: str | None = None,
):
    _get_source(session, source_id)
    return DetectionRepository(session).list_for_source(
        source_id,
        limit=page.limit,
        offset=page.offset,
        object_class=object_class,
        since=window_since(hours),
    )


@router.get("/{source_id}/tracks", response_model=list[TrackRead])
def source_tracks(
    source_id: int,
    session: DbSession,
    page: PaginationDep,
    hours: int = Query(24, ge=1, le=8760),
    object_class: str | None = None,
):
    _get_source(session, source_id)
    return TrackRepository(session).list_for_source(
        source_id,
        limit=page.limit,
        offset=page.offset,
        object_class=object_class,
        since=window_since(hours),
    )


@router.get("/{source_id}/motion", response_model=list[MotionEventRead])
def source_motion(
    source_id: int, session: DbSession, hours: int = Query(24, ge=1, le=8760),
    limit: int = Query(100, ge=1, le=1000),
):
    _get_source(session, source_id)
    return MotionRepository(session).list_for_source(
        source_id, limit=limit, since=window_since(hours)
    )


@router.get("/{source_id}/analysis", response_model=SourceAnalysis)
def source_analysis(
    source_id: int,
    session: DbSession,
    manager: Manager,
    hours: int = Query(24, ge=1, le=8760),
    timeline_limit: int = Query(60, ge=1, le=500),
):
    source = _get_source(session, source_id)
    return build_source_analysis(
        session,
        source,
        hours=hours,
        timeline_limit=timeline_limit,
        runtime=manager.runtime_status(source_id),
    )
