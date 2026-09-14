"""Archive: recordings, playback and source -> recording -> event traceability."""
from __future__ import annotations

import mimetypes
import re

from fastapi import APIRouter, Header, Query, Response
from fastapi.responses import FileResponse, StreamingResponse

from app.api.deps import DbSession, PaginationDep
from app.core.exceptions import NotFoundError, StorageError
from app.models.enums import RecordingStatus
from app.repositories.event_repository import EventRepository
from app.repositories.recording_repository import RecordingRepository
from app.repositories.source_repository import SourceRepository
from app.schemas.intelligence import EventRead, RecordingDetail, RecordingRead
from app.storage.paths import recording_root, resolve_recording

router = APIRouter(prefix="/api", tags=["archive"])

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
CHUNK = 1 << 20


def _detail(session, recording) -> RecordingDetail:
    source = SourceRepository(session).get(recording.source_id)
    detail = RecordingDetail.model_validate(recording)
    detail.source_uid = source.uid if source else None
    detail.source_name = source.name if source else None
    detail.playback_url = f"/api/recordings/{recording.id}/play"
    try:
        detail.exists_on_disk = resolve_recording(recording.file_path).exists()
    except StorageError:
        detail.exists_on_disk = False
    detail.event_count = len(
        EventRepository(session).list_events(
            source_id=recording.source_id, limit=1000
        )
    )
    return detail


@router.get("/recordings", response_model=list[RecordingDetail])
def list_recordings(
    session: DbSession,
    page: PaginationDep,
    source_id: int | None = None,
    status: RecordingStatus | None = None,
):
    recordings = RecordingRepository(session).list_recordings(
        source_id=source_id, status=status, limit=page.limit, offset=page.offset
    )
    return [_detail(session, r) for r in recordings]


@router.get("/recordings/{recording_id}", response_model=RecordingDetail)
def get_recording(recording_id: int, session: DbSession):
    recording = RecordingRepository(session).get(recording_id)
    if recording is None:
        raise NotFoundError(f"recording {recording_id} not found")
    return _detail(session, recording)


@router.get("/recordings/{recording_id}/events", response_model=list[EventRead])
def recording_events(recording_id: int, session: DbSession,
                     limit: int = Query(200, ge=1, le=1000)):
    """Events captured while this recording session was open."""
    recording = RecordingRepository(session).get(recording_id)
    if recording is None:
        raise NotFoundError(f"recording {recording_id} not found")
    events = EventRepository(session).list_events(
        source_id=recording.source_id,
        since=recording.started_at,
        until=recording.ended_at,
        limit=limit,
    )
    return [e for e in events if e.recording_id in (None, recording_id)]


@router.get("/recordings/{recording_id}/play")
def play_recording(
    recording_id: int, session: DbSession, range_header: str | None = Header(None, alias="Range")
):
    """Stream a recording with HTTP range support so the player can seek."""
    recording = RecordingRepository(session).get(recording_id)
    if recording is None:
        raise NotFoundError(f"recording {recording_id} not found")

    path = resolve_recording(recording.file_path)
    if not path.exists():
        raise NotFoundError("recording file is no longer on disk")

    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"

    if not range_header:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    match = RANGE_RE.match(range_header)
    if not match:
        return FileResponse(path, media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    start = int(match.group(1)) if match.group(1) else 0
    end = int(match.group(2)) if match.group(2) else size - 1
    start = max(0, min(start, size - 1))
    end = max(start, min(end, size - 1))
    length = end - start + 1

    def iter_file():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


@router.get("/archive/tree", response_model=dict)
def archive_tree(session: DbSession):
    """Source -> recording-session summary for the archive browser."""
    sources = SourceRepository(session).list_sources()
    repo = RecordingRepository(session)
    return {
        "sources": [
            {
                "id": s.id,
                "uid": s.uid,
                "name": s.name,
                "type": str(getattr(s.type, "value", s.type)),
                "recordings": repo.stats(s.id),
            }
            for s in sources
        ],
        "totals": repo.stats(),
        "storage_root": str(recording_root()),
    }
