"""Recording session persistence."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, func, select, update

from app.models.enums import RecordingStatus
from app.models.recording import RecordingSession
from app.repositories.base import BaseRepository


class RecordingRepository(BaseRepository[RecordingSession]):
    model = RecordingSession

    def start(
        self,
        *,
        source_id: int,
        file_path: str,
        started_at: datetime | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        codec: str | None = None,
    ) -> RecordingSession:
        session = RecordingSession(
            source_id=source_id,
            file_path=file_path,
            status=RecordingStatus.ACTIVE,
            started_at=started_at or datetime.now(timezone.utc),
            fps=fps,
            width=width,
            height=height,
            codec=codec,
        )
        return self.add(session)

    def finish(
        self,
        recording_id: int,
        *,
        status: RecordingStatus,
        frame_count: int,
        file_size_bytes: int | None,
        ended_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        ended = ended_at or datetime.now(timezone.utc)
        recording = self.get(recording_id)
        if recording is None:
            return
        started = recording.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        self.session.execute(
            update(RecordingSession)
            .where(RecordingSession.id == recording_id)
            .values(
                status=status,
                ended_at=ended,
                duration_seconds=max(0.0, (ended - started).total_seconds()),
                frame_count=frame_count,
                file_size_bytes=file_size_bytes,
                error=error,
            )
        )

    def active_for_source(self, source_id: int) -> RecordingSession | None:
        return self.session.execute(
            select(RecordingSession)
            .where(
                RecordingSession.source_id == source_id,
                RecordingSession.status == RecordingStatus.ACTIVE,
            )
            .order_by(desc(RecordingSession.started_at))
            .limit(1)
        ).scalar_one_or_none()

    def list_for_source(
        self, source_id: int, *, limit: int = 100, offset: int = 0
    ) -> list[RecordingSession]:
        return list(
            self.session.execute(
                select(RecordingSession)
                .where(RecordingSession.source_id == source_id)
                .order_by(desc(RecordingSession.started_at))
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )

    def list_recordings(
        self,
        *,
        source_id: int | None = None,
        status: RecordingStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RecordingSession]:
        stmt = select(RecordingSession).order_by(desc(RecordingSession.started_at))
        if source_id is not None:
            stmt = stmt.where(RecordingSession.source_id == source_id)
        if status is not None:
            stmt = stmt.where(RecordingSession.status == status)
        return list(self.session.execute(stmt.limit(limit).offset(offset)).scalars().all())

    def close_orphans(self) -> int:
        """Mark recordings left ACTIVE by a crash as INTERRUPTED.

        Runs at startup. The file on disk is whatever the writer managed to
        flush; it is never appended to, a new session always gets a new file.
        """
        result = self.session.execute(
            update(RecordingSession)
            .where(RecordingSession.status == RecordingStatus.ACTIVE)
            .values(
                status=RecordingStatus.INTERRUPTED,
                ended_at=datetime.now(timezone.utc),
                error="process restarted while recording was active",
            )
        )
        return int(result.rowcount or 0)

    def stats(self, source_id: int | None = None) -> dict:
        stmt = select(
            func.count(RecordingSession.id),
            func.coalesce(func.sum(RecordingSession.duration_seconds), 0.0),
            func.coalesce(func.sum(RecordingSession.file_size_bytes), 0),
        )
        if source_id is not None:
            stmt = stmt.where(RecordingSession.source_id == source_id)
        count, duration, size = self.session.execute(stmt).one()
        return {
            "count": int(count or 0),
            "total_duration_seconds": float(duration or 0.0),
            "total_bytes": int(size or 0),
        }
