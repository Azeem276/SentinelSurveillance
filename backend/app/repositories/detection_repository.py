"""Track, detection and motion-event persistence."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, func, select, update

from app.models.detection import Detection, MotionEvent, Track
from app.models.enums import ProximityZone, RecognitionState, TrackStatus
from app.repositories.base import BaseRepository


class TrackRepository(BaseRepository[Track]):
    model = Track

    def create(
        self,
        *,
        source_id: int,
        track_key: int,
        object_class: str,
        first_seen_at: datetime,
        first_frame: int,
        recording_id: int | None = None,
    ) -> Track:
        return self.add(
            Track(
                source_id=source_id,
                track_key=track_key,
                object_class=object_class,
                status=TrackStatus.ACTIVE,
                first_seen_at=first_seen_at,
                last_seen_at=first_seen_at,
                first_frame=first_frame,
                last_frame=first_frame,
                recording_id=recording_id,
                trajectory=[],
            )
        )

    def update_state(self, track_id: int, **values) -> None:
        if not values:
            return
        self.session.execute(update(Track).where(Track.id == track_id).values(**values))

    def close(
        self,
        track_id: int,
        *,
        last_seen_at: datetime | None = None,
        duration_seconds: float | None = None,
        trajectory: list | None = None,
    ) -> None:
        values: dict = {
            "status": TrackStatus.ENDED,
            "last_seen_at": last_seen_at or datetime.now(timezone.utc),
        }
        if duration_seconds is not None:
            values["duration_seconds"] = duration_seconds
        if trajectory is not None:
            values["trajectory"] = trajectory
        self.session.execute(update(Track).where(Track.id == track_id).values(**values))

    def close_orphans(self) -> int:
        result = self.session.execute(
            update(Track)
            .where(Track.status == TrackStatus.ACTIVE)
            .values(status=TrackStatus.ENDED)
        )
        return int(result.rowcount or 0)

    def list_for_source(
        self,
        source_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
        object_class: str | None = None,
        since: datetime | None = None,
    ) -> list[Track]:
        stmt = select(Track).where(Track.source_id == source_id)
        if object_class:
            stmt = stmt.where(Track.object_class == object_class)
        if since:
            stmt = stmt.where(Track.first_seen_at >= since)
        stmt = stmt.order_by(desc(Track.first_seen_at)).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def list_for_identity(self, identity_id: int, *, limit: int = 100) -> list[Track]:
        return list(
            self.session.execute(
                select(Track)
                .where(Track.identity_id == identity_id)
                .order_by(desc(Track.first_seen_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )

    def clear_identity(self, identity_id: int) -> int:
        """Detach a deleted identity from historical tracks (events survive)."""
        result = self.session.execute(
            update(Track)
            .where(Track.identity_id == identity_id)
            .values(identity_id=None, recognition_state=RecognitionState.UNFAMILIAR)
        )
        return int(result.rowcount or 0)

    def class_counts(self, source_id: int, *, since: datetime | None = None) -> dict[str, int]:
        stmt = select(Track.object_class, func.count(Track.id)).where(
            Track.source_id == source_id
        )
        if since:
            stmt = stmt.where(Track.first_seen_at >= since)
        rows = self.session.execute(stmt.group_by(Track.object_class)).all()
        return {str(cls): int(count) for cls, count in rows}

    def recognition_counts(
        self, source_id: int, *, since: datetime | None = None
    ) -> dict[str, int]:
        stmt = select(Track.recognition_state, func.count(Track.id)).where(
            Track.source_id == source_id, Track.object_class == "person"
        )
        if since:
            stmt = stmt.where(Track.first_seen_at >= since)
        rows = self.session.execute(stmt.group_by(Track.recognition_state)).all()
        return {str(getattr(s, "value", s)): int(c) for s, c in rows}


class DetectionRepository(BaseRepository[Detection]):
    model = Detection

    def bulk_add(self, detections: list[Detection]) -> None:
        if not detections:
            return
        self.session.add_all(detections)
        self.session.flush()

    def list_for_source(
        self,
        source_id: int,
        *,
        limit: int = 200,
        offset: int = 0,
        object_class: str | None = None,
        since: datetime | None = None,
        track_id: int | None = None,
    ) -> list[Detection]:
        stmt = select(Detection).where(Detection.source_id == source_id)
        if object_class:
            stmt = stmt.where(Detection.object_class == object_class)
        if since:
            stmt = stmt.where(Detection.timestamp >= since)
        if track_id is not None:
            stmt = stmt.where(Detection.track_id == track_id)
        stmt = stmt.order_by(desc(Detection.timestamp)).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_class(self, source_id: int, *, since: datetime | None = None) -> dict[str, int]:
        stmt = select(Detection.object_class, func.count(Detection.id)).where(
            Detection.source_id == source_id
        )
        if since:
            stmt = stmt.where(Detection.timestamp >= since)
        rows = self.session.execute(stmt.group_by(Detection.object_class)).all()
        return {str(cls): int(count) for cls, count in rows}


class MotionRepository(BaseRepository[MotionEvent]):
    model = MotionEvent

    def start(
        self,
        *,
        source_id: int,
        started_at: datetime,
        frame_start: int,
        recording_id: int | None = None,
        area_ratio: float = 0.0,
    ) -> MotionEvent:
        return self.add(
            MotionEvent(
                source_id=source_id,
                started_at=started_at,
                frame_start=frame_start,
                recording_id=recording_id,
                peak_area_ratio=area_ratio,
            )
        )

    def end(
        self,
        motion_id: int,
        *,
        ended_at: datetime,
        frame_end: int,
        peak_area_ratio: float | None = None,
    ) -> None:
        event = self.get(motion_id)
        if event is None:
            return
        started = event.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        values: dict = {
            "ended_at": ended_at,
            "frame_end": frame_end,
            "duration_seconds": max(0.0, (ended_at - started).total_seconds()),
        }
        if peak_area_ratio is not None:
            values["peak_area_ratio"] = max(event.peak_area_ratio, peak_area_ratio)
        self.session.execute(
            update(MotionEvent).where(MotionEvent.id == motion_id).values(**values)
        )

    def close_orphans(self) -> int:
        now = datetime.now(timezone.utc)
        result = self.session.execute(
            update(MotionEvent).where(MotionEvent.ended_at.is_(None)).values(ended_at=now)
        )
        return int(result.rowcount or 0)

    def list_for_source(
        self, source_id: int, *, limit: int = 100, since: datetime | None = None
    ) -> list[MotionEvent]:
        stmt = select(MotionEvent).where(MotionEvent.source_id == source_id)
        if since:
            stmt = stmt.where(MotionEvent.started_at >= since)
        return list(
            self.session.execute(stmt.order_by(desc(MotionEvent.started_at)).limit(limit))
            .scalars()
            .all()
        )
