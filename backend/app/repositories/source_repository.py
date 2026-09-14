"""Video source persistence."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, update

from app.models.enums import SourceStatus
from app.models.source import VideoSource
from app.repositories.base import BaseRepository


class SourceRepository(BaseRepository[VideoSource]):
    model = VideoSource

    def get_by_uid(self, uid: str) -> VideoSource | None:
        return self.session.execute(
            select(VideoSource).where(VideoSource.uid == uid)
        ).scalar_one_or_none()

    def list_sources(self, *, enabled_only: bool = False) -> list[VideoSource]:
        stmt = select(VideoSource).order_by(VideoSource.display_order, VideoSource.id)
        if enabled_only:
            stmt = stmt.where(VideoSource.enabled.is_(True))
        return list(self.session.execute(stmt).scalars().all())

    def list_active_for_surveillance(self) -> list[VideoSource]:
        stmt = (
            select(VideoSource)
            .where(VideoSource.enabled.is_(True), VideoSource.surveillance_enabled.is_(True))
            .order_by(VideoSource.display_order, VideoSource.id)
        )
        return list(self.session.execute(stmt).scalars().all())

    def set_status(
        self, source_id: int, status: SourceStatus, *, error: str | None = None
    ) -> None:
        values: dict = {"status": status, "last_error": error}
        if status is SourceStatus.AVAILABLE:
            values["last_seen_at"] = datetime.now(timezone.utc)
        self.session.execute(
            update(VideoSource).where(VideoSource.id == source_id).values(**values)
        )

    def set_intelligence(self, source_id: int, enabled: bool) -> None:
        self.session.execute(
            update(VideoSource)
            .where(VideoSource.id == source_id)
            .values(intelligence_enabled=enabled)
        )

    def set_intelligence_all(self, enabled: bool) -> int:
        result = self.session.execute(
            update(VideoSource).values(intelligence_enabled=enabled)
        )
        return int(result.rowcount or 0)

    def set_surveillance(self, source_id: int, enabled: bool) -> None:
        self.session.execute(
            update(VideoSource)
            .where(VideoSource.id == source_id)
            .values(surveillance_enabled=enabled)
        )

    def next_display_order(self) -> int:
        stmt = select(VideoSource.display_order).order_by(VideoSource.display_order.desc())
        current = self.session.execute(stmt.limit(1)).scalar()
        return int(current or 0) + 1

    def reset_all_statuses(self) -> None:
        """Called at startup: no source is running yet after a restart."""
        self.session.execute(
            update(VideoSource)
            .where(VideoSource.status != SourceStatus.IDLE)
            .values(status=SourceStatus.IDLE)
        )
