"""Security event, alert and system-setting persistence."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select, update

from app.models.event import Alert, SecurityEvent, SystemSetting
from app.models.enums import AlertState, AlertType, EventSeverity, EventType
from app.repositories.base import BaseRepository


class EventRepository(BaseRepository[SecurityEvent]):
    model = SecurityEvent

    def create(
        self,
        *,
        source_id: int,
        event_type: EventType,
        started_at: datetime,
        severity: EventSeverity = EventSeverity.INFO,
        track_id: int | None = None,
        identity_id: int | None = None,
        face_id: int | None = None,
        recording_id: int | None = None,
        motion_event_id: int | None = None,
        confidence: float | None = None,
        label: str | None = None,
        message: str | None = None,
        snapshot_path: str | None = None,
        dedup_key: str | None = None,
        is_open: bool = False,
        metadata: dict | None = None,
        ended_at: datetime | None = None,
    ) -> SecurityEvent:
        event = SecurityEvent(
            source_id=source_id,
            event_type=event_type,
            severity=severity,
            started_at=started_at,
            ended_at=ended_at,
            is_open=is_open,
            track_id=track_id,
            identity_id=identity_id,
            face_id=face_id,
            recording_id=recording_id,
            motion_event_id=motion_event_id,
            confidence=confidence,
            label=label,
            message=message,
            snapshot_path=snapshot_path,
            dedup_key=dedup_key,
            event_metadata=metadata or {},
        )
        return self.add(event)

    def find_open(self, source_id: int, dedup_key: str) -> SecurityEvent | None:
        return self.session.execute(
            select(SecurityEvent)
            .where(
                SecurityEvent.source_id == source_id,
                SecurityEvent.dedup_key == dedup_key,
                SecurityEvent.is_open.is_(True),
            )
            .order_by(desc(SecurityEvent.started_at))
            .limit(1)
        ).scalar_one_or_none()

    def touch_open(
        self, event_id: int, *, when: datetime, increment: bool = True, **values: Any
    ) -> None:
        event = self.get(event_id)
        if event is None:
            return
        started = event.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        payload: dict[str, Any] = {
            "duration_seconds": max(0.0, (when - started).total_seconds()),
            **values,
        }
        if increment:
            payload["occurrence_count"] = SecurityEvent.occurrence_count + 1
        self.session.execute(
            update(SecurityEvent).where(SecurityEvent.id == event_id).values(**payload)
        )

    def close(self, event_id: int, *, when: datetime | None = None, **values: Any) -> None:
        when = when or datetime.now(timezone.utc)
        event = self.get(event_id)
        if event is None:
            return
        started = event.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        self.session.execute(
            update(SecurityEvent)
            .where(SecurityEvent.id == event_id)
            .values(
                is_open=False,
                ended_at=when,
                duration_seconds=max(0.0, (when - started).total_seconds()),
                **values,
            )
        )

    def close_orphans(self) -> int:
        now = datetime.now(timezone.utc)
        result = self.session.execute(
            update(SecurityEvent)
            .where(SecurityEvent.is_open.is_(True))
            .values(is_open=False, ended_at=now)
        )
        return int(result.rowcount or 0)

    def list_events(
        self,
        *,
        source_id: int | None = None,
        event_types: list[EventType] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        identity_id: int | None = None,
        track_id: int | None = None,
        severity: EventSeverity | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SecurityEvent]:
        stmt = select(SecurityEvent)
        if source_id is not None:
            stmt = stmt.where(SecurityEvent.source_id == source_id)
        if event_types:
            stmt = stmt.where(SecurityEvent.event_type.in_(event_types))
        if since:
            stmt = stmt.where(SecurityEvent.started_at >= since)
        if until:
            stmt = stmt.where(SecurityEvent.started_at <= until)
        if identity_id is not None:
            stmt = stmt.where(SecurityEvent.identity_id == identity_id)
        if track_id is not None:
            stmt = stmt.where(SecurityEvent.track_id == track_id)
        if severity is not None:
            stmt = stmt.where(SecurityEvent.severity == severity)
        stmt = stmt.order_by(desc(SecurityEvent.started_at)).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def count_by_type(
        self, source_id: int, *, since: datetime | None = None
    ) -> dict[str, int]:
        stmt = select(SecurityEvent.event_type, func.count(SecurityEvent.id)).where(
            SecurityEvent.source_id == source_id
        )
        if since:
            stmt = stmt.where(SecurityEvent.started_at >= since)
        rows = self.session.execute(stmt.group_by(SecurityEvent.event_type)).all()
        return {str(getattr(t, "value", t)): int(c) for t, c in rows}


class AlertRepository(BaseRepository[Alert]):
    model = Alert

    def create(
        self,
        *,
        source_id: int,
        alert_type: AlertType,
        reason: str,
        started_at: datetime | None = None,
        track_id: int | None = None,
        identity_id: int | None = None,
        security_event_id: int | None = None,
        metadata: dict | None = None,
    ) -> Alert:
        return self.add(
            Alert(
                source_id=source_id,
                alert_type=alert_type,
                state=AlertState.ACTIVE,
                reason=reason,
                started_at=started_at or datetime.now(timezone.utc),
                track_id=track_id,
                identity_id=identity_id,
                security_event_id=security_event_id,
                alert_metadata=metadata or {},
            )
        )

    def stop(
        self,
        alert_id: int,
        *,
        stopped_by: str = "operator",
        state: AlertState = AlertState.STOPPED,
        when: datetime | None = None,
    ) -> Alert | None:
        alert = self.get(alert_id)
        if alert is None or alert.state is not AlertState.ACTIVE:
            return alert
        self.session.execute(
            update(Alert)
            .where(Alert.id == alert_id)
            .values(
                state=state,
                stopped_at=when or datetime.now(timezone.utc),
                stopped_by=stopped_by,
                acknowledged=True,
            )
        )
        self.session.flush()
        return self.get(alert_id)

    def active_alerts(self, *, source_id: int | None = None) -> list[Alert]:
        stmt = select(Alert).where(Alert.state == AlertState.ACTIVE)
        if source_id is not None:
            stmt = stmt.where(Alert.source_id == source_id)
        return list(self.session.execute(stmt.order_by(desc(Alert.started_at))).scalars().all())

    def active_for_track(self, source_id: int, track_id: int) -> Alert | None:
        return self.session.execute(
            select(Alert)
            .where(
                Alert.source_id == source_id,
                Alert.track_id == track_id,
                Alert.state == AlertState.ACTIVE,
            )
            .limit(1)
        ).scalar_one_or_none()

    def stop_all(self, *, source_id: int | None = None, stopped_by: str = "operator") -> int:
        stmt = update(Alert).where(Alert.state == AlertState.ACTIVE)
        if source_id is not None:
            stmt = stmt.where(Alert.source_id == source_id)
        result = self.session.execute(
            stmt.values(
                state=AlertState.STOPPED,
                stopped_at=datetime.now(timezone.utc),
                stopped_by=stopped_by,
                acknowledged=True,
            )
        )
        return int(result.rowcount or 0)

    def list_alerts(
        self,
        *,
        source_id: int | None = None,
        state: AlertState | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Alert]:
        stmt = select(Alert)
        if source_id is not None:
            stmt = stmt.where(Alert.source_id == source_id)
        if state is not None:
            stmt = stmt.where(Alert.state == state)
        if since:
            stmt = stmt.where(Alert.started_at >= since)
        return list(
            self.session.execute(
                stmt.order_by(desc(Alert.started_at)).limit(limit).offset(offset)
            )
            .scalars()
            .all()
        )


class SettingsRepository(BaseRepository[SystemSetting]):
    model = SystemSetting

    def get_value(self, key: str, default: Any = None) -> Any:
        row = self.session.get(SystemSetting, key)
        if row is None:
            return default
        return row.value.get("value", default) if isinstance(row.value, dict) else row.value

    def set_value(self, key: str, value: Any, *, description: str | None = None) -> None:
        row = self.session.get(SystemSetting, key)
        payload = {"value": value}
        if row is None:
            self.session.add(
                SystemSetting(key=key, value=payload, description=description)
            )
        else:
            row.value = payload
            if description:
                row.description = description
        self.session.flush()

    def all_values(self) -> dict[str, Any]:
        rows = self.session.execute(select(SystemSetting)).scalars().all()
        return {
            r.key: (r.value.get("value") if isinstance(r.value, dict) else r.value)
            for r in rows
        }
