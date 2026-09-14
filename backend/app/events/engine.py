"""Event emission with deduplication and aggregation.

The rule this module enforces: a *continuing condition* is one row, not one
row per frame. An unknown person standing in front of a camera for a minute
produces a single UNKNOWN_FACE event whose duration grows, and which is
closed when the track ends.

Every persisted event is also published to the WebSocket bus so the UI
reflects it immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.events.bus import get_bus
from app.models.enums import EventSeverity, EventType
from app.models.event import SecurityEvent
from app.repositories.event_repository import EventRepository

log = get_logger(__name__)

# Events that represent a state with a duration rather than an instant.
CONTINUING_EVENTS = frozenset(
    {
        EventType.MOTION_STARTED,
        EventType.UNKNOWN_FACE,
        EventType.PROXIMITY_A_ENTERED,
        EventType.PROXIMITY_B_ENTERED,
        EventType.OBJECT_TRACK_STARTED,
        EventType.ALARM_STARTED,
    }
)

DEFAULT_SEVERITY: dict[EventType, EventSeverity] = {
    EventType.UNKNOWN_FACE: EventSeverity.WARNING,
    EventType.PROXIMITY_B_ENTERED: EventSeverity.NOTICE,
    EventType.ALARM_STARTED: EventSeverity.CRITICAL,
    EventType.ALARM_STOPPED: EventSeverity.NOTICE,
    EventType.TEMPORARY_FAMILIAR_DETECTED: EventSeverity.NOTICE,
    EventType.SOURCE_UNAVAILABLE: EventSeverity.WARNING,
    EventType.RECORDING_INTERRUPTED: EventSeverity.WARNING,
    EventType.IDENTITY_CLASSIFIED: EventSeverity.NOTICE,
    EventType.UNIDENTIFIED_MOVEMENT: EventSeverity.NOTICE,
}


@dataclass(slots=True)
class EventContext:
    """Identifiers attached to every event emitted for one source."""

    source_id: int
    source_uid: str
    recording_id: int | None = None


class EventEmitter:
    """Persists and broadcasts events for a single source.

    Instances are cheap and are created per database transaction; the
    cross-frame dedup state lives in ``DedupCache``, which is owned by the
    long-lived pipeline.
    """

    def __init__(self, session: Session, context: EventContext) -> None:
        self.session = session
        self.context = context
        self.repo = EventRepository(session)
        self.bus = get_bus()

    # ------------------------------------------------------------- emit
    def emit(
        self,
        event_type: EventType,
        *,
        when: datetime | None = None,
        severity: EventSeverity | None = None,
        broadcast: bool = True,
        **fields,
    ) -> SecurityEvent:
        """Record an instantaneous event."""
        when = when or datetime.now(timezone.utc)
        event = self.repo.create(
            source_id=self.context.source_id,
            event_type=event_type,
            started_at=when,
            ended_at=when,
            severity=severity or DEFAULT_SEVERITY.get(event_type, EventSeverity.INFO),
            recording_id=fields.pop("recording_id", self.context.recording_id),
            **fields,
        )
        if broadcast:
            self._broadcast(event)
        return event

    def open(
        self,
        event_type: EventType,
        *,
        dedup_key: str,
        when: datetime | None = None,
        severity: EventSeverity | None = None,
        **fields,
    ) -> SecurityEvent:
        """Open a continuing event, or extend the one already open.

        This is the single place event flooding is prevented.
        """
        when = when or datetime.now(timezone.utc)
        existing = self.repo.find_open(self.context.source_id, dedup_key)
        if existing is not None:
            self.repo.touch_open(existing.id, when=when)
            return existing

        event = self.repo.create(
            source_id=self.context.source_id,
            event_type=event_type,
            started_at=when,
            is_open=True,
            dedup_key=dedup_key,
            severity=severity or DEFAULT_SEVERITY.get(event_type, EventSeverity.INFO),
            recording_id=fields.pop("recording_id", self.context.recording_id),
            **fields,
        )
        self._broadcast(event)
        return event

    def touch(self, event_id: int | None, *, when: datetime | None = None) -> None:
        if event_id is None:
            return
        self.repo.touch_open(event_id, when=when or datetime.now(timezone.utc))

    def close(
        self,
        event_id: int | None,
        *,
        when: datetime | None = None,
        closing_type: EventType | None = None,
        broadcast: bool = True,
        **fields,
    ) -> None:
        """Close an open event and optionally emit its paired closing event."""
        if event_id is None:
            return
        when = when or datetime.now(timezone.utc)
        event = self.repo.get(event_id)
        self.repo.close(event_id, when=when)
        if closing_type is not None and event is not None:
            self.emit(
                closing_type,
                when=when,
                track_id=event.track_id,
                identity_id=event.identity_id,
                label=event.label,
                broadcast=broadcast,
                **fields,
            )

    # -------------------------------------------------------- broadcasting
    def _broadcast(self, event: SecurityEvent) -> None:
        self.bus.emit(
            "event",
            source_uid=self.context.source_uid,
            event={
                "id": event.id,
                "event_type": str(getattr(event.event_type, "value", event.event_type)),
                "severity": str(getattr(event.severity, "value", event.severity)),
                "source_id": event.source_id,
                "track_id": event.track_id,
                "identity_id": event.identity_id,
                "face_id": event.face_id,
                "recording_id": event.recording_id,
                "label": event.label,
                "message": event.message,
                "confidence": event.confidence,
                "started_at": event.started_at.isoformat(),
                "is_open": event.is_open,
                "metadata": event.event_metadata,
            },
        )


class DedupCache:
    """Rate-limits repeated *instantaneous* events for one source.

    Continuing events are handled by ``EventEmitter.open``; this covers things
    like OBJECT_DETECTED where we want a heartbeat, not a flood.
    """

    def __init__(self, *, window_seconds: float = 30.0) -> None:
        self.window = timedelta(seconds=window_seconds)
        self._last: dict[str, datetime] = {}

    def should_emit(self, key: str, *, when: datetime | None = None,
                    window_seconds: float | None = None) -> bool:
        when = when or datetime.now(timezone.utc)
        window = (
            timedelta(seconds=window_seconds) if window_seconds is not None else self.window
        )
        previous = self._last.get(key)
        if previous is not None and (when - previous) < window:
            return False
        self._last[key] = when
        return True

    def forget(self, key: str) -> None:
        self._last.pop(key, None)

    def forget_prefix(self, prefix: str) -> None:
        for key in [k for k in self._last if k.startswith(prefix)]:
            del self._last[key]

    def clear(self) -> None:
        self._last.clear()
