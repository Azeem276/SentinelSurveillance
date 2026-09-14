"""Audio/visual alert service.

All sound is driven from here, never from a UI component reacting to a raw
detection. The backend owns the alert *state*; the frontend only plays what
the state machine tells it to.

Guarantees:
  * One track produces at most one active alarm, however many frames it spans.
  * A continuous alarm keeps running until an operator stops it.
  * A beep fires once per triggering condition, not once per frame.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.logging import ALARM_STARTED, ALARM_STOPPED, get_logger
from app.events.bus import get_bus
from app.models.enums import AlertState, AlertType
from app.models.event import Alert
from app.repositories.event_repository import AlertRepository

log = get_logger(__name__)


@dataclass(slots=True)
class AlertSnapshot:
    id: int
    source_id: int
    source_uid: str | None
    alert_type: str
    state: str
    reason: str
    started_at: str
    track_id: int | None
    identity_id: int | None
    identity_label: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_uid": self.source_uid,
            "alert_type": self.alert_type,
            "state": self.state,
            "reason": self.reason,
            "started_at": self.started_at,
            "track_id": self.track_id,
            "identity_id": self.identity_id,
            "identity": self.identity_label,
        }


def _snapshot(alert: Alert, source_uid: str | None, label: str | None = None) -> AlertSnapshot:
    return AlertSnapshot(
        id=alert.id,
        source_id=alert.source_id,
        source_uid=source_uid,
        alert_type=str(getattr(alert.alert_type, "value", alert.alert_type)),
        state=str(getattr(alert.state, "value", alert.state)),
        reason=alert.reason,
        started_at=alert.started_at.isoformat(),
        track_id=alert.track_id,
        identity_id=alert.identity_id,
        identity_label=label,
    )


class AlertEngine:
    """Process-wide alert state machine.

    The database is the source of truth for history; this object keeps the
    in-memory latch so a frame-rate pipeline never has to query to know
    whether a track is already alarming.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # (source_id, track_key) -> alert id, for continuous alarms only.
        self._active_by_track: dict[tuple[int, int], int] = {}
        self._active_ids: set[int] = set()
        self.bus = get_bus()

    # ------------------------------------------------------------- raise
    def beep(
        self,
        session: Session,
        *,
        source_id: int,
        source_uid: str,
        reason: str,
        track_key: int | None = None,
        track_id: int | None = None,
        identity_id: int | None = None,
        identity_label: str | None = None,
        security_event_id: int | None = None,
    ) -> Alert:
        """Fire a single beep. Recorded for the audit trail, then auto-cleared."""
        repo = AlertRepository(session)
        alert = repo.create(
            source_id=source_id,
            alert_type=AlertType.BEEP,
            reason=reason,
            track_id=track_id,
            identity_id=identity_id,
            security_event_id=security_event_id,
            metadata={"track_key": track_key, "identity": identity_label},
        )
        # A beep is instantaneous: it has no "stop" semantics.
        repo.stop(alert.id, stopped_by="auto", state=AlertState.AUTO_CLEARED)
        session.flush()
        snapshot = _snapshot(alert, source_uid, identity_label)
        snapshot.state = AlertState.AUTO_CLEARED.value
        self.bus.emit("alert_beep", source_uid=source_uid, alert=snapshot.to_dict())
        log.info("alert_beep", source=source_uid, reason=reason, identity=identity_label)
        return alert

    def start_continuous(
        self,
        session: Session,
        *,
        source_id: int,
        source_uid: str,
        reason: str,
        track_key: int | None = None,
        track_id: int | None = None,
        identity_id: int | None = None,
        identity_label: str | None = None,
        security_event_id: int | None = None,
    ) -> Alert | None:
        """Start a continuous alarm unless this track already has one."""
        key = (source_id, track_key) if track_key is not None else None
        with self._lock:
            if key is not None and key in self._active_by_track:
                return None

            repo = AlertRepository(session)
            alert = repo.create(
                source_id=source_id,
                alert_type=AlertType.CONTINUOUS_ALARM,
                reason=reason,
                track_id=track_id,
                identity_id=identity_id,
                security_event_id=security_event_id,
                metadata={"track_key": track_key, "identity": identity_label},
            )
            session.flush()
            if key is not None:
                self._active_by_track[key] = alert.id
            self._active_ids.add(alert.id)

        self.bus.emit(
            "alarm_started",
            source_uid=source_uid,
            alert=_snapshot(alert, source_uid, identity_label).to_dict(),
        )
        log.warning(
            ALARM_STARTED, source=source_uid, alert_id=alert.id, reason=reason,
            track_key=track_key, identity=identity_label,
        )
        return alert

    # -------------------------------------------------------------- stop
    def stop(
        self,
        session: Session,
        alert_id: int,
        *,
        source_uid: str | None = None,
        stopped_by: str = "operator",
    ) -> Alert | None:
        repo = AlertRepository(session)
        alert = repo.stop(alert_id, stopped_by=stopped_by)
        if alert is None:
            return None
        with self._lock:
            self._active_ids.discard(alert_id)
            for key, value in list(self._active_by_track.items()):
                if value == alert_id:
                    del self._active_by_track[key]
        self.bus.emit(
            "alarm_stopped",
            source_uid=source_uid,
            alert={"id": alert_id, "stopped_by": stopped_by,
                   "stopped_at": datetime.now(timezone.utc).isoformat()},
        )
        log.info(ALARM_STOPPED, alert_id=alert_id, stopped_by=stopped_by, source=source_uid)
        return alert

    def stop_all(
        self, session: Session, *, source_id: int | None = None, stopped_by: str = "operator"
    ) -> int:
        repo = AlertRepository(session)
        count = repo.stop_all(source_id=source_id, stopped_by=stopped_by)
        with self._lock:
            if source_id is None:
                self._active_by_track.clear()
                self._active_ids.clear()
            else:
                for key in [k for k in self._active_by_track if k[0] == source_id]:
                    self._active_ids.discard(self._active_by_track.pop(key))
        self.bus.emit("alarms_cleared", source_uid=None, count=count, stopped_by=stopped_by)
        log.info(ALARM_STOPPED, count=count, stopped_by=stopped_by, source_id=source_id)
        return count

    def release_track(self, source_id: int, track_key: int) -> int | None:
        """Forget the latch when a track ends.

        The alarm itself stays ACTIVE until an operator stops it - that is the
        specified behaviour - but the track no longer holds the latch.
        """
        with self._lock:
            return self._active_by_track.pop((source_id, track_key), None)

    # ------------------------------------------------------------ queries
    def is_alarming(self, source_id: int, track_key: int) -> bool:
        with self._lock:
            return (source_id, track_key) in self._active_by_track

    def alert_id_for_track(self, source_id: int, track_key: int) -> int | None:
        with self._lock:
            return self._active_by_track.get((source_id, track_key))

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active_ids)

    def hydrate(self, session: Session) -> None:
        """Rebuild the in-memory latch from the database after a restart."""
        repo = AlertRepository(session)
        with self._lock:
            self._active_by_track.clear()
            self._active_ids.clear()
            for alert in repo.active_alerts():
                self._active_ids.add(alert.id)
                track_key = (alert.alert_metadata or {}).get("track_key")
                if track_key is not None and alert.alert_type is AlertType.CONTINUOUS_ALARM:
                    self._active_by_track[(alert.source_id, int(track_key))] = alert.id

    def reset(self) -> None:
        with self._lock:
            self._active_by_track.clear()
            self._active_ids.clear()


_engine: AlertEngine | None = None
_engine_lock = threading.Lock()


def get_alert_engine() -> AlertEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AlertEngine()
    return _engine


def reset_alert_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
