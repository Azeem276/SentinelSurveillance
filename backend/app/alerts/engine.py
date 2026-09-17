"""Audio/visual alert service.

All sound is driven from here, never from a UI component reacting to a raw
detection. The backend owns the alert *state*; the frontend only plays what
the state machine tells it to.

Guarantees:
  * One track produces at most one active alarm, however many frames it spans.
  * A timed alarm clears itself after its duration; a continuous alarm keeps
    running until an operator stops it.
  * An alarm raised against a person who is subsequently *recognised* is
    cancelled automatically - the system corrects itself rather than making
    the operator silence its own mistake.
  * A beep fires once per triggering condition, not once per frame.

Muting
------
``sound_enabled`` controls audio only. Alerts are still raised, recorded,
broadcast and shown; the UI simply does not play them. Muting is therefore
safe to leave on during a demo without turning the security engine off, and
it is persisted so it survives a restart.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import ALARM_STARTED, ALARM_STOPPED, get_logger
from app.events.bus import get_bus
from app.models.enums import AlertState, AlertType
from app.models.event import Alert
from app.repositories.event_repository import AlertRepository, SettingsRepository

log = get_logger(__name__)

KEY_ALARM_SOUND = "alarm_sound_enabled"


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
    expires_at: str | None = None
    sound: bool = True

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
            "expires_at": self.expires_at,
            "sound": self.sound,
        }


def _snapshot(
    alert: Alert,
    source_uid: str | None,
    label: str | None = None,
    *,
    sound: bool = True,
) -> AlertSnapshot:
    metadata = alert.alert_metadata or {}
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
        expires_at=metadata.get("expires_at"),
        sound=sound,
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
        # alert id -> when a timed alarm should clear itself.
        self._expiry: dict[int, datetime] = {}
        self._sound_enabled = get_settings().alarm_sound_enabled
        self.bus = get_bus()

    # -------------------------------------------------------------- sound
    @property
    def sound_enabled(self) -> bool:
        with self._lock:
            return self._sound_enabled

    def set_sound_enabled(
        self, enabled: bool, *, session: Session | None = None
    ) -> bool:
        """Turn audible alerts on or off without touching anything else.

        Detection, recognition, events, alert records and the visual alarm
        state are all unaffected - this is a speaker switch, not a kill switch.
        """
        with self._lock:
            self._sound_enabled = bool(enabled)
        if session is not None:
            SettingsRepository(session).set_value(
                KEY_ALARM_SOUND,
                bool(enabled),
                description="Play audible alarms and beeps in the console",
            )
        self.bus.emit("alarm_sound", source_uid=None, enabled=bool(enabled))
        log.info("alarm_sound_changed", enabled=bool(enabled))
        return bool(enabled)

    def hydrate_sound(self, session: Session) -> bool:
        """Restore the operator's persisted mute preference at startup."""
        stored = SettingsRepository(session).get_value(KEY_ALARM_SOUND, None)
        if stored is not None:
            with self._lock:
                self._sound_enabled = bool(stored)
        return self.sound_enabled

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
        snapshot = _snapshot(alert, source_uid, identity_label, sound=self.sound_enabled)
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
        duration_seconds: int | None = None,
    ) -> Alert | None:
        """Start an alarm unless this track already has one.

        ``duration_seconds`` makes it a *timed* alarm: it is recorded exactly
        like a continuous one, but :meth:`sweep_expired` clears it once the
        window elapses, so a first-time unknown does not leave a siren running
        until somebody walks over to the console. Omit it for the escalated,
        operator-cleared behaviour.
        """
        key = (source_id, track_key) if track_key is not None else None
        expires_at: datetime | None = None
        if duration_seconds and duration_seconds > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)

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
                metadata={
                    "track_key": track_key,
                    "identity": identity_label,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "duration_seconds": duration_seconds,
                },
            )
            session.flush()
            if key is not None:
                self._active_by_track[key] = alert.id
            self._active_ids.add(alert.id)
            if expires_at is not None:
                self._expiry[alert.id] = expires_at

        self.bus.emit(
            "alarm_started",
            source_uid=source_uid,
            alert=_snapshot(
                alert, source_uid, identity_label, sound=self.sound_enabled
            ).to_dict(),
        )
        log.warning(
            ALARM_STARTED, source=source_uid, alert_id=alert.id, reason=reason,
            track_key=track_key, identity=identity_label,
            duration_seconds=duration_seconds,
        )
        return alert

    def start_timed(self, session: Session, *, duration_seconds: int, **kwargs) -> Alert | None:
        """Convenience wrapper: a self-clearing alarm."""
        return self.start_continuous(
            session, duration_seconds=duration_seconds, **kwargs
        )

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
            self._expiry.pop(alert_id, None)
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
                self._expiry.clear()
            else:
                for key in [k for k in self._active_by_track if k[0] == source_id]:
                    alert_id = self._active_by_track.pop(key)
                    self._active_ids.discard(alert_id)
                    self._expiry.pop(alert_id, None)
        self.bus.emit("alarms_cleared", source_uid=None, count=count, stopped_by=stopped_by)
        log.info(ALARM_STOPPED, count=count, stopped_by=stopped_by, source_id=source_id)
        return count

    def stop_for_track(
        self,
        session: Session,
        *,
        source_id: int,
        track_key: int,
        source_uid: str | None = None,
        reason: str = "identity_resolved",
    ) -> Alert | None:
        """Cancel the alarm a track is holding, because it was wrong.

        Called when a person who tripped the alarm is subsequently recognised.
        Leaving the siren running after the system has changed its own mind
        trains operators to ignore it, which is the most expensive failure a
        security product can have.
        """
        with self._lock:
            alert_id = self._active_by_track.get((source_id, track_key))
        if alert_id is None:
            return None
        return self.stop(session, alert_id, source_uid=source_uid, stopped_by=reason)

    def sweep_expired(self, session: Session, *, now: datetime | None = None) -> int:
        """Clear timed alarms whose window has elapsed. Returns how many."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            due = [aid for aid, when in self._expiry.items() if when <= now]
        cleared = 0
        for alert_id in due:
            if self.stop(session, alert_id, stopped_by="auto_timeout") is not None:
                cleared += 1
            else:
                with self._lock:
                    self._expiry.pop(alert_id, None)
        return cleared

    def release_track(self, source_id: int, track_key: int) -> int | None:
        """Forget the latch when a track ends.

        A continuous alarm stays ACTIVE until an operator stops it - that is
        the specified behaviour - but the track no longer holds the latch. A
        timed alarm still expires on its own schedule.
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
            self._expiry.clear()
            for alert in repo.active_alerts():
                self._active_ids.add(alert.id)
                metadata = alert.alert_metadata or {}
                track_key = metadata.get("track_key")
                if track_key is not None and alert.alert_type is AlertType.CONTINUOUS_ALARM:
                    self._active_by_track[(alert.source_id, int(track_key))] = alert.id
                # Timed alarms that outlived the process still expire, and one
                # already past its window is cleared by the next sweep.
                expires = metadata.get("expires_at")
                if expires:
                    try:
                        when = datetime.fromisoformat(str(expires))
                        if when.tzinfo is None:
                            when = when.replace(tzinfo=timezone.utc)
                        self._expiry[alert.id] = when
                    except ValueError:  # pragma: no cover - defensive
                        pass
        self.hydrate_sound(session)

    def reset(self) -> None:
        with self._lock:
            self._active_by_track.clear()
            self._active_ids.clear()
            self._expiry.clear()


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
