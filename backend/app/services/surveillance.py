"""Surveillance manager: the global state machine.

    Surveillance OFF -> nothing captures, nothing records, no intelligence
    Surveillance ON  -> workers run, recording starts, intelligence *may* run
    Intelligence     -> requires surveillance ON *and* the global switch ON
                        *and* the per-source switch ON

State is persisted in ``system_settings`` so a restart restores the
operator's intent rather than silently coming back disabled.
"""
from __future__ import annotations

import threading
from typing import Any

from app.core.exceptions import NotFoundError, PreconditionError
from app.core.logging import (
    INTELLIGENCE_STARTED, INTELLIGENCE_STOPPED, get_logger,
)
from app.db.session import session_scope
from app.events.bus import get_bus
from app.models.enums import SourceStatus
from app.repositories.event_repository import SettingsRepository
from app.repositories.source_repository import SourceRepository
from app.services.frame_hub import get_frame_hub
from app.services.source_worker import SourceWorker

log = get_logger(__name__)

KEY_SURVEILLANCE = "surveillance_active"
KEY_INTELLIGENCE = "intelligence_active"


class SurveillanceManager:
    """Owns the worker threads. One instance per process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[int, SourceWorker] = {}
        self._surveillance_active = False
        self._intelligence_active = False
        self.bus = get_bus()
        self.hub = get_frame_hub()

    # --------------------------------------------------------------- state
    @property
    def surveillance_active(self) -> bool:
        return self._surveillance_active

    @property
    def intelligence_active(self) -> bool:
        return self._surveillance_active and self._intelligence_active

    def hydrate(self) -> None:
        """Restore persisted intent at startup (without auto-starting)."""
        with session_scope() as session:
            repo = SettingsRepository(session)
            self._intelligence_active = bool(repo.get_value(KEY_INTELLIGENCE, True))
            wanted = bool(repo.get_value(KEY_SURVEILLANCE, False))
        if wanted:
            log.info("surveillance_autostart", reason="persisted state was ON")
            try:
                self.start_surveillance()
            except Exception as exc:  # pragma: no cover
                log.error("surveillance_autostart_failed", error=str(exc))

    def _persist(self, key: str, value: Any) -> None:
        try:
            with session_scope() as session:
                SettingsRepository(session).set_value(key, value)
        except Exception as exc:  # pragma: no cover
            log.warning("settings_persist_failed", key=key, error=str(exc))

    # -------------------------------------------------------- surveillance
    def start_surveillance(self) -> dict[str, Any]:
        with self._lock:
            self._surveillance_active = True
            self._persist(KEY_SURVEILLANCE, True)

            with session_scope() as session:
                sources = SourceRepository(session).list_active_for_surveillance()
                specs = [
                    (s.id, s.uid, s.name, bool(s.intelligence_enabled),
                     bool(s.recording_enabled))
                    for s in sources
                ]

            started: list[str] = []
            for source_id, uid, name, intel, recording in specs:
                if self._start_worker(
                    source_id, uid, name, intelligence=intel, recording=recording
                ):
                    started.append(uid)

        log.info("surveillance_started", sources=len(started))
        self._broadcast_system_state()
        return {"surveillance": True, "started_sources": started}

    def stop_surveillance(self) -> dict[str, Any]:
        with self._lock:
            self._surveillance_active = False
            self._persist(KEY_SURVEILLANCE, False)
            stopped = self._stop_all_workers()
        log.info("surveillance_stopped", sources=stopped)
        self._broadcast_system_state()
        return {"surveillance": False, "stopped_sources": stopped}

    # -------------------------------------------------------- intelligence
    def start_intelligence(self) -> dict[str, Any]:
        if not self._surveillance_active:
            raise PreconditionError(
                "Intelligence cannot run while surveillance is off. "
                "Activate surveillance first."
            )
        with self._lock:
            self._intelligence_active = True
            self._persist(KEY_INTELLIGENCE, True)
            for worker in self._workers.values():
                worker.set_global_intelligence(True)
        log.info(INTELLIGENCE_STARTED, scope="global")
        self._broadcast_system_state()
        return {"intelligence": True}

    def stop_intelligence(self) -> dict[str, Any]:
        with self._lock:
            self._intelligence_active = False
            self._persist(KEY_INTELLIGENCE, False)
            for worker in self._workers.values():
                worker.set_global_intelligence(False)
        log.info(INTELLIGENCE_STOPPED, scope="global")
        self._broadcast_system_state()
        return {"intelligence": False}

    def set_source_intelligence(self, source_id: int, enabled: bool) -> dict[str, Any]:
        with session_scope() as session:
            repo = SourceRepository(session)
            source = repo.get(source_id)
            if source is None:
                raise NotFoundError(f"source {source_id} not found")
            repo.set_intelligence(source_id, enabled)
            uid = source.uid
        with self._lock:
            worker = self._workers.get(source_id)
            if worker is not None:
                worker.set_intelligence(enabled)
        log.info(
            INTELLIGENCE_STARTED if enabled else INTELLIGENCE_STOPPED,
            scope="source", source=uid,
        )
        self._broadcast_system_state()
        return {
            "source_id": source_id,
            "intelligence_enabled": enabled,
            "effective": enabled and self.intelligence_active,
        }

    def set_all_source_intelligence(self, enabled: bool) -> int:
        """Global control also flips every per-source switch (section 10)."""
        with session_scope() as session:
            count = SourceRepository(session).set_intelligence_all(enabled)
        with self._lock:
            for worker in self._workers.values():
                worker.set_intelligence(enabled)
        self._broadcast_system_state()
        return count

    # -------------------------------------------------------------- source
    def _start_worker(
        self,
        source_id: int,
        uid: str,
        name: str,
        *,
        intelligence: bool,
        recording: bool = True,
    ) -> bool:
        existing = self._workers.get(source_id)
        if existing is not None and existing.is_alive():
            return False
        worker = SourceWorker(
            source_id=source_id,
            source_uid=uid,
            source_name=name,
            intelligence_enabled=intelligence,
            global_intelligence=self._intelligence_active,
            recording_enabled=recording,
        )
        self._workers[source_id] = worker
        worker.start()
        return True

    def start_source(self, source_id: int) -> dict[str, Any]:
        if not self._surveillance_active:
            raise PreconditionError("Activate surveillance before starting a source")
        with session_scope() as session:
            source = SourceRepository(session).get(source_id)
            if source is None:
                raise NotFoundError(f"source {source_id} not found")
            if not source.enabled:
                raise PreconditionError(f"source {source.uid} is disabled")
            spec = (source.id, source.uid, source.name,
                    bool(source.intelligence_enabled), bool(source.recording_enabled))
        with self._lock:
            started = self._start_worker(
                spec[0], spec[1], spec[2], intelligence=spec[3], recording=spec[4]
            )
        self._broadcast_system_state()
        return {"source_id": source_id, "started": started}

    def stop_source(self, source_id: int, *, timeout: float = 8.0) -> dict[str, Any]:
        with self._lock:
            worker = self._workers.pop(source_id, None)
        if worker is None:
            return {"source_id": source_id, "stopped": False}
        worker.stop()
        worker.join(timeout=timeout)
        self._broadcast_system_state()
        return {"source_id": source_id, "stopped": True, "clean": not worker.is_alive()}

    def restart_source(self, source_id: int) -> dict[str, Any]:
        self.stop_source(source_id)
        if not self._surveillance_active:
            return {"source_id": source_id, "restarted": False}
        return self.start_source(source_id)

    def reload_source(self, source_id: int) -> None:
        """Apply changed settings to a running worker."""
        with self._lock:
            worker = self._workers.get(source_id)
        if worker is not None:
            worker.reload_config()

    def _stop_all_workers(self, *, timeout: float = 8.0) -> int:
        workers = list(self._workers.values())
        self._workers.clear()
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=timeout)
        self.hub.clear()
        return len(workers)

    def shutdown(self) -> None:
        with self._lock:
            self._stop_all_workers(timeout=5.0)
        try:
            with session_scope() as session:
                SourceRepository(session).reset_all_statuses()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------ identity
    def invalidate_identity(self, identity_id: int) -> None:
        """Propagate an identity deletion/expiry into every live pipeline."""
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            if worker.pipeline is not None:
                worker.pipeline.invalidate_identity(identity_id)

    # ------------------------------------------------------------- queries
    def worker(self, source_id: int) -> SourceWorker | None:
        with self._lock:
            return self._workers.get(source_id)

    def is_running(self, source_id: int) -> bool:
        worker = self.worker(source_id)
        return worker is not None and worker.is_alive() and not worker.stopping

    def runtime_status(self, source_id: int) -> dict[str, Any]:
        worker = self.worker(source_id)
        if worker is None:
            return {
                "running": False,
                "status": SourceStatus.IDLE.value,
                "recording": False,
                "intelligence": False,
                "motion": False,
                "active_tracks": 0,
            }
        return worker.snapshot()

    def all_runtime_status(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            workers = dict(self._workers)
        return {sid: w.snapshot() for sid, w in workers.items()}

    def system_state(self) -> dict[str, Any]:
        with self._lock:
            workers = list(self._workers.values())
        running = [w for w in workers if w.is_alive()]
        return {
            "surveillance_active": self._surveillance_active,
            "intelligence_active": self.intelligence_active,
            "intelligence_switch": self._intelligence_active,
            "running_sources": len(running),
            "recording_sources": sum(
                1 for w in running if w.writer is not None and w.writer.is_open
            ),
            "active_tracks": sum(
                w.pipeline.active_track_count() for w in running if w.pipeline
            ),
        }

    def _broadcast_system_state(self) -> None:
        self.bus.emit("system_state", source_uid=None, state=self.system_state())


_manager: SurveillanceManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> SurveillanceManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = SurveillanceManager()
    return _manager


def reset_manager() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.shutdown()
        _manager = None
