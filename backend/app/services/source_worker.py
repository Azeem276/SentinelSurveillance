"""One capture/analysis thread per video source.

The worker is the only place the four subsystems meet, and it is written so
that a failure in any of them is contained:

  * source cannot open      -> status UNAVAILABLE, worker retries/stops, others run on
  * frame read fails        -> recording session closed safely, source marked ended
  * recording write fails   -> session marked interrupted, capture continues
  * inference throws        -> counted, frame skipped, capture continues

Surveillance is the parent state: no surveillance means no recording and no
intelligence. Intelligence additionally requires both the global switch and
the per-source switch.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.core.logging import (
    RECORDING_INTERRUPTED, SOURCE_STARTED, SOURCE_STOPPED, SOURCE_UNAVAILABLE, get_logger,
)
from app.db.session import session_scope
from app.events.bus import get_bus
from app.events.engine import EventContext, EventEmitter
from app.intelligence.pipeline import IntelligencePipeline, PipelineConfig, build_pipeline
from app.intelligence.proximity.base import CameraCalibration, ProximityZones
from app.intelligence.proximity.estimator import PinholeProximityEstimator
from app.models.enums import EventType, RecordingStatus, SourceStatus, SourceType
from app.recording.engine import RecordingEngine, RecordingSessionWriter
from app.repositories.recording_repository import RecordingRepository
from app.repositories.source_repository import SourceRepository
from app.security.rules import AlertPolicy
from app.services.frame_hub import get_frame_hub
from app.storage.paths import resolve_video
from app.video.base import Frame, SourceUnavailableError, VideoSourceAdapter
from app.video.file_source import FileVideoSource
from app.video.rtsp_source import RTSPCameraSource

log = get_logger(__name__)


def build_source_adapter(record: Any) -> VideoSourceAdapter:
    """Create the right adapter for a persisted source row.

    This function is the *only* place source type maps to implementation.
    Adding a new source kind means adding a branch here, nothing else.
    """
    source_type = SourceType(str(getattr(record.type, "value", record.type)))
    if source_type is SourceType.FILE:
        path = resolve_video(record.uri)
        return FileVideoSource(
            record.uid,
            path,
            name=record.name,
            loop=bool(record.loop_playback),
            realtime=True,
            max_fps=get_settings().target_fps * 2,
        )
    if source_type is SourceType.RTSP:
        return RTSPCameraSource(record.uid, record.uri, name=record.name)
    if source_type is SourceType.WEBCAM:
        # A webcam index is just another OpenCV capture; reuse the file source
        # machinery via an int-like URI handled by RTSP adapter semantics.
        return RTSPCameraSource(record.uid, record.uri, name=record.name)
    raise SourceUnavailableError(f"unsupported source type: {source_type}")


class SourceWorker(threading.Thread):
    """Captures, records and analyses a single source."""

    def __init__(
        self,
        *,
        source_id: int,
        source_uid: str,
        source_name: str,
        intelligence_enabled: bool,
        global_intelligence: bool,
        recording_enabled: bool = True,
        pipeline_factory=build_pipeline,
    ) -> None:
        super().__init__(name=f"source-{source_uid}", daemon=True)
        self.source_id = source_id
        self.source_uid = source_uid
        self.source_name = source_name
        self._stop = threading.Event()
        self._intelligence_enabled = intelligence_enabled
        self._global_intelligence = global_intelligence
        self._recording_enabled = recording_enabled
        self._pipeline_factory = pipeline_factory

        self.adapter: VideoSourceAdapter | None = None
        self.pipeline: IntelligencePipeline | None = None
        self.writer: RecordingSessionWriter | None = None
        self.recording_db_id: int | None = None
        self.status = SourceStatus.IDLE
        self.last_error: str | None = None

        self.hub = get_frame_hub()
        self.bus = get_bus()
        self.recorder = RecordingEngine()
        self._settings = get_settings()
        self._frames_read = 0
        self._started_at: datetime | None = None
        self._last_overlay_push = 0.0

    # ------------------------------------------------------------ control
    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def set_intelligence(self, enabled: bool) -> None:
        self._intelligence_enabled = enabled
        self._emit_state()

    def set_global_intelligence(self, enabled: bool) -> None:
        self._global_intelligence = enabled
        self._emit_state()

    @property
    def intelligence_active(self) -> bool:
        return self._global_intelligence and self._intelligence_enabled

    # ------------------------------------------------------------- status
    def _set_status(self, status: SourceStatus, error: str | None = None) -> None:
        self.status = status
        self.last_error = error
        try:
            with session_scope() as session:
                SourceRepository(session).set_status(self.source_id, status, error=error)
        except Exception as exc:  # pragma: no cover - DB hiccup must not kill capture
            log.warning("source_status_persist_failed", source=self.source_uid, error=str(exc))
        self._emit_state()

    def _emit_state(self) -> None:
        self.bus.emit(
            "source_state",
            source_uid=self.source_uid,
            state={
                "source_id": self.source_id,
                "uid": self.source_uid,
                "status": str(getattr(self.status, "value", self.status)),
                "recording": self.writer is not None and self.writer.is_open,
                "recording_id": self.recording_db_id,
                "intelligence": self.intelligence_active,
                "error": self.last_error,
                "frames": self._frames_read,
                "active_tracks": self.pipeline.active_track_count() if self.pipeline else 0,
            },
        )

    def snapshot(self) -> dict[str, Any]:
        stats = self.pipeline.stats.to_dict() if self.pipeline else {}
        caps = self.adapter.capabilities if self.adapter else None
        return {
            "source_id": self.source_id,
            "uid": self.source_uid,
            "name": self.source_name,
            "status": str(getattr(self.status, "value", self.status)),
            "running": self.is_alive() and not self.stopping,
            "recording": self.writer is not None and self.writer.is_open,
            "recording_id": self.recording_db_id,
            "intelligence": self.intelligence_active,
            "intelligence_enabled": self._intelligence_enabled,
            "frames_read": self._frames_read,
            "active_tracks": self.pipeline.active_track_count() if self.pipeline else 0,
            "motion": self.pipeline.motion.is_active if self.pipeline else False,
            "error": self.last_error,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "capabilities": {
                "width": caps.width, "height": caps.height, "fps": caps.fps,
                "frame_count": caps.frame_count, "is_live": caps.is_live,
            } if caps else None,
            "stats": stats,
        }

    # -------------------------------------------------------------- setup
    def _load_config(self) -> tuple[PipelineConfig, bool, CameraCalibration]:
        with session_scope() as session:
            record = SourceRepository(session).get(self.source_id)
            if record is None:
                raise SourceUnavailableError(f"source {self.source_id} no longer exists")
            settings = self._settings
            zones = ProximityZones(
                recognition_distance=record.proximity_a,
                alarm_distance=record.proximity_b,
            )
            policy = AlertPolicy.from_dict(
                record.alert_policy,
                defaults=AlertPolicy(
                    temporary_familiar=settings.temporary_familiar_alert,
                    unrecognizable_policy=settings.alarm_unrecognizable_policy,
                ),
            )
            config = PipelineConfig(
                source_id=record.id,
                source_uid=record.uid,
                source_name=record.name,
                zones=zones,
                policy=policy,
                recognition_threshold=(
                    record.recognition_threshold or settings.face_recognition_threshold
                ),
                recognition_enabled=record.recognition_enabled,
                detection_interval=settings.detection_interval,
                frame_max_width=settings.frame_max_width,
            )
            calibration = CameraCalibration.from_dict(record.calibration)
            self.source_name = record.name
            return config, bool(record.recording_enabled), calibration

    def reload_config(self) -> None:
        """Apply source setting changes to a running worker."""
        if self.pipeline is None:
            return
        try:
            config, recording_enabled, calibration = self._load_config()
        except Exception as exc:
            log.warning("source_config_reload_failed", source=self.source_uid, error=str(exc))
            return
        self.pipeline.config = config
        self.pipeline.proximity = PinholeProximityEstimator(calibration)
        self._recording_enabled = recording_enabled
        log.info(
            "source_config_reloaded",
            source=self.source_uid,
            proximity_a=config.zones.recognition_distance,
            proximity_b=config.zones.alarm_distance,
            threshold=config.recognition_threshold,
        )

    # ----------------------------------------------------------- recording
    def _start_recording(self, frame: Frame) -> None:
        if not self._recording_enabled or not self._settings.recording_enabled:
            return
        if self.writer is not None and self.writer.is_open:
            return
        caps = self.adapter.capabilities if self.adapter else None
        fps = caps.fps if caps and caps.fps > 0 else self._settings.recording_fps
        try:
            writer = self.recorder.start_session(
                source_uid=self.source_uid,
                width=frame.width,
                height=frame.height,
                fps=min(fps, self._settings.recording_fps * 2),
            )
        except Exception as exc:
            log.error(
                RECORDING_INTERRUPTED, source=self.source_uid, stage="open", error=str(exc)
            )
            self._recording_enabled = False  # do not retry every frame
            return

        self.writer = writer
        with session_scope() as session:
            record = RecordingRepository(session).start(
                source_id=self.source_id,
                file_path=writer.relative_path,
                started_at=writer.started_at,
                fps=writer.fps,
                width=writer.width,
                height=writer.height,
                codec=writer.fourcc,
            )
            self.recording_db_id = record.id
            EventEmitter(
                session, EventContext(self.source_id, self.source_uid, record.id)
            ).emit(
                EventType.RECORDING_STARTED,
                when=writer.started_at,
                recording_id=record.id,
                message=f"Recording started: {writer.relative_path}",
            )
        if self.pipeline is not None:
            self.pipeline.recording_id = self.recording_db_id
        self._emit_state()

    def _stop_recording(self, *, interrupted: bool = False, error: str | None = None) -> None:
        writer = self.writer
        if writer is None:
            return
        self.writer = None
        result = writer.close(interrupted=interrupted, error=error)
        recording_id = self.recording_db_id
        self.recording_db_id = None
        if self.pipeline is not None:
            self.pipeline.recording_id = None

        if recording_id is None:
            return
        status = (
            RecordingStatus.INTERRUPTED
            if (interrupted or result.error)
            else RecordingStatus.COMPLETED
        )
        try:
            with session_scope() as session:
                RecordingRepository(session).finish(
                    recording_id,
                    status=status,
                    frame_count=result.frame_count,
                    file_size_bytes=result.file_size_bytes,
                    ended_at=result.ended_at,
                    error=result.error,
                )
                EventEmitter(
                    session, EventContext(self.source_id, self.source_uid, recording_id)
                ).emit(
                    EventType.RECORDING_INTERRUPTED
                    if status is RecordingStatus.INTERRUPTED
                    else EventType.RECORDING_STOPPED,
                    when=result.ended_at,
                    recording_id=recording_id,
                    message=(
                        f"Recording {status.value.lower()}: {result.relative_path} "
                        f"({result.frame_count} frames)"
                    ),
                    metadata={"frames": result.frame_count, "error": result.error},
                )
        except Exception as exc:  # pragma: no cover
            log.error("recording_finalise_failed", source=self.source_uid, error=str(exc))
        self._emit_state()

    # ---------------------------------------------------------------- run
    def run(self) -> None:  # noqa: PLR0912, PLR0915 - the lifecycle is inherently branchy
        self._started_at = datetime.now(timezone.utc)
        self._set_status(SourceStatus.STARTING)

        try:
            config, recording_enabled, calibration = self._load_config()
            self._recording_enabled = recording_enabled and self._recording_enabled
            with session_scope() as session:
                record = SourceRepository(session).get(self.source_id)
                self.adapter = build_source_adapter(record)
            self.adapter.open()
        except Exception as exc:
            log.error(SOURCE_UNAVAILABLE, source=self.source_uid, error=str(exc))
            self._set_status(SourceStatus.UNAVAILABLE, str(exc))
            self._emit_source_event(EventType.SOURCE_UNAVAILABLE, str(exc))
            return

        self.pipeline = self._pipeline_factory(config, session_factory=_session_factory)
        self.pipeline.proximity = PinholeProximityEstimator(calibration)

        self._set_status(SourceStatus.AVAILABLE)
        log.info(
            SOURCE_STARTED,
            source=self.source_uid,
            type=self.adapter.source_type,
            intelligence=self.intelligence_active,
            recording=self._recording_enabled,
        )
        self._emit_source_event(EventType.SOURCE_AVAILABLE, "Source available")

        target_interval = 1.0 / max(1.0, self._settings.target_fps)
        last_process = 0.0
        last_frame: Frame | None = None

        try:
            while not self._stop.is_set():
                try:
                    frame = self.adapter.read()
                except SourceUnavailableError as exc:
                    log.warning(SOURCE_UNAVAILABLE, source=self.source_uid, error=str(exc))
                    self._stop_recording(interrupted=True, error=str(exc))
                    self._set_status(SourceStatus.UNAVAILABLE, str(exc))
                    self._emit_source_event(EventType.SOURCE_UNAVAILABLE, str(exc))
                    break

                if frame is None:
                    # Clip finished (or live source gave up): close cleanly.
                    self._stop_recording()
                    self._set_status(SourceStatus.ENDED)
                    log.info("source_stream_ended", source=self.source_uid,
                             frames=self._frames_read)
                    break

                last_frame = frame
                self._frames_read += 1

                if self._recording_enabled:
                    self._start_recording(frame)
                    if self.writer is not None:
                        if self.recorder.should_rotate(self.writer):
                            # Rotation closes cleanly and opens a NEW file.
                            self._stop_recording()
                            self._start_recording(frame)
                        if self.writer is not None and not self.writer.write(frame.image):
                            self._stop_recording(
                                interrupted=True, error=self.writer.error if self.writer else None
                            )

                now = time.perf_counter()
                if now - last_process < target_interval:
                    # Frame is still recorded and previewed, just not analysed.
                    self.hub.publish(self.source_uid, frame.image, frame.frame_number)
                    if self.pipeline is not None:
                        self.pipeline.stats.dropped_frames += 1
                    continue
                last_process = now

                analysis = self.pipeline.process(
                    frame, intelligence_enabled=self.intelligence_active
                )
                self.hub.publish(self.source_uid, frame.image, frame.frame_number)
                self._push_overlay(analysis)

        except Exception as exc:  # pragma: no cover - never let a thread die silently
            log.error("source_worker_crashed", source=self.source_uid, error=str(exc),
                      exc_info=True)
            self._stop_recording(interrupted=True, error=str(exc))
            self._set_status(SourceStatus.ERROR, str(exc))
        finally:
            self._shutdown(last_frame)

    def _shutdown(self, last_frame: Frame | None) -> None:
        # Reaching here after an operator stop is a CLEAN close: the session
        # is finalised as COMPLETED. Crash and source-loss paths have already
        # closed their session as INTERRUPTED before unwinding to here.
        self._stop_recording(interrupted=False)
        if self.pipeline is not None:
            try:
                self.pipeline.close_all(last_frame)
            except Exception as exc:  # pragma: no cover
                log.warning("pipeline_close_failed", source=self.source_uid, error=str(exc))
        if self.adapter is not None:
            self.adapter.release()
        self.hub.clear(self.source_uid)
        if self.status not in (SourceStatus.UNAVAILABLE, SourceStatus.ERROR,
                               SourceStatus.ENDED):
            self._set_status(SourceStatus.IDLE)
        log.info(SOURCE_STOPPED, source=self.source_uid, frames=self._frames_read)
        self._emit_state()

    def _emit_source_event(self, event_type: EventType, message: str) -> None:
        try:
            with session_scope() as session:
                EventEmitter(session, EventContext(self.source_id, self.source_uid)).emit(
                    event_type, message=message
                )
        except Exception:  # pragma: no cover
            pass

    def _push_overlay(self, analysis) -> None:
        """Throttled overlay broadcast: the UI does not need 12 Hz JSON."""
        now = time.perf_counter()
        if now - self._last_overlay_push < 0.12:
            return
        self._last_overlay_push = now
        self.bus.emit(
            "frame_analysis",
            source_uid=self.source_uid,
            frame_number=analysis.frame_number,
            timestamp=analysis.timestamp.isoformat(),
            width=analysis.width,
            height=analysis.height,
            motion=analysis.motion,
            objects=analysis.objects,
            intelligence=self.intelligence_active,
            recording=self.writer is not None and self.writer.is_open,
            inference_ms=round(analysis.inference_ms, 2),
        )


def _session_factory():
    from app.db.session import get_session_factory

    return get_session_factory()()
