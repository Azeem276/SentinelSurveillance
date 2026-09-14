"""Per-source intelligence pipeline.

Order of operations for one processed frame:

    motion -> detection -> tracking -> proximity
           -> (person && inside Proximity A) face detection
           -> face quality gate -> embedding -> recognition
           -> identity classification -> proximity events
           -> security rules -> alerts

Cost control (section 49 of the brief):
  * the detector runs every ``detection_interval`` frames; tracking carries
    identity between detector runs,
  * face work happens only for people inside Proximity A,
  * a recognised track is not re-recognised until its cooldown elapses.

The pipeline owns no database session. It opens short transactions only when
something worth persisting happened, which keeps the frame loop fast.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.alerts.engine import AlertEngine, get_alert_engine
from app.core.config import get_settings
from app.core.logging import DETECTION_ERROR, FACE_RECOGNITION_ERROR, get_logger
from app.events.bus import get_bus
from app.events.engine import DedupCache, EventContext, EventEmitter
from app.intelligence.detection.base import ObjectDetector
from app.intelligence.face.base import FaceDetector, FaceEmbedder, FaceQualityAssessor
from app.intelligence.face.index import FaceRecognitionIndex, get_index
from app.intelligence.motion.detector import MotionDetector
from app.intelligence.proximity.base import ProximityEstimator, ProximityZones
from app.intelligence.track_state import TrackState, new_track_state
from app.intelligence.tracking.base import ObjectTracker
from app.intelligence.types import (
    BBox, DetectedFace, FrameAnalysis, ObjectDetection, UNIDENTIFIED_MOVEMENT,
)
from app.models.detection import Detection
from app.models.enums import (
    EventSeverity, EventType, ProximityZone, RecognitionState, TrackStatus,
)
from app.models.identity import Face
from app.repositories.detection_repository import (
    DetectionRepository, MotionRepository, TrackRepository,
)
from app.repositories.identity_repository import FaceRepository, IdentityRepository
from app.security.rules import (
    AlertPolicy, SecurityAction, evaluate_proximity_b, evaluate_recognition,
    should_attempt_face_recognition,
)
from app.storage.paths import ensure_parent, face_root, relative_to_root, slugify
from app.video.base import Frame

log = get_logger(__name__)

SessionFactory = Callable[[], Session]

# A person box is cropped with a little context before face detection, and
# only its upper portion is searched - faces are not in the legs.
FACE_SEARCH_TOP_FRACTION = 0.55
FACE_CROP_PADDING = 0.08


@dataclass
class PipelineConfig:
    """Everything the pipeline needs to know about one source."""

    source_id: int
    source_uid: str
    source_name: str
    zones: ProximityZones
    policy: AlertPolicy
    recognition_threshold: float
    recognition_enabled: bool = True
    detection_interval: int = 3
    frame_max_width: int = 960
    persist_detections: bool = True
    store_face_crops: bool = True


@dataclass
class PipelineStats:
    frames_seen: int = 0
    frames_processed: int = 0
    detector_runs: int = 0
    face_detections: int = 0
    recognitions: int = 0
    dropped_frames: int = 0
    last_detection_ms: float = 0.0
    last_face_ms: float = 0.0
    last_total_ms: float = 0.0
    avg_detection_ms: float = 0.0
    avg_face_ms: float = 0.0
    avg_total_ms: float = 0.0
    errors: int = 0

    def observe(self, *, total_ms: float, detection_ms: float, face_ms: float) -> None:
        self.last_total_ms = total_ms
        self.last_detection_ms = detection_ms
        self.last_face_ms = face_ms
        n = max(1, self.frames_processed)
        # Exponential moving average keeps this O(1) and recency-weighted.
        alpha = 2.0 / min(n + 1, 60)
        self.avg_total_ms = (1 - alpha) * self.avg_total_ms + alpha * total_ms
        self.avg_detection_ms = (1 - alpha) * self.avg_detection_ms + alpha * detection_ms
        self.avg_face_ms = (1 - alpha) * self.avg_face_ms + alpha * face_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_seen": self.frames_seen,
            "frames_processed": self.frames_processed,
            "detector_runs": self.detector_runs,
            "face_detections": self.face_detections,
            "recognitions": self.recognitions,
            "dropped_frames": self.dropped_frames,
            "detection_latency_ms": round(self.avg_detection_ms, 2),
            "face_latency_ms": round(self.avg_face_ms, 2),
            "inference_latency_ms": round(self.avg_total_ms, 2),
            "last_inference_ms": round(self.last_total_ms, 2),
            "errors": self.errors,
        }


class IntelligencePipeline:
    """Stateful per-source analysis. One instance per running source."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        detector: ObjectDetector,
        tracker: ObjectTracker,
        motion: MotionDetector,
        proximity: ProximityEstimator,
        face_detector: FaceDetector | None = None,
        face_quality: FaceQualityAssessor | None = None,
        face_embedder: FaceEmbedder | None = None,
        recognition_index: FaceRecognitionIndex | None = None,
        session_factory: SessionFactory | None = None,
        alert_engine: AlertEngine | None = None,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.motion = motion
        self.proximity = proximity
        self.face_detector = face_detector
        self.face_quality = face_quality
        self.face_embedder = face_embedder
        self.index = recognition_index or get_index()
        self.session_factory = session_factory
        self.alerts = alert_engine or get_alert_engine()
        self.bus = get_bus()

        self.stats = PipelineStats()
        self.tracks: dict[int, TrackState] = {}
        self.dedup = DedupCache(window_seconds=30.0)

        self._recording_id: int | None = None
        self._motion_event_id: int | None = None
        self._motion_open_event_id: int | None = None
        self._motion_peak = 0.0
        self._motion_started_frame = 0
        self._last_detections: list[ObjectDetection] = []
        self._frames_since_detector = 10_000
        self._last_overlay: list[dict] = []

    # ------------------------------------------------------------ context
    @property
    def recording_id(self) -> int | None:
        return self._recording_id

    @recording_id.setter
    def recording_id(self, value: int | None) -> None:
        self._recording_id = value

    def _event_context(self) -> EventContext:
        return EventContext(
            source_id=self.config.source_id,
            source_uid=self.config.source_uid,
            recording_id=self._recording_id,
        )

    @contextmanager
    def _session(self):
        """Short write transaction; failures never break the frame loop."""
        if self.session_factory is None:
            yield None
            return
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            self.stats.errors += 1
            log.error("pipeline_persist_failed", source=self.config.source_uid, error=str(exc))
        finally:
            session.close()

    def update_config(self, **changes) -> None:
        for key, value in changes.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    # -------------------------------------------------------------- frame
    def process(self, frame: Frame, *, intelligence_enabled: bool = True) -> FrameAnalysis:
        """Analyse one frame. Never raises: a bad frame must not kill a source."""
        started = time.perf_counter()
        self.stats.frames_seen += 1
        analysis = FrameAnalysis(
            source_uid=self.config.source_uid,
            source_id=self.config.source_id,
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
        )

        try:
            image, scale = self._prepare(frame.image)

            motion_result = self.motion.process(image)
            analysis.motion = motion_result.is_motion
            analysis.motion_area_ratio = motion_result.area_ratio
            self._handle_motion(motion_result.is_motion, motion_result.area_ratio, frame)

            if not intelligence_enabled:
                self._finish(analysis, started, 0.0, 0.0)
                return analysis

            detection_ms = 0.0
            detections = self._last_detections
            if self._frames_since_detector >= max(1, self.config.detection_interval):
                t0 = time.perf_counter()
                detections = self._run_detector(image)
                detection_ms = (time.perf_counter() - t0) * 1000.0
                self._last_detections = detections
                self._frames_since_detector = 0
                self.stats.detector_runs += 1
                analysis.ran_detector = True
            else:
                self._frames_since_detector += 1

            tracked = self.tracker.update(detections)
            face_ms = self._update_tracks(tracked, image, frame, scale)
            self._retire_tracks(frame)

            analysis.objects = self._last_overlay
            self.stats.frames_processed += 1
            self._finish(analysis, started, detection_ms, face_ms)
            return analysis

        except Exception as exc:  # pragma: no cover - safety net
            self.stats.errors += 1
            log.error(
                DETECTION_ERROR, source=self.config.source_uid, error=str(exc), exc_info=True
            )
            self._finish(analysis, started, 0.0, 0.0)
            return analysis

    def _finish(
        self, analysis: FrameAnalysis, started: float, detection_ms: float, face_ms: float
    ) -> None:
        total = (time.perf_counter() - started) * 1000.0
        analysis.inference_ms = total
        analysis.detection_ms = detection_ms
        analysis.face_ms = face_ms
        self.stats.observe(total_ms=total, detection_ms=detection_ms, face_ms=face_ms)

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale for inference; ``scale`` maps back to original pixels."""
        max_width = self.config.frame_max_width
        h, w = image.shape[:2]
        if max_width and w > max_width:
            factor = max_width / float(w)
            resized = cv2.resize(image, (max_width, max(1, int(h * factor))))
            return resized, 1.0 / factor
        return image, 1.0

    def _run_detector(self, image: np.ndarray) -> list[ObjectDetection]:
        try:
            if self.detector.supports_native_tracking:
                return self.detector.track(image)
            return self.detector.detect(image)
        except Exception as exc:
            self.stats.errors += 1
            log.error(DETECTION_ERROR, source=self.config.source_uid, error=str(exc))
            return []

    # ------------------------------------------------------------- motion
    def _handle_motion(self, is_motion: bool, area_ratio: float, frame: Frame) -> None:
        """Open/close exactly one motion episode, never one per frame."""
        if is_motion:
            self._motion_peak = max(self._motion_peak, area_ratio)
            if self._motion_event_id is None:
                self._motion_started_frame = frame.frame_number
                with self._session() as session:
                    if session is None:
                        return
                    motion_repo = MotionRepository(session)
                    record = motion_repo.start(
                        source_id=self.config.source_id,
                        started_at=frame.timestamp,
                        frame_start=frame.frame_number,
                        recording_id=self._recording_id,
                        area_ratio=area_ratio,
                    )
                    self._motion_event_id = record.id
                    emitter = EventEmitter(session, self._event_context())
                    event = emitter.open(
                        EventType.MOTION_STARTED,
                        dedup_key="motion",
                        when=frame.timestamp,
                        motion_event_id=record.id,
                        message="Motion started",
                        metadata={"area_ratio": round(area_ratio, 5)},
                    )
                    self._motion_open_event_id = event.id
                self.bus.emit(
                    "motion", source_uid=self.config.source_uid, active=True,
                    area_ratio=round(area_ratio, 5),
                )
        elif self._motion_event_id is not None:
            motion_id = self._motion_event_id
            open_event_id = self._motion_open_event_id
            peak = self._motion_peak
            self._motion_event_id = None
            self._motion_open_event_id = None
            self._motion_peak = 0.0
            with self._session() as session:
                if session is None:
                    return
                MotionRepository(session).end(
                    motion_id,
                    ended_at=frame.timestamp,
                    frame_end=frame.frame_number,
                    peak_area_ratio=peak,
                )
                emitter = EventEmitter(session, self._event_context())
                emitter.close(
                    open_event_id,
                    when=frame.timestamp,
                    closing_type=EventType.MOTION_ENDED,
                    motion_event_id=motion_id,
                    message="Motion ended",
                )
            self.bus.emit("motion", source_uid=self.config.source_uid, active=False)

    # ------------------------------------------------------------- tracks
    def _update_tracks(
        self, tracked: list, image: np.ndarray, frame: Frame, scale: float
    ) -> float:
        face_ms_total = 0.0
        overlay: list[dict] = []
        pending_detections: list[Detection] = []

        for obj in tracked:
            state = self.tracks.get(obj.track_key)
            is_new = state is None
            if state is None:
                state = new_track_state(
                    track_key=obj.track_key,
                    source_id=self.config.source_id,
                    object_class=obj.object_class,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp,
                )
                self.tracks[obj.track_key] = state

            # Overlay/DB geometry is always in original frame coordinates.
            full_bbox = obj.bbox.scaled(scale).clipped(frame.width, frame.height)
            state.observe(
                bbox=full_bbox,
                confidence=obj.confidence,
                object_class=obj.object_class,
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
            )

            proximity = self.proximity.estimate(
                full_bbox,
                frame_width=frame.width,
                frame_height=frame.height,
                object_class=obj.object_class,
                zones=self.config.zones,
            )
            previous_zone, new_zone = state.update_proximity(proximity)

            if is_new:
                self._persist_new_track(state, frame)

            if previous_zone is not new_zone:
                self._handle_zone_transition(state, previous_zone, new_zone, frame)

            # Face work: people inside the recognition zone only.
            if should_attempt_face_recognition(
                object_class=obj.object_class,
                in_recognition_zone=state.in_recognition_zone,
                recognition_enabled=self.config.recognition_enabled,
            ) and state.should_attempt_recognition():
                t0 = time.perf_counter()
                self._process_face(state, image, frame, scale)
                face_ms_total += (time.perf_counter() - t0) * 1000.0
            elif obj.object_class == "person" and not state.in_recognition_zone:
                # Too far to recognise: explicitly *pending*, never unfamiliar.
                if state.recognition_state in (
                    RecognitionState.NO_FACE,
                    RecognitionState.UNKNOWN_PENDING_RECOGNITION,
                ):
                    state.recognition_state = RecognitionState.UNKNOWN_PENDING_RECOGNITION

            self._apply_security(state, frame)

            if self.config.persist_detections and (
                self._frames_since_detector == 0 or is_new
            ):
                pending_detections.append(
                    Detection(
                        source_id=self.config.source_id,
                        track_id=state.db_track_id,
                        recording_id=self._recording_id,
                        timestamp=frame.timestamp,
                        frame_number=frame.frame_number,
                        object_class=state.object_class,
                        confidence=state.confidence,
                        bbox_x1=full_bbox.x1,
                        bbox_y1=full_bbox.y1,
                        bbox_x2=full_bbox.x2,
                        bbox_y2=full_bbox.y2,
                        distance_m=proximity.distance_m,
                        proximity_zone=proximity.zone,
                    )
                )

            overlay.append(state.to_overlay())

        if pending_detections:
            with self._session() as session:
                if session is not None:
                    DetectionRepository(session).bulk_add(pending_detections)

        self._last_overlay = overlay
        return face_ms_total

    def _persist_new_track(self, state: TrackState, frame: Frame) -> None:
        with self._session() as session:
            if session is None:
                return
            repo = TrackRepository(session)
            record = repo.create(
                source_id=self.config.source_id,
                track_key=state.track_key,
                object_class=state.object_class,
                first_seen_at=state.first_seen_at,
                first_frame=state.first_frame,
                recording_id=self._recording_id,
            )
            state.db_track_id = record.id
            state.persisted = True

            emitter = EventEmitter(session, self._event_context())
            # An object the detector could not confidently classify is still
            # activity worth recording.
            event_type = (
                EventType.UNIDENTIFIED_MOVEMENT
                if state.object_class in ("", UNIDENTIFIED_MOVEMENT)
                else EventType.OBJECT_TRACK_STARTED
            )
            event = emitter.open(
                event_type,
                dedup_key=f"track:{state.track_key}",
                when=state.first_seen_at,
                track_id=record.id,
                label=state.object_class,
                confidence=state.confidence,
                message=f"{state.object_class} detected (track #{state.track_key})",
                metadata={"track_key": state.track_key},
            )
            state.track_event_id = event.id

        self.bus.emit(
            "track_started",
            source_uid=self.config.source_uid,
            track_id=state.track_key,
            db_track_id=state.db_track_id,
            object_class=state.object_class,
        )

    def _retire_tracks(self, frame: Frame) -> None:
        """Close tracks the tracker has dropped.

        Draining (rather than peeking) guarantees each ended track is closed
        exactly once, even if a frame is skipped.
        """
        drain = getattr(self.tracker, "drain_removed", None)
        removed = list(drain()) if callable(drain) else []
        if not removed:
            return
        for key in removed:
            state = self.tracks.pop(key, None)
            if state is None:
                continue
            self._close_track(state, frame)

    def _close_track(self, state: TrackState, frame: Frame) -> None:
        with self._session() as session:
            if session is None:
                return
            emitter = EventEmitter(session, self._event_context())
            # Close any zone/unknown events this track still holds open.
            for event_id, closing in (
                (state.zone_b_open_event_id, EventType.PROXIMITY_B_EXITED),
                (state.zone_a_open_event_id, EventType.PROXIMITY_A_EXITED),
                (state.unknown_event_id, None),
            ):
                if event_id is not None:
                    emitter.close(event_id, when=frame.timestamp, closing_type=closing)
            emitter.close(
                state.track_event_id,
                when=frame.timestamp,
                closing_type=EventType.OBJECT_TRACK_ENDED,
                message=f"{state.object_class} left (track #{state.track_key})",
            )

            if state.db_track_id is not None:
                TrackRepository(session).close(
                    state.db_track_id,
                    last_seen_at=state.last_seen_at,
                    duration_seconds=state.duration_seconds,
                    trajectory=state.trajectory,
                )
                TrackRepository(session).update_state(
                    state.db_track_id,
                    status=TrackStatus.ENDED,
                    detection_count=state.detection_count,
                    max_confidence=state.max_confidence,
                    recognition_state=state.recognition_state,
                    identity_id=state.identity_id,
                    recognition_confidence=state.recognition_confidence,
                    proximity_zone=state.proximity_zone,
                    min_distance_m=state.min_distance_m,
                    last_distance_m=state.distance_m,
                    entered_zone_a=state.ever_entered_a,
                    entered_zone_b=state.ever_entered_b,
                    last_frame=state.last_frame,
                )

        # The alarm itself keeps ringing until an operator stops it; the track
        # simply stops holding the latch.
        self.alerts.release_track(self.config.source_id, state.track_key)
        self.dedup.forget_prefix(f"track:{state.track_key}")
        self.bus.emit(
            "track_ended",
            source_uid=self.config.source_uid,
            track_id=state.track_key,
            duration_seconds=round(state.duration_seconds, 2),
        )

    # ---------------------------------------------------------- proximity
    def _handle_zone_transition(
        self, state: TrackState, previous: ProximityZone, current: ProximityZone, frame: Frame
    ) -> None:
        """Emit A/B enter and exit events exactly once per crossing."""
        was_in_a = previous in (ProximityZone.ZONE_A, ProximityZone.ZONE_B)
        now_in_a = current in (ProximityZone.ZONE_A, ProximityZone.ZONE_B)
        was_in_b = previous is ProximityZone.ZONE_B
        now_in_b = current is ProximityZone.ZONE_B

        if was_in_a == now_in_a and was_in_b == now_in_b:
            return

        with self._session() as session:
            if session is None:
                return
            emitter = EventEmitter(session, self._event_context())
            common = {
                "track_id": state.db_track_id,
                "identity_id": state.identity_id,
                "label": state.display_label,
                "metadata": {
                    "track_key": state.track_key,
                    "distance_m": state.distance_m,
                    "zone": current.value,
                },
            }

            if now_in_a and not was_in_a:
                event = emitter.open(
                    EventType.PROXIMITY_A_ENTERED,
                    dedup_key=f"track:{state.track_key}:zone_a",
                    when=frame.timestamp,
                    message=f"{state.display_label} entered recognition zone",
                    **common,
                )
                state.zone_a_open_event_id = event.id
            elif was_in_a and not now_in_a:
                emitter.close(
                    state.zone_a_open_event_id,
                    when=frame.timestamp,
                    closing_type=EventType.PROXIMITY_A_EXITED,
                )
                state.zone_a_open_event_id = None

            if now_in_b and not was_in_b:
                event = emitter.open(
                    EventType.PROXIMITY_B_ENTERED,
                    dedup_key=f"track:{state.track_key}:zone_b",
                    when=frame.timestamp,
                    severity=EventSeverity.NOTICE,
                    message=f"{state.display_label} entered alarm zone",
                    **common,
                )
                state.zone_b_open_event_id = event.id
            elif was_in_b and not now_in_b:
                emitter.close(
                    state.zone_b_open_event_id,
                    when=frame.timestamp,
                    closing_type=EventType.PROXIMITY_B_EXITED,
                )
                state.zone_b_open_event_id = None

        self.bus.emit(
            "proximity",
            source_uid=self.config.source_uid,
            track_id=state.track_key,
            zone=current.value,
            previous_zone=previous.value,
            distance_m=state.distance_m,
            identity=state.identity_label,
            recognition_state=state.recognition_state.value,
        )

    # --------------------------------------------------------------- face
    def _person_search_region(self, bbox: BBox, width: int, height: int) -> BBox:
        pad_x = bbox.width * FACE_CROP_PADDING
        pad_y = bbox.height * FACE_CROP_PADDING
        return BBox(
            bbox.x1 - pad_x,
            bbox.y1 - pad_y,
            bbox.x2 + pad_x,
            bbox.y1 + bbox.height * FACE_SEARCH_TOP_FRACTION + pad_y,
        ).clipped(width, height)

    def _process_face(
        self, state: TrackState, image: np.ndarray, frame: Frame, scale: float
    ) -> None:
        if self.face_detector is None or self.face_quality is None or state.bbox is None:
            return
        state.mark_recognition_attempted()

        # Search the person's upper body in the *inference-resolution* image.
        inference_bbox = state.bbox.scaled(1.0 / scale)
        region = self._person_search_region(
            inference_bbox, image.shape[1], image.shape[0]
        )
        x1, y1, x2, y2 = region.as_int_tuple()
        if x2 - x1 < 8 or y2 - y1 < 8:
            state.note_no_face()
            return
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            state.note_no_face()
            return

        try:
            faces = self.face_detector.detect(crop)
        except Exception as exc:
            self.stats.errors += 1
            log.warning(FACE_RECOGNITION_ERROR, source=self.config.source_uid, error=str(exc))
            state.note_no_face()
            return

        if not faces:
            state.note_no_face()
            return

        # The biggest face in the person's box is the one that belongs to them.
        face = max(faces, key=lambda f: f.bbox.area)
        self.stats.face_detections += 1

        quality = self.face_quality.assess(face, crop)
        face.quality = quality

        # Face geometry back into original-frame coordinates.
        full_face_bbox = BBox(
            (x1 + face.bbox.x1) * scale,
            (y1 + face.bbox.y1) * scale,
            (x1 + face.bbox.x2) * scale,
            (y1 + face.bbox.y2) * scale,
        ).clipped(frame.width, frame.height)

        if not quality.ok:
            state.note_unusable_face()
            self._persist_face(
                state, face, full_face_bbox, frame,
                recognition_state=RecognitionState.FACE_UNRECOGNIZABLE,
                match_score=None, store_crop=False,
            )
            self.bus.emit(
                "face",
                source_uid=self.config.source_uid,
                track_id=state.track_key,
                state=RecognitionState.FACE_UNRECOGNIZABLE.value,
                reason=quality.reason,
                quality=quality.to_dict(),
            )
            return

        if self.face_embedder is None:
            state.note_unusable_face()
            return

        embedding = self.face_embedder.embed(crop, face)
        if embedding is None:
            state.note_unusable_face()
            return

        match = self.index.recognize(
            embedding, threshold=self.config.recognition_threshold
        )
        self.stats.recognitions += 1
        previous_state = state.recognition_state
        previous_identity = state.identity_id
        changed = state.apply_match(match)

        self._persist_face(
            state, face, full_face_bbox, frame,
            recognition_state=state.recognition_state,
            match_score=match.score,
            store_crop=state.recognition_state is RecognitionState.UNFAMILIAR,
            embedding=embedding,
        )

        if changed or previous_identity != state.identity_id:
            self._on_identity_settled(state, previous_state, frame, match_score=match.score)

    def _persist_face(
        self,
        state: TrackState,
        face: DetectedFace,
        full_bbox: BBox,
        frame: Frame,
        *,
        recognition_state: RecognitionState,
        match_score: float | None,
        store_crop: bool,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Persist a face observation.

        Crops are written to disk only when they are actionable (an unfamiliar
        face awaiting review) or belong to a new identity - we do not
        accumulate face imagery unnecessarily.
        """
        quality = face.quality
        # One stored face per track per condition: reviewing 400 crops of the
        # same person helps nobody.
        dedup_key = f"track:{state.track_key}:face:{recognition_state.value}"
        if not self.dedup.should_emit(dedup_key, when=frame.timestamp, window_seconds=60.0):
            return

        image_path: str | None = None
        if store_crop and self.config.store_face_crops and face.crop is not None:
            image_path = self._write_face_crop(state, face.crop, frame)

        with self._session() as session:
            if session is None:
                return
            record = Face(
                source_id=self.config.source_id,
                track_id=state.db_track_id,
                identity_id=state.identity_id,
                recording_id=self._recording_id,
                detected_at=frame.timestamp,
                frame_number=frame.frame_number,
                image_path=image_path,
                bbox_x1=full_bbox.x1,
                bbox_y1=full_bbox.y1,
                bbox_x2=full_bbox.x2,
                bbox_y2=full_bbox.y2,
                detection_confidence=face.confidence,
                quality_score=quality.score if quality else 0.0,
                blur_score=quality.blur if quality else None,
                brightness=quality.brightness if quality else None,
                face_pixels=quality.face_pixels if quality else None,
                quality_ok=bool(quality and quality.ok),
                quality_reason=quality.reason if quality else None,
                recognition_state=recognition_state,
                match_score=match_score,
            )
            FaceRepository(session).add(record)

            emitter = EventEmitter(session, self._event_context())
            if recognition_state is RecognitionState.UNFAMILIAR:
                # One open UNKNOWN_FACE event per track, extended over time.
                event = emitter.open(
                    EventType.UNKNOWN_FACE,
                    dedup_key=f"track:{state.track_key}:unknown",
                    when=frame.timestamp,
                    track_id=state.db_track_id,
                    face_id=record.id,
                    severity=EventSeverity.WARNING,
                    label="UNKNOWN",
                    message="Unfamiliar person detected",
                    confidence=match_score,
                    metadata={"track_key": state.track_key},
                )
                state.unknown_event_id = event.id
            elif recognition_state is RecognitionState.FACE_UNRECOGNIZABLE:
                emitter.emit(
                    EventType.FACE_UNRECOGNIZABLE,
                    when=frame.timestamp,
                    track_id=state.db_track_id,
                    face_id=record.id,
                    label="FACE DETECTED",
                    message=f"Face detected but unrecognizable ({quality.reason if quality else 'unknown'})",
                    metadata=quality.to_dict() if quality else {},
                )
            else:
                emitter.emit(
                    EventType.FACE_DETECTED,
                    when=frame.timestamp,
                    track_id=state.db_track_id,
                    face_id=record.id,
                    identity_id=state.identity_id,
                    label=state.display_label,
                    confidence=match_score,
                )

    def _write_face_crop(self, state: TrackState, crop: np.ndarray, frame: Frame) -> str | None:
        try:
            stamp = frame.timestamp.astimezone().strftime("%Y%m%d_%H%M%S")
            name = (
                f"{slugify(self.config.source_uid)}_track{state.track_key}_{stamp}_"
                f"{frame.frame_number}.jpg"
            )
            path = face_root() / "unfamiliar" / name
            ensure_parent(path)
            if not cv2.imwrite(str(path), crop):
                return None
            return relative_to_root(path, face_root())
        except Exception as exc:
            log.warning("face_crop_write_failed", source=self.config.source_uid, error=str(exc))
            return None

    def _on_identity_settled(
        self,
        state: TrackState,
        previous_state: RecognitionState,
        frame: Frame,
        *,
        match_score: float,
    ) -> None:
        """React to a track reaching a new recognition conclusion."""
        current = state.recognition_state
        self.bus.emit(
            "recognition",
            source_uid=self.config.source_uid,
            track_id=state.track_key,
            previous_state=previous_state.value,
            state=current.value,
            identity_id=state.identity_id,
            identity=state.identity_label,
            score=round(match_score, 3),
        )

        if current not in (
            RecognitionState.PERMANENT_FAMILIAR,
            RecognitionState.TEMPORARY_FAMILIAR,
        ):
            return

        with self._session() as session:
            if session is None:
                return
            emitter = EventEmitter(session, self._event_context())
            event_type = (
                EventType.PERMANENT_FAMILIAR_DETECTED
                if current is RecognitionState.PERMANENT_FAMILIAR
                else EventType.TEMPORARY_FAMILIAR_DETECTED
            )
            emitter.emit(
                event_type,
                when=frame.timestamp,
                track_id=state.db_track_id,
                identity_id=state.identity_id,
                label=state.identity_label,
                confidence=match_score,
                message=f"Recognised {state.identity_label}",
            )
            emitter.emit(
                EventType.FACE_RECOGNIZED,
                when=frame.timestamp,
                track_id=state.db_track_id,
                identity_id=state.identity_id,
                label=state.identity_label,
                confidence=match_score,
            )
            if state.db_track_id is not None:
                TrackRepository(session).update_state(
                    state.db_track_id,
                    identity_id=state.identity_id,
                    recognition_state=current,
                    recognition_confidence=match_score,
                )
            if state.identity_id is not None:
                IdentityRepository(session).touch_seen(state.identity_id, frame.timestamp)

            # A temporary familiar beeps once, on recognition.
            decision = evaluate_recognition(
                recognition_state=current, policy=self.config.policy
            )
            if decision.action is SecurityAction.BEEP and not state.beeped:
                state.beeped = True
                self.alerts.beep(
                    session,
                    source_id=self.config.source_id,
                    source_uid=self.config.source_uid,
                    reason=decision.reason,
                    track_key=state.track_key,
                    track_id=state.db_track_id,
                    identity_id=state.identity_id,
                    identity_label=state.identity_label,
                )

    # ----------------------------------------------------------- security
    def _apply_security(self, state: TrackState, frame: Frame) -> None:
        """Run the rule engine for a track inside the alarm zone."""
        if not state.is_person or not state.in_alarm_zone:
            return

        already = self.alerts.is_alarming(self.config.source_id, state.track_key)
        decision = evaluate_proximity_b(
            recognition_state=state.recognition_state,
            policy=self.config.policy,
            already_alarming=already,
        )
        if decision.action is SecurityAction.NONE:
            return

        with self._session() as session:
            if session is None:
                return
            emitter = EventEmitter(session, self._event_context())
            if decision.action is SecurityAction.START_CONTINUOUS_ALARM:
                event = emitter.emit(
                    EventType.ALARM_STARTED,
                    when=frame.timestamp,
                    track_id=state.db_track_id,
                    identity_id=state.identity_id,
                    severity=EventSeverity.CRITICAL,
                    label=state.display_label,
                    message=f"Alarm: {decision.reason}",
                    metadata={
                        "track_key": state.track_key,
                        "distance_m": state.distance_m,
                        "recognition_state": state.recognition_state.value,
                    },
                )
                alert = self.alerts.start_continuous(
                    session,
                    source_id=self.config.source_id,
                    source_uid=self.config.source_uid,
                    reason=decision.reason,
                    track_key=state.track_key,
                    track_id=state.db_track_id,
                    identity_id=state.identity_id,
                    identity_label=state.identity_label,
                    security_event_id=event.id,
                )
                if alert is not None:
                    state.alarm_alert_id = alert.id
            elif decision.action is SecurityAction.BEEP:
                key = f"track:{state.track_key}:beep_b"
                if self.dedup.should_emit(key, when=frame.timestamp, window_seconds=30.0):
                    self.alerts.beep(
                        session,
                        source_id=self.config.source_id,
                        source_uid=self.config.source_uid,
                        reason=decision.reason,
                        track_key=state.track_key,
                        track_id=state.db_track_id,
                        identity_id=state.identity_id,
                        identity_label=state.identity_label,
                    )

    # ----------------------------------------------------------- identity
    def invalidate_identity(self, identity_id: int) -> None:
        """Called when an identity is deleted or expires while tracks are live."""
        for state in self.tracks.values():
            state.invalidate_identity(identity_id)

    def close_all(self, frame: Frame | None = None) -> None:
        """Close every open track/motion record (source or app shutting down)."""
        now = datetime.now(timezone.utc)
        pseudo = frame or Frame(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            frame_number=0,
            timestamp=now,
            source_uid=self.config.source_uid,
        )
        for state in list(self.tracks.values()):
            self._close_track(state, pseudo)
        self.tracks.clear()
        if self._motion_event_id is not None:
            self._handle_motion(False, 0.0, pseudo)

    def overlay(self) -> list[dict]:
        return self._last_overlay

    def active_track_count(self) -> int:
        return len(self.tracks)


def build_pipeline(
    config: PipelineConfig,
    *,
    session_factory: SessionFactory | None = None,
    detector: ObjectDetector | None = None,
) -> IntelligencePipeline:
    """Wire up the default (YOLO + YuNet + SFace) implementations."""
    from app.intelligence.detection.yolo import YOLOObjectDetector
    from app.intelligence.face.detector import YuNetFaceDetector
    from app.intelligence.face.embedder import SFaceEmbedder
    from app.intelligence.face.quality import DefaultFaceQualityAssessor
    from app.intelligence.motion.detector import MOG2MotionDetector
    from app.intelligence.proximity.estimator import PinholeProximityEstimator
    from app.intelligence.tracking.iou_tracker import IoUTracker, PassthroughTracker

    settings = get_settings()
    detector = detector or YOLOObjectDetector()
    tracker = PassthroughTracker() if detector.supports_native_tracking else IoUTracker()

    return IntelligencePipeline(
        config,
        detector=detector,
        tracker=tracker,
        motion=MOG2MotionDetector(),
        proximity=PinholeProximityEstimator(),
        face_detector=YuNetFaceDetector(),
        face_quality=DefaultFaceQualityAssessor(),
        face_embedder=SFaceEmbedder(),
        session_factory=session_factory,
        alert_engine=get_alert_engine(),
    )
