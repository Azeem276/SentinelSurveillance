"""Per-track runtime state: identity memory, proximity history, alarm latches.

This module owns the temporal-consistency rules that keep the system from
flip-flopping:

* A track remembers the identity it was recognised as. One bad crop does not
  turn "Azeem" into "Unknown"; switching identities requires
  ``votes_to_switch`` consecutive disagreeing observations.
* A person who has never been inside Proximity A is UNKNOWN_PENDING_RECOGNITION,
  never UNFAMILIAR. Recognition was not possible yet, so no conclusion is drawn.
* Recognition is not re-run every frame: once a track is identified, it is
  re-verified only after a cooldown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import get_settings
from app.intelligence.types import BBox, ProximityResult, RecognitionMatch
from app.models.enums import ProximityZone, RecognitionState

# States that represent a settled recognition conclusion.
CONCLUSIVE_STATES = frozenset(
    {
        RecognitionState.PERMANENT_FAMILIAR,
        RecognitionState.TEMPORARY_FAMILIAR,
        RecognitionState.UNFAMILIAR,
    }
)
FAMILIAR_STATES = frozenset(
    {RecognitionState.PERMANENT_FAMILIAR, RecognitionState.TEMPORARY_FAMILIAR}
)


@dataclass(slots=True)
class TrackState:
    """Mutable runtime state for one tracked object on one source."""

    track_key: int
    source_id: int
    object_class: str
    first_seen_at: datetime
    last_seen_at: datetime
    first_frame: int
    last_frame: int

    db_track_id: int | None = None
    bbox: BBox | None = None
    confidence: float = 0.0
    max_confidence: float = 0.0
    detection_count: int = 0
    trajectory: list[dict] = field(default_factory=list)

    # --- recognition ------------------------------------------------------
    recognition_state: RecognitionState = RecognitionState.NO_FACE
    identity_id: int | None = None
    identity_label: str | None = None
    recognition_confidence: float = 0.0
    frames_since_recognition: int = 1_000_000
    face_attempts: int = 0
    faces_detected: int = 0
    last_face_quality_ok: bool = False
    _pending_identity: int | None = None
    _pending_votes: int = 0
    _pending_state: RecognitionState | None = None
    _pending_label: str | None = None
    _pending_score: float = 0.0

    # --- proximity --------------------------------------------------------
    proximity_zone: ProximityZone = ProximityZone.UNKNOWN
    distance_m: float | None = None
    min_distance_m: float | None = None
    ever_entered_a: bool = False
    ever_entered_b: bool = False
    zone_a_open_event_id: int | None = None
    zone_b_open_event_id: int | None = None

    # --- alerts -----------------------------------------------------------
    alarm_alert_id: int | None = None
    beeped: bool = False
    unknown_event_id: int | None = None
    track_event_id: int | None = None
    persisted: bool = False

    # ------------------------------------------------------------ updating
    def observe(
        self,
        *,
        bbox: BBox,
        confidence: float,
        object_class: str,
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.object_class = object_class
        self.max_confidence = max(self.max_confidence, confidence)
        self.detection_count += 1
        self.last_frame = frame_number
        self.last_seen_at = timestamp
        self.frames_since_recognition += 1
        cx, cy = bbox.center
        # Cap the stored path so a long-lived track cannot grow without bound.
        if len(self.trajectory) < 600:
            self.trajectory.append(
                {"f": frame_number, "x": round(cx, 1), "y": round(cy, 1),
                 "t": timestamp.isoformat()}
            )

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.last_seen_at - self.first_seen_at).total_seconds())

    @property
    def is_person(self) -> bool:
        return self.object_class == "person"

    # ---------------------------------------------------------- proximity
    def update_proximity(self, result: ProximityResult) -> tuple[ProximityZone, ProximityZone]:
        """Apply a proximity reading. Returns (previous_zone, new_zone)."""
        previous = self.proximity_zone
        self.proximity_zone = result.zone
        self.distance_m = result.distance_m
        if result.distance_m is not None:
            self.min_distance_m = (
                result.distance_m
                if self.min_distance_m is None
                else min(self.min_distance_m, result.distance_m)
            )
        if result.zone in (ProximityZone.ZONE_A, ProximityZone.ZONE_B):
            self.ever_entered_a = True
        if result.zone is ProximityZone.ZONE_B:
            self.ever_entered_b = True
        return previous, result.zone

    @property
    def in_recognition_zone(self) -> bool:
        return self.proximity_zone in (ProximityZone.ZONE_A, ProximityZone.ZONE_B)

    @property
    def in_alarm_zone(self) -> bool:
        return self.proximity_zone is ProximityZone.ZONE_B

    # --------------------------------------------------------- recognition
    def should_attempt_recognition(self, *, cooldown: int | None = None) -> bool:
        """Face work is expensive: only run it when it can change something."""
        if not self.is_person:
            return False
        if not self.in_recognition_zone:
            return False
        cooldown = (
            get_settings().recognition_cooldown_frames if cooldown is None else cooldown
        )
        if self.recognition_state in FAMILIAR_STATES:
            # Confirmed identity: re-verify only occasionally.
            return self.frames_since_recognition >= cooldown
        if self.recognition_state is RecognitionState.UNFAMILIAR:
            # Keep trying at a slower cadence in case a better face appears.
            return self.frames_since_recognition >= max(1, cooldown // 2)
        # NO_FACE / PENDING / UNRECOGNIZABLE: try on every processed frame.
        return True

    def mark_recognition_attempted(self) -> None:
        self.face_attempts += 1
        self.frames_since_recognition = 0

    def note_no_face(self) -> None:
        """A person is visible but no face was found in the crop."""
        self.last_face_quality_ok = False
        if self.recognition_state in CONCLUSIVE_STATES:
            return  # keep what we already concluded
        self.recognition_state = (
            RecognitionState.UNKNOWN_PENDING_RECOGNITION
            if not self.ever_entered_a
            else RecognitionState.NO_FACE
        )

    def note_unusable_face(self) -> None:
        """A face was found but failed the quality gate: no conclusion drawn."""
        self.faces_detected += 1
        self.last_face_quality_ok = False
        if self.recognition_state in FAMILIAR_STATES:
            return  # a blurry frame never demotes a confirmed identity
        self.recognition_state = RecognitionState.FACE_UNRECOGNIZABLE

    def apply_match(self, match: RecognitionMatch, *, votes_to_switch: int | None = None) -> bool:
        """Fold a recognition result into the track. Returns True if it changed.

        Agreement with the current identity is applied immediately; a
        disagreement must be observed ``votes_to_switch`` times in a row.
        """
        self.faces_detected += 1
        self.last_face_quality_ok = True
        votes = (
            get_settings().face_votes_to_switch_identity
            if votes_to_switch is None
            else votes_to_switch
        )

        # First conclusive observation, or the same conclusion again.
        same_identity = match.identity_id == self.identity_id
        currently_unset = self.recognition_state not in CONCLUSIVE_STATES

        if currently_unset or same_identity:
            changed = (
                self.recognition_state is not match.state
                or self.identity_id != match.identity_id
            )
            self.recognition_state = match.state
            self.identity_id = match.identity_id
            self.identity_label = match.label
            self.recognition_confidence = match.score
            self._reset_pending()
            return changed

        # Disagreement: require repeated evidence before switching.
        if self._pending_identity == match.identity_id and self._pending_state is match.state:
            self._pending_votes += 1
        else:
            self._pending_identity = match.identity_id
            self._pending_state = match.state
            self._pending_label = match.label
            self._pending_votes = 1
        self._pending_score = match.score

        if self._pending_votes >= votes:
            self.recognition_state = match.state
            self.identity_id = match.identity_id
            self.identity_label = match.label
            self.recognition_confidence = match.score
            self._reset_pending()
            return True
        return False

    def _reset_pending(self) -> None:
        self._pending_identity = None
        self._pending_votes = 0
        self._pending_state = None
        self._pending_label = None
        self._pending_score = 0.0

    def invalidate_identity(self, identity_id: int) -> None:
        """Called when an identity is deleted or expires mid-track."""
        if self.identity_id != identity_id:
            return
        self.identity_id = None
        self.identity_label = None
        self.recognition_confidence = 0.0
        self.recognition_state = RecognitionState.NO_FACE
        self.frames_since_recognition = 1_000_000
        self._reset_pending()

    # ------------------------------------------------------------ display
    @property
    def display_label(self) -> str:
        if self.identity_label:
            return self.identity_label
        return {
            RecognitionState.UNFAMILIAR: "UNKNOWN",
            RecognitionState.FACE_UNRECOGNIZABLE: "FACE DETECTED",
            RecognitionState.UNKNOWN_PENDING_RECOGNITION: "PENDING",
            RecognitionState.NO_FACE: self.object_class.upper(),
        }.get(self.recognition_state, self.object_class.upper())

    def to_overlay(self) -> dict:
        """Compact payload for the live WebSocket overlay."""
        bbox = self.bbox
        return {
            "track_id": self.track_key,
            "db_track_id": self.db_track_id,
            "class": self.object_class,
            "confidence": round(self.confidence, 3),
            "bbox": bbox.to_dict() if bbox else None,
            "recognition_state": self.recognition_state.value,
            "identity_id": self.identity_id,
            "identity": self.identity_label,
            "label": self.display_label,
            "match_score": round(self.recognition_confidence, 3),
            "proximity_zone": self.proximity_zone.value,
            "distance_m": round(self.distance_m, 2) if self.distance_m is not None else None,
            "alarm": self.alarm_alert_id is not None,
            "duration_seconds": round(self.duration_seconds, 1),
        }


def new_track_state(
    *, track_key: int, source_id: int, object_class: str, frame_number: int,
    timestamp: datetime | None = None,
) -> TrackState:
    now = timestamp or datetime.now(timezone.utc)
    return TrackState(
        track_key=track_key,
        source_id=source_id,
        object_class=object_class,
        first_seen_at=now,
        last_seen_at=now,
        first_frame=frame_number,
        last_frame=frame_number,
    )
