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
* A track owns a :class:`~app.intelligence.face.profile.FaceProfile` - many
  views of the same face - and is identified from the profile as a whole.
  Being declared UNFAMILIAR additionally requires the profile to say so
  repeatedly over ``UNKNOWN_CONFIRM_SECONDS``, so no single frame, angle or
  shadow can raise an alarm on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import get_settings
from app.intelligence.face.profile import FaceProfile, ProfileVerdict
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

    # --- face profile (many views of this person's face) ------------------
    profile: FaceProfile = field(default_factory=FaceProfile)
    last_verdict: ProfileVerdict | None = None
    # Sustained-unknown evidence. An alarm needs all three of these.
    unknown_since: datetime | None = None
    unknown_verdicts: int = 0
    unknown_confirmed: bool = False
    # How many times this track has been through the alarm cycle, which is
    # what escalates a repeat offender from a timed alarm to a continuous one.
    alarm_cycles: int = 0
    last_alarm_at: datetime | None = None
    profile_persisted: bool = False

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
        """Face work is expensive: only run it when it can change something.

        While the profile is still filling, every processed frame is useful -
        each one is a chance at an angle or a light level we do not have yet.
        Once it is full the track has all the evidence it is going to get, so
        it drops back to periodic re-verification.
        """
        if not self.is_person:
            return False
        if not self.in_recognition_zone:
            return False
        cooldown = (
            get_settings().recognition_cooldown_frames if cooldown is None else cooldown
        )
        if not self.profile.is_full:
            if self.recognition_state in FAMILIAR_STATES:
                # Known already, but still worth widening their gallery - just
                # not at full rate.
                return self.frames_since_recognition >= max(1, cooldown // 4)
            return True
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
        """A person is visible but no face could be attributed to them."""
        self.last_face_quality_ok = False
        if self.recognition_state in CONCLUSIVE_STATES:
            return  # keep what we already concluded
        if self.unknown_since is not None:
            # Mid-assessment. Losing sight of the face for a frame does not
            # undo the evidence gathered so far, and flipping the label back
            # and forth would make the overlay unreadable.
            self.recognition_state = RecognitionState.UNKNOWN_PENDING_RECOGNITION
            return
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

    # ------------------------------------------------------ profile verdict
    def apply_profile_verdict(
        self,
        verdict: ProfileVerdict,
        *,
        now: datetime | None = None,
        confirm_seconds: float | None = None,
        min_verdicts: int | None = None,
    ) -> bool:
        """Fold a whole-profile conclusion into the track.

        Returns True when the track's recognition state changed.

        The asymmetry here is deliberate and is the point of the whole
        mechanism. Recognising somebody is good news and is applied as soon as
        the profile agrees. Declaring somebody unknown is the step that can set
        off an alarm, so it additionally has to hold for ``confirm_seconds``
        across ``min_verdicts`` separate verdicts. A person who turns their
        head, walks through a shadow, or is briefly mistaken for nobody simply
        never reaches that bar.
        """
        settings = get_settings()
        now = now or datetime.now(timezone.utc)
        if confirm_seconds is None:
            confirm_seconds = settings.unknown_confirm_seconds
        if min_verdicts is None:
            min_verdicts = settings.unknown_confirm_min_verdicts

        self.last_verdict = verdict

        if not verdict.conclusive:
            # No conclusion: hold whatever we already believe. An inconclusive
            # profile is not evidence of anything and must not decay into one.
            return False

        if verdict.is_familiar:
            self._clear_unknown_evidence()
            self.last_face_quality_ok = True
            changed = (
                self.recognition_state is not verdict.state
                or self.identity_id != verdict.identity_id
            )
            self.recognition_state = verdict.state
            self.identity_id = verdict.identity_id
            self.identity_label = verdict.label
            self.recognition_confidence = verdict.score
            self._reset_pending()
            return changed

        # Unfamiliar: start (or continue) accumulating evidence.
        self.last_face_quality_ok = True
        if self.recognition_state in FAMILIAR_STATES:
            # A previously identified person is not demoted by the profile
            # drifting; that needs the same repeated evidence as a switch.
            return self._apply_unknown_against_known(verdict)

        if self.unknown_since is None:
            self.unknown_since = now
        self.unknown_verdicts += 1
        self.recognition_confidence = verdict.score

        elapsed = (now - self.unknown_since).total_seconds()
        if (
            not self.unknown_confirmed
            and elapsed >= confirm_seconds
            and self.unknown_verdicts >= max(1, min_verdicts)
        ):
            self.unknown_confirmed = True
            changed = self.recognition_state is not RecognitionState.UNFAMILIAR
            self.recognition_state = RecognitionState.UNFAMILIAR
            self.identity_id = None
            self.identity_label = None
            return changed

        if not self.unknown_confirmed:
            # Still gathering: an unconfirmed unknown is *pending*, which the
            # rule engine treats as "no conclusion" and never alarms on.
            if self.recognition_state not in CONCLUSIVE_STATES:
                self.recognition_state = RecognitionState.UNKNOWN_PENDING_RECOGNITION
        return False

    def _apply_unknown_against_known(self, verdict: ProfileVerdict) -> bool:
        """An unfamiliar verdict for a track that already has an identity."""
        votes = get_settings().face_votes_to_switch_identity
        if self._pending_state is RecognitionState.UNFAMILIAR:
            self._pending_votes += 1
        else:
            self._pending_state = RecognitionState.UNFAMILIAR
            self._pending_identity = None
            self._pending_label = None
            self._pending_votes = 1
        self._pending_score = verdict.score
        if self._pending_votes < votes:
            return False
        self.recognition_state = RecognitionState.UNFAMILIAR
        self.identity_id = None
        self.identity_label = None
        self.recognition_confidence = verdict.score
        self._reset_pending()
        # The evidence clock starts now, so even a demotion cannot alarm
        # instantly.
        self.unknown_since = None
        self.unknown_verdicts = 0
        self.unknown_confirmed = False
        return True

    def _clear_unknown_evidence(self) -> None:
        self.unknown_since = None
        self.unknown_verdicts = 0
        self.unknown_confirmed = False

    @property
    def unknown_evidence_seconds(self) -> float:
        if self.unknown_since is None:
            return 0.0
        return max(0.0, (self.last_seen_at - self.unknown_since).total_seconds())

    @property
    def alarm_eligible(self) -> bool:
        """Only a *confirmed* unknown may raise an alarm."""
        return (
            self.recognition_state is RecognitionState.UNFAMILIAR
            and self.unknown_confirmed
        )

    def alarm_rearmed(self, now: datetime, *, gap_seconds: float | None = None) -> bool:
        """Has the quiet gap after this track's last alarm elapsed?

        Without this a 30-second alarm would simply restart on the very next
        frame, which is a continuous alarm wearing a disguise.
        """
        if self.last_alarm_at is None:
            return True
        if gap_seconds is None:
            gap_seconds = get_settings().alarm_rearm_seconds
        return (now - self.last_alarm_at).total_seconds() >= gap_seconds

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
        self._clear_unknown_evidence()
        self.last_verdict = None

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
            # Evidence gathering, so the operator can see *why* a person is
            # still unlabelled rather than assuming the system missed them.
            "face_samples": self.profile.count,
            "pose_coverage": self.profile.bucket_coverage,
            "unknown_seconds": round(self.unknown_evidence_seconds, 1),
            "unknown_confirmed": self.unknown_confirmed,
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
