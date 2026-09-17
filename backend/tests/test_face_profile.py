"""Multi-view face profiling, crowd-safe face assignment, unknown confirmation.

These cover the three mechanisms that decide whether the system gets a person
right, and they are written around the failure modes that motivated them:

  * one frame is a bad witness       -> FaceProfile / ProfileVerdict
  * a neighbour's face is not yours  -> assign_faces
  * a stranger accusation is serious -> TrackState unknown confirmation
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.intelligence.face.assign import FaceCandidate, assign_faces, containment
from app.intelligence.face.profile import (
    FaceProfile, ProfileVerdict, light_bucket, new_sample, yaw_bucket,
)
from app.intelligence.track_state import new_track_state
from app.intelligence.types import (
    BBox, DetectedFace, FaceQuality, ProximityResult, RecognitionMatch,
)
from app.models.enums import ProximityZone, RecognitionState

NOW = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- helpers
def quality(*, score=0.8, yaw=0.0, brightness=128.0, pixels=80, penalty=0.0):
    return FaceQuality(
        ok=True, score=score, reason="ok", face_pixels=pixels,
        blur=100.0, brightness=brightness, detection_confidence=0.95,
        yaw=yaw, threshold_penalty=penalty,
    )


def sample(*, score=0.8, yaw=0.0, brightness=128.0, frame=1, penalty=0.0):
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = 1.0
    return new_sample(
        embedding=vector,
        quality=quality(score=score, yaw=yaw, brightness=brightness, penalty=penalty),
        frame_number=frame,
        timestamp=NOW,
    )


class FakeIndex:
    """Returns a scripted match per sample, in order."""

    def __init__(self, matches: list[RecognitionMatch]) -> None:
        self.matches = matches
        self.calls = 0

    def recognize_batch(self, embeddings, *, threshold=None):
        self.calls += 1
        return [self.matches[i % len(self.matches)] for i in range(len(embeddings))]


def familiar(identity_id=7, score=0.62, label="Azeem"):
    return RecognitionMatch(
        state=RecognitionState.PERMANENT_FAMILIAR,
        identity_id=identity_id, label=label, score=score,
    )


def stranger(score=0.12):
    return RecognitionMatch(state=RecognitionState.UNFAMILIAR, score=score)


def face(x1, y1, x2, y2, confidence=0.95):
    return DetectedFace(bbox=BBox(x1, y1, x2, y2), confidence=confidence)


def proximity(distance, zone):
    return ProximityResult(distance_m=distance, zone=zone, confidence=0.8, method="test")


# ------------------------------------------------------------- collecting
class TestProfileCollection:
    def test_samples_are_bucketed_by_pose_and_lighting(self):
        assert yaw_bucket(-0.9) != yaw_bucket(0.0) != yaw_bucket(0.9)
        assert light_bucket(30.0) != light_bucket(128.0) != light_bucket(220.0)

    def test_a_profile_accumulates_observations(self):
        profile = FaceProfile(max_samples=20, per_bucket=5)
        for i in range(5):
            assert profile.add(sample(frame=i))
        assert profile.count == 5

    def test_one_pose_cannot_monopolise_the_profile(self):
        """A hundred frontal frames must not crowd out the profile views."""
        profile = FaceProfile(max_samples=50, per_bucket=3)
        for i in range(30):
            profile.add(sample(yaw=0.0, score=0.5, frame=i))
        assert profile.count == 3

        # A genuinely new angle still gets in.
        assert profile.add(sample(yaw=0.9, score=0.5, frame=99))
        assert profile.bucket_coverage == 2

    def test_a_better_view_replaces_a_worse_one_in_its_bucket(self):
        profile = FaceProfile(max_samples=10, per_bucket=1)
        profile.add(sample(score=0.3))
        profile.add(sample(score=0.9))
        assert profile.count == 1
        assert profile.best_sample().quality == pytest.approx(0.9)

    def test_a_worse_view_is_rejected(self):
        profile = FaceProfile(max_samples=10, per_bucket=1)
        profile.add(sample(score=0.9))
        assert not profile.add(sample(score=0.2))

    def test_the_profile_is_bounded(self):
        profile = FaceProfile(max_samples=6, per_bucket=2)
        for i in range(200):
            profile.add(sample(yaw=(i % 5) * 0.4 - 0.8, score=0.5, frame=i))
        assert profile.count <= 6

    def test_the_gallery_spreads_across_buckets(self):
        """Enrolling N copies of one angle would defeat the whole point."""
        profile = FaceProfile(max_samples=40, per_bucket=5)
        for yaw in (-0.9, 0.0, 0.9):
            for i in range(5):
                profile.add(sample(yaw=yaw, score=0.5 + i / 100, frame=i))
        gallery = profile.gallery(limit=3)
        assert len({yaw_bucket(s.yaw) for s in gallery}) == 3


# --------------------------------------------------------------- deciding
class TestProfileVerdict:
    def test_too_few_samples_is_not_a_conclusion(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        profile.add(sample())
        verdict = profile.verdict(FakeIndex([familiar()]), threshold=0.4)
        assert not verdict.conclusive
        assert verdict.reason == "insufficient_samples"

    def test_an_agreeing_majority_identifies_the_track(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))
        verdict = profile.verdict(FakeIndex([familiar()]), threshold=0.4, min_samples=5)
        assert verdict.is_familiar
        assert verdict.identity_id == 7
        assert verdict.support == pytest.approx(1.0)

    def test_a_minority_of_matches_is_not_enough(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))
        # 1 match in every 4 samples.
        index = FakeIndex([familiar(), stranger(), stranger(), stranger()])
        verdict = profile.verdict(index, threshold=0.4, min_samples=5)
        assert not verdict.is_familiar

    def test_a_consistent_nobody_is_reported_unfamiliar(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))
        verdict = profile.verdict(
            FakeIndex([stranger()]), threshold=0.4, min_samples=5
        )
        assert verdict.is_unfamiliar
        assert verdict.reason == "profile_unfamiliar"

    def test_a_clear_majority_too_close_to_the_runner_up_is_ambiguous(self):
        """A look-alike must not silently inherit a trusted label."""
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))

        def person(identity_id, label, score):
            return RecognitionMatch(
                state=RecognitionState.PERMANENT_FAMILIAR,
                identity_id=identity_id, label=label, score=score,
            )

        # 7 votes for A against 3 for B: a clear majority, but the two mean
        # scores are a hundredth apart, which is not a distinction worth
        # putting somebody's name to.
        index = FakeIndex(
            [person(1, "A", 0.61)] * 7 + [person(2, "B", 0.60)] * 3
        )
        verdict = profile.verdict(index, threshold=0.4, min_samples=5, margin=0.2)
        assert not verdict.conclusive
        assert verdict.reason == "ambiguous_margin"
        assert verdict.identity_id == 1  # reported, but not acted on

    def test_an_evenly_split_profile_concludes_nothing(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))
        index = FakeIndex([familiar(identity_id=1), stranger()])
        verdict = profile.verdict(index, threshold=0.4, min_samples=5)
        assert not verdict.conclusive
        assert verdict.reason == "split_evidence"

    def test_an_angled_sample_must_clear_a_higher_bar(self):
        """The pose penalty is what makes the soft gate safe."""
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100, penalty=0.3))
        # Scores 0.45 clear the raw 0.4 threshold but not 0.4 + 0.3.
        verdict = profile.verdict(
            FakeIndex([familiar(score=0.45)]), threshold=0.4, min_samples=5
        )
        assert verdict.is_unfamiliar

    def test_reverdict_is_throttled(self):
        profile = FaceProfile(max_samples=50, per_bucket=10)
        for i in range(10):
            profile.add(sample(frame=i, score=0.5 + i / 100))
        assert profile.should_reverdict()
        profile.verdict(FakeIndex([familiar()]), threshold=0.4, min_samples=5)
        assert not profile.should_reverdict(every=3)


# --------------------------------------------------- crowd face ownership
class TestFaceAssignment:
    def test_containment_measures_overlap_of_the_face(self):
        assert containment(BBox(10, 10, 20, 20), BBox(0, 0, 100, 100)) == 1.0
        assert containment(BBox(-10, -10, 0, 0), BBox(0, 0, 100, 100)) == 0.0

    def test_a_face_is_given_to_the_person_it_sits_in(self):
        people = [
            FaceCandidate(track_key=1, bbox=BBox(0, 0, 100, 300)),
            FaceCandidate(track_key=2, bbox=BBox(200, 0, 300, 300)),
        ]
        faces = [face(30, 10, 70, 60), face(230, 10, 270, 60)]
        result = assign_faces(faces, people)
        assert result[1].face.bbox.x1 == 30
        assert result[2].face.bbox.x1 == 230

    def test_one_face_cannot_serve_two_people(self):
        """The crowd bug: overlapping boxes both claiming the same face."""
        people = [
            FaceCandidate(track_key=1, bbox=BBox(0, 0, 200, 300)),
            FaceCandidate(track_key=2, bbox=BBox(20, 0, 220, 300)),
        ]
        result = assign_faces([face(60, 10, 100, 60)], people)
        assert len(result) == 1

    def test_a_face_outside_the_person_box_is_refused(self):
        people = [FaceCandidate(track_key=1, bbox=BBox(0, 0, 100, 300))]
        assert assign_faces([face(500, 500, 540, 560)], people) == {}

    def test_a_face_at_the_feet_loses_to_a_face_at_the_head(self):
        people = [FaceCandidate(track_key=1, bbox=BBox(0, 0, 100, 300))]
        head = face(30, 10, 70, 60)
        knees = face(30, 240, 70, 290)
        result = assign_faces([knees, head], people)
        assert result[1].face is head

    def test_a_bystanders_larger_face_does_not_win(self):
        """Biggest-face-in-the-box is exactly the heuristic being replaced."""
        person = FaceCandidate(track_key=1, bbox=BBox(0, 0, 100, 300))
        own = face(30, 10, 70, 60)
        # A much bigger face from somebody closer, overlapping low in the box.
        bystander = face(10, 150, 110, 290)
        result = assign_faces([bystander, own], [person])
        assert result[1].face is own

    def test_no_people_or_no_faces_is_empty(self):
        assert assign_faces([], [FaceCandidate(1, BBox(0, 0, 10, 10))]) == {}
        assert assign_faces([face(0, 0, 5, 5)], []) == {}


# ------------------------------------------------- unknown confirmation
@pytest.fixture()
def track():
    state = new_track_state(
        track_key=3, source_id=1, object_class="person", frame_number=1, timestamp=NOW
    )
    state.update_proximity(proximity(2.0, ProximityZone.ZONE_B))
    return state


def unfamiliar_verdict(score=0.1):
    return ProfileVerdict(
        state=RecognitionState.UNFAMILIAR, score=score, support=0.9,
        samples=10, reason="profile_unfamiliar",
    )


def familiar_verdict(identity_id=7, label="Azeem", score=0.7):
    return ProfileVerdict(
        state=RecognitionState.PERMANENT_FAMILIAR, identity_id=identity_id,
        label=label, score=score, support=0.9, samples=10, reason="profile_match",
    )


class TestUnknownConfirmation:
    def test_one_unfamiliar_verdict_does_not_confirm(self):
        state = new_track_state(
            track_key=3, source_id=1, object_class="person", frame_number=1, timestamp=NOW
        )
        state.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        assert not state.unknown_confirmed
        assert not state.alarm_eligible
        assert state.recognition_state is RecognitionState.UNKNOWN_PENDING_RECOGNITION

    def test_confirmation_needs_both_time_and_repetition(self, track):
        track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        # Enough elapsed time, but only two verdicts.
        track.apply_profile_verdict(
            unfamiliar_verdict(), now=NOW + timedelta(seconds=30)
        )
        assert not track.unknown_confirmed

        track.apply_profile_verdict(
            unfamiliar_verdict(), now=NOW + timedelta(seconds=31)
        )
        assert track.unknown_confirmed
        assert track.recognition_state is RecognitionState.UNFAMILIAR
        assert track.alarm_eligible

    def test_repetition_without_elapsed_time_does_not_confirm(self, track):
        for _ in range(10):
            track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        assert not track.unknown_confirmed

    def test_being_recognised_clears_the_unknown_evidence(self, track):
        track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        track.apply_profile_verdict(
            familiar_verdict(), now=NOW + timedelta(seconds=5)
        )
        assert track.identity_id == 7
        assert not track.unknown_confirmed
        assert track.unknown_since is None
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR

    def test_an_inconclusive_verdict_changes_nothing(self, track):
        track.apply_profile_verdict(familiar_verdict(), now=NOW)
        track.apply_profile_verdict(
            ProfileVerdict(state=None, reason="split_evidence"), now=NOW
        )
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR
        assert track.identity_id == 7

    def test_a_known_person_is_not_demoted_by_one_unfamiliar_verdict(self, track):
        track.apply_profile_verdict(familiar_verdict(), now=NOW)
        track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        assert track.identity_id == 7
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR

    def test_a_demoted_person_still_cannot_alarm_immediately(self, track):
        """Even a demotion restarts the evidence clock."""
        track.apply_profile_verdict(familiar_verdict(), now=NOW)
        for _ in range(3):
            track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        assert track.recognition_state is RecognitionState.UNFAMILIAR
        assert not track.alarm_eligible

    def test_the_alarm_rearm_gap_is_enforced(self, track):
        assert track.alarm_rearmed(NOW)
        track.last_alarm_at = NOW
        assert not track.alarm_rearmed(NOW + timedelta(seconds=5), gap_seconds=20)
        assert track.alarm_rearmed(NOW + timedelta(seconds=25), gap_seconds=20)

    def test_invalidating_an_identity_also_clears_evidence(self, track):
        track.apply_profile_verdict(familiar_verdict(), now=NOW)
        track.invalidate_identity(7)
        assert track.unknown_since is None
        assert track.last_verdict is None

    def test_the_overlay_reports_evidence_gathering(self, track):
        track.profile.add(sample())
        track.apply_profile_verdict(unfamiliar_verdict(), now=NOW)
        payload = track.to_overlay()
        assert payload["face_samples"] == 1
        assert payload["unknown_confirmed"] is False
