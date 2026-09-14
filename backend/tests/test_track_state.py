"""Track-level identity memory and temporal consistency.

Covers the distinction the brief calls the most important one:

    TRACK ID  !=  IDENTITY ID
    NO_FACE != FACE_UNRECOGNIZABLE != UNFAMILIAR != FAMILIAR
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.intelligence.track_state import new_track_state
from app.intelligence.types import BBox, ProximityResult, RecognitionMatch
from app.models.enums import ProximityZone, RecognitionState

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def track():
    return new_track_state(
        track_key=17, source_id=1, object_class="person", frame_number=100, timestamp=NOW
    )


def observe(track, frame=101, seconds=1, height=200):
    track.observe(
        bbox=BBox(100, 100, 160, 100 + height),
        confidence=0.9,
        object_class=track.object_class,
        frame_number=frame,
        timestamp=NOW + timedelta(seconds=seconds),
    )


def match(identity_id, state, score=0.8, label="Azeem"):
    return RecognitionMatch(state=state, identity_id=identity_id, label=label, score=score)


def proximity(distance, zone):
    return ProximityResult(distance_m=distance, zone=zone, confidence=0.8, method="test")


class TestTrackIdentityDistinction:
    def test_a_new_track_has_no_identity(self, track):
        assert track.track_key == 17
        assert track.identity_id is None
        assert track.recognition_state is RecognitionState.NO_FACE

    def test_track_key_is_not_the_identity_id(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        assert track.track_key == 17
        assert track.identity_id == 42

    def test_observations_accumulate_on_one_track(self, track):
        for i in range(5):
            observe(track, frame=100 + i, seconds=i)
        assert track.detection_count == 5
        assert track.last_frame == 104
        assert len(track.trajectory) == 5

    def test_trajectory_is_bounded(self, track):
        for i in range(900):
            observe(track, frame=i, seconds=i)
        assert len(track.trajectory) <= 600


class TestProximityMemory:
    def test_zone_transitions_are_reported(self, track):
        previous, current = track.update_proximity(proximity(20.0, ProximityZone.FAR))
        assert current is ProximityZone.FAR
        previous, current = track.update_proximity(proximity(8.0, ProximityZone.ZONE_A))
        assert previous is ProximityZone.FAR
        assert current is ProximityZone.ZONE_A

    def test_entering_zones_is_latched(self, track):
        track.update_proximity(proximity(8.0, ProximityZone.ZONE_A))
        assert track.ever_entered_a and not track.ever_entered_b
        track.update_proximity(proximity(2.0, ProximityZone.ZONE_B))
        assert track.ever_entered_b
        # Walking away does not un-happen the breach.
        track.update_proximity(proximity(30.0, ProximityZone.FAR))
        assert track.ever_entered_a and track.ever_entered_b

    def test_minimum_distance_is_retained(self, track):
        for d in (12.0, 6.0, 2.5, 9.0):
            track.update_proximity(
                proximity(d, ProximityZone.ZONE_B if d < 3 else ProximityZone.ZONE_A)
            )
        assert track.min_distance_m == 2.5
        assert track.distance_m == 9.0

    def test_zone_b_counts_as_inside_the_recognition_zone(self, track):
        track.update_proximity(proximity(2.0, ProximityZone.ZONE_B))
        assert track.in_recognition_zone
        assert track.in_alarm_zone


class TestRecognitionLifecycle:
    def test_far_person_is_pending_not_unfamiliar(self, track):
        """The critical false-alarm guard."""
        track.update_proximity(proximity(30.0, ProximityZone.FAR))
        track.note_no_face()
        assert track.recognition_state is RecognitionState.UNKNOWN_PENDING_RECOGNITION
        assert track.recognition_state is not RecognitionState.UNFAMILIAR

    def test_poor_quality_face_is_unrecognizable_not_unfamiliar(self, track):
        track.update_proximity(proximity(5.0, ProximityZone.ZONE_A))
        track.note_unusable_face()
        assert track.recognition_state is RecognitionState.FACE_UNRECOGNIZABLE
        assert track.identity_id is None

    def test_first_conclusive_match_is_applied_immediately(self, track):
        changed = track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        assert changed
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR
        assert track.identity_label == "Azeem"

    def test_unfamiliar_is_a_conclusive_state(self, track):
        track.apply_match(
            RecognitionMatch(state=RecognitionState.UNFAMILIAR, score=0.2)
        )
        assert track.recognition_state is RecognitionState.UNFAMILIAR
        assert track.identity_id is None


class TestTemporalConsistency:
    def test_one_bad_frame_does_not_demote_a_known_identity(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.note_unusable_face()
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR
        assert track.identity_id == 42

    def test_a_missing_face_does_not_demote_a_known_identity(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.note_no_face()
        assert track.recognition_state is RecognitionState.PERMANENT_FAMILIAR

    def test_switching_identity_needs_repeated_evidence(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR, label="Azeem"))

        # Two disagreeing observations are not enough.
        track.apply_match(match(99, RecognitionState.PERMANENT_FAMILIAR, label="John"),
                          votes_to_switch=3)
        assert track.identity_id == 42
        track.apply_match(match(99, RecognitionState.PERMANENT_FAMILIAR, label="John"),
                          votes_to_switch=3)
        assert track.identity_id == 42

        # The third consecutive one flips it.
        track.apply_match(match(99, RecognitionState.PERMANENT_FAMILIAR, label="John"),
                          votes_to_switch=3)
        assert track.identity_id == 99
        assert track.identity_label == "John"

    def test_agreement_resets_a_pending_switch(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.apply_match(match(99, RecognitionState.PERMANENT_FAMILIAR), votes_to_switch=3)
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR), votes_to_switch=3)
        track.apply_match(match(99, RecognitionState.PERMANENT_FAMILIAR), votes_to_switch=3)
        assert track.identity_id == 42

    def test_invalidating_an_identity_resets_the_track(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.invalidate_identity(42)
        assert track.identity_id is None
        assert track.recognition_state is RecognitionState.NO_FACE

    def test_invalidating_a_different_identity_is_a_no_op(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.invalidate_identity(7)
        assert track.identity_id == 42


class TestRecognitionScheduling:
    def test_non_people_are_never_scheduled(self):
        dog = new_track_state(track_key=1, source_id=1, object_class="dog",
                              frame_number=0, timestamp=NOW)
        dog.update_proximity(proximity(2.0, ProximityZone.ZONE_B))
        assert not dog.should_attempt_recognition()

    def test_people_outside_zone_a_are_not_scheduled(self, track):
        track.update_proximity(proximity(30.0, ProximityZone.FAR))
        assert not track.should_attempt_recognition()

    def test_people_inside_zone_a_are_scheduled(self, track):
        track.update_proximity(proximity(6.0, ProximityZone.ZONE_A))
        assert track.should_attempt_recognition()

    def test_an_identified_track_waits_for_its_cooldown(self, track):
        track.update_proximity(proximity(6.0, ProximityZone.ZONE_A))
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        track.mark_recognition_attempted()
        assert not track.should_attempt_recognition(cooldown=45)
        for _ in range(46):
            observe(track)
        assert track.should_attempt_recognition(cooldown=45)


class TestDisplay:
    def test_named_identity_is_displayed(self, track):
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR, label="Azeem"))
        assert track.display_label == "Azeem"

    def test_unfamiliar_displays_as_unknown(self, track):
        track.apply_match(RecognitionMatch(state=RecognitionState.UNFAMILIAR))
        assert track.display_label == "UNKNOWN"

    def test_unrecognizable_does_not_claim_an_identity(self, track):
        track.note_unusable_face()
        assert track.display_label == "FACE DETECTED"

    def test_overlay_payload_is_complete(self, track):
        observe(track)
        track.update_proximity(proximity(2.0, ProximityZone.ZONE_B))
        track.apply_match(match(42, RecognitionState.PERMANENT_FAMILIAR))
        payload = track.to_overlay()
        assert payload["track_id"] == 17
        assert payload["identity_id"] == 42
        assert payload["recognition_state"] == "PERMANENT_FAMILIAR"
        assert payload["proximity_zone"] == "ZONE_B"
        assert payload["bbox"] is not None
