"""The security decision engine.

These tests encode the exact alarm policy from the brief, including the two
edge cases that prevent false alarms:

  * a person who has never been inside Proximity A is PENDING, not UNFAMILIAR
  * a face that could not be evaluated is UNRECOGNIZABLE, and does not alarm
"""
from __future__ import annotations

import pytest

from app.models.enums import AlertType, RecognitionState
from app.security.rules import (
    AlertPolicy, SecurityAction, evaluate_proximity_b, evaluate_recognition,
    should_attempt_face_recognition,
)

DEFAULT = AlertPolicy()


class TestProximityBRules:
    def test_permanent_familiar_never_alarms(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.PERMANENT_FAMILIAR, policy=DEFAULT
        )
        assert decision.action is SecurityAction.NONE
        assert not decision.raises_alert

    def test_temporary_familiar_beeps_by_default(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR, policy=DEFAULT
        )
        assert decision.action is SecurityAction.BEEP
        assert decision.alert_type is AlertType.BEEP

    def test_temporary_familiar_policy_can_be_silent(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR,
            policy=AlertPolicy(temporary_familiar="none"),
        )
        assert decision.action is SecurityAction.NONE

    def test_temporary_familiar_policy_can_be_continuous(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR,
            policy=AlertPolicy(temporary_familiar="continuous"),
        )
        assert decision.action is SecurityAction.START_CONTINUOUS_ALARM

    def test_confirmed_unfamiliar_triggers_a_timed_alarm_first(self):
        """First offence is loud but self-clearing, not a permanent siren."""
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNFAMILIAR, policy=DEFAULT
        )
        assert decision.action is SecurityAction.START_TIMED_ALARM
        assert decision.alert_type is AlertType.CONTINUOUS_ALARM
        assert decision.severity == "CRITICAL"
        assert decision.duration_seconds == 30

    def test_unconfirmed_unfamiliar_does_not_alarm(self):
        """The guard against one bad frame or one bad angle raising a siren."""
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNFAMILIAR,
            policy=DEFAULT,
            unknown_confirmed=False,
        )
        assert decision.action is SecurityAction.NONE
        assert decision.reason == "unfamiliar_pending_confirmation"

    def test_repeat_offender_escalates_to_a_continuous_alarm(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNFAMILIAR,
            policy=DEFAULT,
            alarm_cycles=3,
        )
        assert decision.action is SecurityAction.START_CONTINUOUS_ALARM
        assert decision.duration_seconds is None
        assert decision.severity == "CRITICAL"

    def test_escalation_threshold_is_configurable(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNFAMILIAR,
            policy=DEFAULT,
            alarm_cycles=1,
            escalate_after_cycles=1,
        )
        assert decision.action is SecurityAction.START_CONTINUOUS_ALARM

    def test_already_alarming_track_does_not_raise_again(self):
        """Frame-rate idempotence: one track, one alarm."""
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNFAMILIAR,
            policy=DEFAULT,
            already_alarming=True,
        )
        assert decision.action is SecurityAction.NONE


class TestInconclusiveRecognitionDoesNotAlarm:
    """Recognition failing is not evidence of an intruder."""

    def test_unrecognizable_face_does_not_alarm_by_default(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.FACE_UNRECOGNIZABLE, policy=DEFAULT
        )
        assert decision.action is SecurityAction.NONE
        assert "no_alarm_by_policy" in decision.reason

    def test_unrecognizable_can_alarm_when_site_policy_demands_it(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.FACE_UNRECOGNIZABLE,
            policy=AlertPolicy(unrecognizable_policy="alarm"),
        )
        assert decision.action is SecurityAction.START_CONTINUOUS_ALARM

    def test_pending_recognition_never_alarms(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.UNKNOWN_PENDING_RECOGNITION,
            policy=DEFAULT,
        )
        assert decision.action is SecurityAction.NONE

    def test_no_face_never_alarms(self):
        decision = evaluate_proximity_b(
            recognition_state=RecognitionState.NO_FACE, policy=DEFAULT
        )
        assert decision.action is SecurityAction.NONE

    @pytest.mark.parametrize(
        "state",
        [
            RecognitionState.NO_FACE,
            RecognitionState.UNKNOWN_PENDING_RECOGNITION,
            RecognitionState.FACE_UNRECOGNIZABLE,
        ],
    )
    def test_inconclusive_states_are_all_silent(self, state):
        assert not evaluate_proximity_b(
            recognition_state=state, policy=DEFAULT
        ).raises_alert


class TestRecognitionFeedback:
    def test_temporary_familiar_beeps_once_on_recognition(self):
        decision = evaluate_recognition(
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR, policy=DEFAULT
        )
        assert decision.action is SecurityAction.BEEP

    def test_permanent_familiar_is_silent_on_recognition(self):
        decision = evaluate_recognition(
            recognition_state=RecognitionState.PERMANENT_FAMILIAR, policy=DEFAULT
        )
        assert decision.action is SecurityAction.NONE

    def test_permanent_familiar_can_be_configured_to_beep(self):
        decision = evaluate_recognition(
            recognition_state=RecognitionState.PERMANENT_FAMILIAR,
            policy=AlertPolicy(permanent_familiar="beep"),
        )
        assert decision.action is SecurityAction.BEEP

    def test_recognition_beep_can_be_disabled_entirely(self):
        decision = evaluate_recognition(
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR,
            policy=AlertPolicy(beep_on_recognition=False),
        )
        assert decision.action is SecurityAction.NONE


class TestFaceProcessingGate:
    def test_people_inside_the_recognition_zone_are_processed(self):
        assert should_attempt_face_recognition(
            object_class="person", in_recognition_zone=True, recognition_enabled=True
        )

    def test_people_outside_the_recognition_zone_are_not(self):
        assert not should_attempt_face_recognition(
            object_class="person", in_recognition_zone=False, recognition_enabled=True
        )

    @pytest.mark.parametrize("cls", ["dog", "cat", "car", "truck", "bicycle"])
    def test_non_people_never_get_face_processing(self, cls):
        assert not should_attempt_face_recognition(
            object_class=cls, in_recognition_zone=True, recognition_enabled=True
        )

    def test_recognition_can_be_disabled_per_source(self):
        assert not should_attempt_face_recognition(
            object_class="person", in_recognition_zone=True, recognition_enabled=False
        )


class TestAlertPolicyParsing:
    def test_defaults_are_the_safe_ones(self):
        policy = AlertPolicy.from_dict(None)
        assert policy.temporary_familiar == "beep"
        assert policy.unrecognizable_policy == "ignore"

    def test_partial_overrides_inherit_the_rest(self):
        policy = AlertPolicy.from_dict(
            {"temporary_familiar": "continuous"},
            defaults=AlertPolicy(unrecognizable_policy="alarm"),
        )
        assert policy.temporary_familiar == "continuous"
        assert policy.unrecognizable_policy == "alarm"

    def test_round_trips_through_a_dict(self):
        policy = AlertPolicy(temporary_familiar="none", unrecognizable_policy="alarm")
        assert AlertPolicy.from_dict(policy.to_dict()) == policy
