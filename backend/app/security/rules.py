"""The security decision engine.

Deliberately a pure function of (recognition state, proximity zone, policy).
It knows nothing about models, cameras or databases, which is what makes the
rules testable and auditable.

    PERMANENT_FAMILIAR                      -> never alarms
    TEMPORARY_FAMILIAR  + crossed B         -> configured temporary policy
    UNFAMILIAR (confirmed) + crossed B      -> timed alarm, then escalation
    UNFAMILIAR (unconfirmed)                -> never alarms
    FACE_UNRECOGNIZABLE + crossed B         -> no alarm by default (policy)
    UNKNOWN_PENDING_RECOGNITION             -> never alarms

Two cases carry the weight here.

``FACE_UNRECOGNIZABLE`` means recognition *failed*, which is not evidence of an
intruder. Sites that want the stricter behaviour set
``unrecognizable_policy="alarm"``.

``UNFAMILIAR`` is now graded by how much evidence stands behind it. An unknown
person who has only just been judged unknown gets nothing; one whose face
profile has said "nobody" repeatedly over the confirmation window gets a timed
alarm that stops itself; one who keeps coming back is escalated to a
continuous alarm that an operator has to clear. That ladder exists because the
old behaviour - one bad frame, immediate never-ending siren - punished the
operator for the recogniser's worst moment rather than its considered view.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.core.config import get_settings
from app.models.enums import AlertType, RecognitionState


class SecurityAction(StrEnum):
    NONE = "NONE"
    BEEP = "BEEP"
    START_TIMED_ALARM = "START_TIMED_ALARM"
    START_CONTINUOUS_ALARM = "START_CONTINUOUS_ALARM"
    STOP_ALARM = "STOP_ALARM"


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    """Per-source alert configuration."""

    temporary_familiar: str = "beep"          # none | beep | continuous
    unrecognizable_policy: str = "ignore"     # ignore | alarm
    permanent_familiar: str = "none"          # none | beep
    beep_on_recognition: bool = True

    @classmethod
    def from_dict(cls, data: dict | None, *, defaults: "AlertPolicy | None" = None
                  ) -> "AlertPolicy":
        base = defaults or cls()
        data = data or {}
        return cls(
            temporary_familiar=str(data.get("temporary_familiar", base.temporary_familiar)),
            unrecognizable_policy=str(
                data.get("unrecognizable_policy", base.unrecognizable_policy)
            ),
            permanent_familiar=str(data.get("permanent_familiar", base.permanent_familiar)),
            beep_on_recognition=bool(
                data.get("beep_on_recognition", base.beep_on_recognition)
            ),
        )

    def to_dict(self) -> dict:
        return {
            "temporary_familiar": self.temporary_familiar,
            "unrecognizable_policy": self.unrecognizable_policy,
            "permanent_familiar": self.permanent_familiar,
            "beep_on_recognition": self.beep_on_recognition,
        }


@dataclass(frozen=True, slots=True)
class SecurityDecision:
    action: SecurityAction
    alert_type: AlertType | None
    reason: str
    severity: str = "INFO"
    # For START_TIMED_ALARM: how long the siren runs before clearing itself.
    duration_seconds: int | None = None

    @property
    def raises_alert(self) -> bool:
        return self.action in (
            SecurityAction.BEEP,
            SecurityAction.START_TIMED_ALARM,
            SecurityAction.START_CONTINUOUS_ALARM,
        )


NO_ACTION = SecurityDecision(SecurityAction.NONE, None, "no_action")


def evaluate_proximity_b(
    *,
    recognition_state: RecognitionState,
    policy: AlertPolicy,
    already_alarming: bool = False,
    unknown_confirmed: bool = True,
    alarm_cycles: int = 0,
    alarm_duration_seconds: int | None = None,
    escalate_after_cycles: int | None = None,
) -> SecurityDecision:
    """Decide what happens when a tracked person is inside the alarm zone.

    ``already_alarming`` makes the function idempotent: a track that is
    already alarming must not raise a second alarm on the next frame.

    ``unknown_confirmed`` says whether the UNFAMILIAR verdict is backed by the
    full evidence window. An unconfirmed unknown is treated as inconclusive -
    the person is being assessed, not accused.

    ``alarm_cycles`` counts how many timed alarms this track has already run
    through. Past ``escalate_after_cycles`` the system stops giving the benefit
    of the doubt and raises a continuous alarm instead.
    """
    settings = get_settings()
    if alarm_duration_seconds is None:
        alarm_duration_seconds = settings.alarm_duration_seconds
    if escalate_after_cycles is None:
        escalate_after_cycles = settings.alarm_escalate_after_cycles
    if recognition_state is RecognitionState.PERMANENT_FAMILIAR:
        return SecurityDecision(
            SecurityAction.NONE, None, "permanent_familiar_no_alarm", "INFO"
        )

    if recognition_state is RecognitionState.TEMPORARY_FAMILIAR:
        if already_alarming:
            return NO_ACTION
        mode = policy.temporary_familiar
        if mode == "continuous":
            return SecurityDecision(
                SecurityAction.START_CONTINUOUS_ALARM,
                AlertType.CONTINUOUS_ALARM,
                "temporary_familiar_in_alarm_zone",
                "WARNING",
            )
        if mode == "beep":
            return SecurityDecision(
                SecurityAction.BEEP,
                AlertType.BEEP,
                "temporary_familiar_in_alarm_zone",
                "NOTICE",
            )
        return SecurityDecision(SecurityAction.NONE, None, "temporary_familiar_policy_none")

    if recognition_state is RecognitionState.UNFAMILIAR:
        if already_alarming:
            return NO_ACTION
        if not unknown_confirmed:
            # Judged unknown, but not yet for long enough to act on.
            return SecurityDecision(
                SecurityAction.NONE, None, "unfamiliar_pending_confirmation", "NOTICE"
            )
        if alarm_cycles >= max(1, escalate_after_cycles):
            return SecurityDecision(
                SecurityAction.START_CONTINUOUS_ALARM,
                AlertType.CONTINUOUS_ALARM,
                "unfamiliar_person_repeatedly_in_alarm_zone",
                "CRITICAL",
            )
        return SecurityDecision(
            SecurityAction.START_TIMED_ALARM,
            AlertType.CONTINUOUS_ALARM,
            "unfamiliar_person_in_alarm_zone",
            "CRITICAL",
            duration_seconds=alarm_duration_seconds,
        )

    if recognition_state is RecognitionState.FACE_UNRECOGNIZABLE:
        # Recognition was attempted and failed. That is not proof of an
        # intruder, so by default we do not alarm.
        if policy.unrecognizable_policy == "alarm":
            if already_alarming:
                return NO_ACTION
            return SecurityDecision(
                SecurityAction.START_CONTINUOUS_ALARM,
                AlertType.CONTINUOUS_ALARM,
                "unrecognizable_face_in_alarm_zone_policy",
                "WARNING",
            )
        return SecurityDecision(
            SecurityAction.NONE, None, "unrecognizable_face_no_alarm_by_policy", "NOTICE"
        )

    # NO_FACE / UNKNOWN_PENDING_RECOGNITION: nothing has been concluded.
    return SecurityDecision(
        SecurityAction.NONE, None, "recognition_inconclusive_no_alarm", "INFO"
    )


def evaluate_recognition(
    *, recognition_state: RecognitionState, policy: AlertPolicy
) -> SecurityDecision:
    """Decide the one-off feedback given the moment an identity is settled.

    A temporary familiar gets a single beep on recognition; a permanent
    familiar is silent unless the site configures otherwise.
    """
    if not policy.beep_on_recognition:
        return NO_ACTION
    if recognition_state is RecognitionState.TEMPORARY_FAMILIAR:
        return SecurityDecision(
            SecurityAction.BEEP, AlertType.BEEP, "temporary_familiar_recognized", "NOTICE"
        )
    if (
        recognition_state is RecognitionState.PERMANENT_FAMILIAR
        and policy.permanent_familiar == "beep"
    ):
        return SecurityDecision(
            SecurityAction.BEEP, AlertType.BEEP, "permanent_familiar_recognized", "INFO"
        )
    return NO_ACTION


def should_attempt_face_recognition(
    *, object_class: str, in_recognition_zone: bool, recognition_enabled: bool
) -> bool:
    """Faces are only processed for people who are inside Proximity A."""
    return object_class == "person" and in_recognition_zone and recognition_enabled
