"""Proximity estimation and zone classification.

Design note - honesty about what this measures
----------------------------------------------
A single uncalibrated camera cannot measure metric distance. What the MVP
estimator does is convert *apparent object size* into an approximate distance
using a pinhole model plus per-source calibration (vertical FOV and a
reference object height). That is a defensible approximation, not a
measurement, and every result carries a ``confidence`` and a ``method`` so the
UI and the rule engine can treat it accordingly.

The interface exists so a later version can swap in monocular depth, stereo,
homography-to-ground-plane or real camera intrinsics without the security
engine changing at all.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from app.core.exceptions import ValidationError
from app.intelligence.types import BBox, ProximityResult
from app.models.enums import ProximityZone


@dataclass(frozen=True, slots=True)
class ProximityZones:
    """The two configured boundaries for one source.

    Invariant: ``recognition_distance`` (Proximity A) is FARTHER from the
    camera than ``alarm_distance`` (Proximity B).
    """

    recognition_distance: float   # Proximity A, metres
    alarm_distance: float         # Proximity B, metres

    def __post_init__(self) -> None:
        if self.recognition_distance <= 0:
            raise ValidationError("Proximity A must be greater than 0")
        if self.alarm_distance <= 0:
            raise ValidationError("Proximity B must be greater than 0")
        if self.recognition_distance <= self.alarm_distance:
            raise ValidationError(
                "Proximity A (recognition zone) must be greater than "
                "Proximity B (alarm zone)"
            )

    def classify(self, distance_m: float | None) -> ProximityZone:
        if distance_m is None:
            return ProximityZone.UNKNOWN
        if distance_m <= self.alarm_distance:
            return ProximityZone.ZONE_B
        if distance_m <= self.recognition_distance:
            return ProximityZone.ZONE_A
        return ProximityZone.FAR

    @staticmethod
    def is_within_recognition(zone: ProximityZone) -> bool:
        """Zone B is inside zone A, so both permit recognition."""
        return zone in (ProximityZone.ZONE_A, ProximityZone.ZONE_B)

    @staticmethod
    def is_within_alarm(zone: ProximityZone) -> bool:
        return zone is ProximityZone.ZONE_B


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Per-source geometry used by the MVP estimator."""

    vertical_fov_deg: float = 55.0
    reference_height_m: float = 1.7        # average standing person
    # Multiplier applied to the final distance; lets an operator calibrate a
    # source against a known landmark without touching code.
    distance_scale: float = 1.0
    # Fraction of the reference height typically visible (1.0 = full body).
    visible_height_fraction: float = 1.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "CameraCalibration":
        data = data or {}
        return cls(
            vertical_fov_deg=float(data.get("vertical_fov_deg", 55.0)),
            reference_height_m=float(data.get("reference_height_m", 1.7)),
            distance_scale=float(data.get("distance_scale", 1.0)),
            visible_height_fraction=float(data.get("visible_height_fraction", 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            "vertical_fov_deg": self.vertical_fov_deg,
            "reference_height_m": self.reference_height_m,
            "distance_scale": self.distance_scale,
            "visible_height_fraction": self.visible_height_fraction,
        }


class ProximityEstimator(abc.ABC):
    """Estimates camera-relative distance for one detected object."""

    name = "proximity"

    @abc.abstractmethod
    def estimate(
        self,
        bbox: BBox,
        *,
        frame_width: int,
        frame_height: int,
        object_class: str = "person",
        zones: ProximityZones | None = None,
    ) -> ProximityResult: ...
