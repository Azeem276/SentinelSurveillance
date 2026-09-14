"""Pinhole (apparent-size) proximity estimator.

Model
-----
For a pinhole camera the focal length in pixels is

    f_px = (frame_height / 2) / tan(vertical_fov / 2)

and an object of real height ``H`` metres projecting to ``h`` pixels sits at

    distance = H * f_px / h

Accuracy depends on the object standing upright, being fully visible and
matching the reference height. Confidence is reduced when the box is clipped
by the frame edge (partially visible objects look closer than they are) and
when the class has a wide natural size range.
"""
from __future__ import annotations

import math

from app.core.config import get_settings
from app.intelligence.proximity.base import (
    CameraCalibration, ProximityEstimator, ProximityZones,
)
from app.intelligence.types import BBox, ProximityResult
from app.models.enums import ProximityZone

# Typical real-world heights in metres, used when the object is not a person.
REFERENCE_HEIGHTS: dict[str, float] = {
    "person": 1.70,
    "bicycle": 1.10,
    "motorcycle": 1.30,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.50,
    "dog": 0.55,
    "cat": 0.30,
    "horse": 1.60,
    "sheep": 0.90,
    "cow": 1.50,
    "bird": 0.25,
    "bear": 1.20,
}

# How much we trust the reference height for each class.
CLASS_CONFIDENCE: dict[str, float] = {
    "person": 0.75,
    "car": 0.6,
    "truck": 0.45,
    "bus": 0.45,
    "motorcycle": 0.5,
    "bicycle": 0.5,
    "dog": 0.4,
    "cat": 0.35,
}

MIN_DISTANCE_M = 0.3
MAX_DISTANCE_M = 200.0


class PinholeProximityEstimator(ProximityEstimator):
    name = "pinhole-apparent-size"

    def __init__(self, calibration: CameraCalibration | None = None) -> None:
        settings = get_settings()
        self.calibration = calibration or CameraCalibration(
            vertical_fov_deg=settings.default_vertical_fov_deg,
            reference_height_m=settings.reference_person_height_m,
        )

    def focal_length_px(self, frame_height: int) -> float:
        fov_rad = math.radians(max(1.0, min(179.0, self.calibration.vertical_fov_deg)))
        return (frame_height / 2.0) / math.tan(fov_rad / 2.0)

    def _reference_height(self, object_class: str) -> float:
        if object_class == "person":
            return self.calibration.reference_height_m
        return REFERENCE_HEIGHTS.get(object_class, self.calibration.reference_height_m)

    def estimate(
        self,
        bbox: BBox,
        *,
        frame_width: int,
        frame_height: int,
        object_class: str = "person",
        zones: ProximityZones | None = None,
    ) -> ProximityResult:
        box_h = bbox.height
        if box_h <= 1.0 or frame_height <= 0:
            return ProximityResult(
                distance_m=MAX_DISTANCE_M,
                zone=ProximityZone.UNKNOWN,
                confidence=0.0,
                method=self.name,
            )

        visible_fraction = max(0.05, min(1.0, self.calibration.visible_height_fraction))
        real_height = self._reference_height(object_class) * visible_fraction
        distance = real_height * self.focal_length_px(frame_height) / box_h
        distance *= max(0.05, self.calibration.distance_scale)
        distance = max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance))

        confidence = CLASS_CONFIDENCE.get(object_class, 0.3)

        # A box touching the frame edge is probably cropped: its apparent
        # height understates the object, biasing the distance too large.
        margin = 2.0
        clipped_edges = sum(
            (
                bbox.y1 <= margin,
                bbox.y2 >= frame_height - margin,
                bbox.x1 <= margin,
                bbox.x2 >= frame_width - margin,
            )
        )
        if clipped_edges:
            confidence *= max(0.25, 1.0 - 0.25 * clipped_edges)

        # Very small boxes are dominated by detector jitter.
        if box_h < 0.05 * frame_height:
            confidence *= 0.6

        zone = zones.classify(distance) if zones else ProximityZone.UNKNOWN
        return ProximityResult(
            distance_m=round(distance, 3),
            zone=zone,
            confidence=round(min(1.0, confidence), 3),
            method=self.name,
        )

    # Useful for the settings UI: what box height corresponds to a boundary?
    def box_height_for_distance(
        self, distance_m: float, frame_height: int, object_class: str = "person"
    ) -> float:
        real_height = self._reference_height(object_class) * max(
            0.05, min(1.0, self.calibration.visible_height_fraction)
        )
        scaled = max(MIN_DISTANCE_M, distance_m) / max(0.05, self.calibration.distance_scale)
        return real_height * self.focal_length_px(frame_height) / scaled


def build_estimator(calibration: dict | None) -> PinholeProximityEstimator:
    return PinholeProximityEstimator(CameraCalibration.from_dict(calibration))
