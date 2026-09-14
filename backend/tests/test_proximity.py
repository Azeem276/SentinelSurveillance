"""Proximity zones and estimation.

The brief's core invariant: Proximity A is the RECOGNITION zone and is
FARTHER from the camera than Proximity B, the ALARM zone.
"""
from __future__ import annotations

import math

import pytest

from app.core.exceptions import ValidationError
from app.intelligence.proximity.base import CameraCalibration, ProximityZones
from app.intelligence.proximity.estimator import PinholeProximityEstimator
from app.intelligence.types import BBox
from app.models.enums import ProximityZone

FRAME_W, FRAME_H = 1280, 720


@pytest.fixture()
def zones() -> ProximityZones:
    return ProximityZones(recognition_distance=10.0, alarm_distance=3.0)


@pytest.fixture()
def estimator() -> PinholeProximityEstimator:
    return PinholeProximityEstimator(CameraCalibration())


# ------------------------------------------------------------- invariants
class TestZoneInvariants:
    def test_a_must_exceed_b(self):
        with pytest.raises(ValidationError, match="greater than"):
            ProximityZones(recognition_distance=3.0, alarm_distance=10.0)

    def test_a_equal_to_b_is_rejected(self):
        with pytest.raises(ValidationError):
            ProximityZones(recognition_distance=5.0, alarm_distance=5.0)

    @pytest.mark.parametrize("a,b", [(10.0, 0.0), (10.0, -1.0), (0.0, -1.0)])
    def test_non_positive_distances_rejected(self, a, b):
        with pytest.raises(ValidationError):
            ProximityZones(recognition_distance=a, alarm_distance=b)

    def test_valid_zones_accepted(self):
        z = ProximityZones(15.0, 5.0)
        assert z.recognition_distance == 15.0
        assert z.alarm_distance == 5.0


# ------------------------------------------------------- zone classification
class TestZoneClassification:
    def test_far_outside_a(self, zones):
        assert zones.classify(25.0) is ProximityZone.FAR

    def test_enters_a(self, zones):
        assert zones.classify(9.9) is ProximityZone.ZONE_A

    def test_exactly_on_a_boundary_is_inside(self, zones):
        assert zones.classify(10.0) is ProximityZone.ZONE_A

    def test_enters_b(self, zones):
        assert zones.classify(2.5) is ProximityZone.ZONE_B

    def test_exactly_on_b_boundary_is_inside(self, zones):
        assert zones.classify(3.0) is ProximityZone.ZONE_B

    def test_unknown_when_distance_missing(self, zones):
        assert zones.classify(None) is ProximityZone.UNKNOWN

    def test_zone_b_is_nested_inside_zone_a(self, zones):
        """Being in the alarm zone also means being close enough to recognise."""
        assert ProximityZones.is_within_recognition(ProximityZone.ZONE_B)
        assert ProximityZones.is_within_recognition(ProximityZone.ZONE_A)
        assert not ProximityZones.is_within_recognition(ProximityZone.FAR)
        assert ProximityZones.is_within_alarm(ProximityZone.ZONE_B)
        assert not ProximityZones.is_within_alarm(ProximityZone.ZONE_A)

    def test_full_approach_and_departure_sequence(self, zones):
        """Person outside A -> enters A -> enters B -> exits B -> exits A."""
        distances = [30.0, 12.0, 9.0, 6.0, 2.8, 1.5, 2.9, 5.0, 9.5, 14.0]
        observed = [zones.classify(d) for d in distances]
        assert observed == [
            ProximityZone.FAR, ProximityZone.FAR,
            ProximityZone.ZONE_A, ProximityZone.ZONE_A,
            ProximityZone.ZONE_B, ProximityZone.ZONE_B, ProximityZone.ZONE_B,
            ProximityZone.ZONE_A, ProximityZone.ZONE_A,
            ProximityZone.FAR,
        ]


# ---------------------------------------------------------------- estimator
class TestPinholeEstimator:
    def test_bigger_box_means_closer(self, estimator, zones):
        far = estimator.estimate(BBox(0, 0, 40, 100), frame_width=FRAME_W,
                                 frame_height=FRAME_H, zones=zones)
        near = estimator.estimate(BBox(0, 0, 160, 400), frame_width=FRAME_W,
                                  frame_height=FRAME_H, zones=zones)
        assert near.distance_m < far.distance_m

    def test_distance_matches_the_pinhole_model(self, estimator):
        """A box of the analytically derived height returns that distance."""
        target = 8.0
        height = estimator.box_height_for_distance(target, FRAME_H, "person")
        result = estimator.estimate(
            BBox(500, 100, 540, 100 + height), frame_width=FRAME_W, frame_height=FRAME_H
        )
        assert result.distance_m == pytest.approx(target, rel=0.02)

    def test_focal_length_follows_fov(self):
        narrow = PinholeProximityEstimator(CameraCalibration(vertical_fov_deg=30.0))
        wide = PinholeProximityEstimator(CameraCalibration(vertical_fov_deg=90.0))
        assert narrow.focal_length_px(720) > wide.focal_length_px(720)
        expected = (720 / 2) / math.tan(math.radians(30.0) / 2)
        assert narrow.focal_length_px(720) == pytest.approx(expected)

    def test_distance_scale_calibration_applies(self):
        base = PinholeProximityEstimator(CameraCalibration())
        scaled = PinholeProximityEstimator(CameraCalibration(distance_scale=2.0))
        box = BBox(0, 0, 60, 150)
        a = base.estimate(box, frame_width=FRAME_W, frame_height=FRAME_H)
        b = scaled.estimate(box, frame_width=FRAME_W, frame_height=FRAME_H)
        assert b.distance_m == pytest.approx(a.distance_m * 2, rel=0.01)

    def test_class_specific_reference_heights(self, estimator):
        """A car and a person with the same box height are not the same distance."""
        box = BBox(0, 0, 120, 200)
        person = estimator.estimate(box, frame_width=FRAME_W, frame_height=FRAME_H,
                                    object_class="person")
        car = estimator.estimate(box, frame_width=FRAME_W, frame_height=FRAME_H,
                                 object_class="car")
        assert person.distance_m != car.distance_m

    def test_confidence_drops_for_clipped_boxes(self, estimator):
        whole = estimator.estimate(BBox(400, 200, 460, 400), frame_width=FRAME_W,
                                   frame_height=FRAME_H)
        clipped = estimator.estimate(BBox(0, 0, 60, 200), frame_width=FRAME_W,
                                     frame_height=FRAME_H)
        assert clipped.confidence < whole.confidence

    def test_degenerate_box_is_unknown_not_a_guess(self, estimator, zones):
        result = estimator.estimate(BBox(10, 10, 10, 10), frame_width=FRAME_W,
                                    frame_height=FRAME_H, zones=zones)
        assert result.zone is ProximityZone.UNKNOWN
        assert result.confidence == 0.0

    def test_result_reports_its_method(self, estimator):
        result = estimator.estimate(BBox(0, 0, 50, 120), frame_width=FRAME_W,
                                    frame_height=FRAME_H)
        assert result.method == "pinhole-apparent-size"
        assert 0.0 <= result.confidence <= 1.0

    def test_per_source_zones_are_independent(self, estimator):
        """Different cameras can have different physical layouts."""
        front_door = ProximityZones(10.0, 3.0)
        back_gate = ProximityZones(15.0, 5.0)
        assert front_door.classify(4.0) is ProximityZone.ZONE_A
        assert back_gate.classify(4.0) is ProximityZone.ZONE_B
