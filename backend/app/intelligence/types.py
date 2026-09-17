"""Value objects shared across the intelligence engine.

These are plain dataclasses, not ORM rows: the pipeline runs at frame rate and
must not touch the database on the hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from app.models.enums import ProximityZone, RecognitionState

UNIDENTIFIED_MOVEMENT = "UNIDENTIFIED_MOVEMENT"


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def scaled(self, factor: float) -> "BBox":
        return BBox(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    def clipped(self, width: int, height: int) -> "BBox":
        return BBox(
            max(0.0, min(self.x1, width - 1.0)),
            max(0.0, min(self.y1, height - 1.0)),
            max(0.0, min(self.x2, float(width))),
            max(0.0, min(self.y2, float(height))),
        )

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(slots=True)
class ObjectDetection:
    """A single detector output for one frame (no identity, no track yet)."""

    bbox: BBox
    object_class: str
    confidence: float
    class_id: int = -1
    track_key: int | None = None


@dataclass(slots=True)
class TrackedObject:
    """A detection associated with a persistent tracker id."""

    track_key: int
    bbox: BBox
    object_class: str
    confidence: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    is_confirmed: bool = False

    @property
    def is_person(self) -> bool:
        return self.object_class == "person"


@dataclass(slots=True)
class MotionResult:
    is_motion: bool
    area_ratio: float
    regions: list[BBox] = field(default_factory=list)


@dataclass(slots=True)
class ProximityResult:
    """Estimated camera-relative distance and the zone it falls in.

    ``distance_m`` is an approximation from calibrated scene geometry, not a
    measured physical distance; ``confidence`` reflects that.
    """

    distance_m: float
    zone: ProximityZone
    confidence: float
    method: str


@dataclass(slots=True)
class FaceQuality:
    ok: bool
    score: float
    reason: str
    face_pixels: int
    blur: float
    brightness: float
    detection_confidence: float
    # Signed head yaw in [-1, 1]: negative is turned one way, positive the
    # other, 0 is frontal. Used to spread a track's face profile across poses.
    yaw: float = 0.0
    # Extra cosine similarity this crop must clear because of its pose.
    threshold_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "score": round(self.score, 4),
            "reason": self.reason,
            "face_pixels": self.face_pixels,
            "blur": round(self.blur, 2),
            "brightness": round(self.brightness, 2),
            "detection_confidence": round(self.detection_confidence, 4),
            "yaw": round(self.yaw, 4),
            "threshold_penalty": round(self.threshold_penalty, 4),
        }


@dataclass(slots=True)
class DetectedFace:
    bbox: BBox
    confidence: float
    landmarks: np.ndarray | None = None
    crop: np.ndarray | None = None
    aligned: np.ndarray | None = None
    quality: FaceQuality | None = None
    # Detector-native row (YuNet layout: 4 bbox + 10 landmark + 1 score) kept
    # in full-frame coordinates so the embedder can align the crop precisely.
    raw_row: np.ndarray | None = None


@dataclass(slots=True)
class RecognitionMatch:
    """Outcome of comparing one embedding against the identity index."""

    state: RecognitionState
    identity_id: int | None = None
    label: str | None = None
    score: float = 0.0
    runner_up_score: float = 0.0


@dataclass(slots=True)
class FrameAnalysis:
    """Everything the pipeline learned about one processed frame."""

    source_uid: str
    source_id: int
    frame_number: int
    timestamp: datetime
    width: int
    height: int
    objects: list[dict] = field(default_factory=list)
    motion: bool = False
    motion_area_ratio: float = 0.0
    inference_ms: float = 0.0
    detection_ms: float = 0.0
    face_ms: float = 0.0
    ran_detector: bool = False
