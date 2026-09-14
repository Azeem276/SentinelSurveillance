"""Face quality gate.

This is what keeps the system honest. A face that is too small, too blurry,
badly exposed or steeply angled is reported as FACE_UNRECOGNIZABLE - it is
never pushed through the embedder and guessed at. "We could not evaluate this
face" and "we evaluated it and it matches nobody" are different security
facts, and only the second one justifies an unfamiliar-person alarm.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.core.config import get_settings
from app.intelligence.face.base import FaceQualityAssessor
from app.intelligence.types import DetectedFace, FaceQuality


class DefaultFaceQualityAssessor(FaceQualityAssessor):
    """Threshold-based gate over pixel size, sharpness, exposure and pose."""

    def __init__(
        self,
        *,
        min_pixels: int | None = None,
        min_blur: float | None = None,
        min_brightness: float | None = None,
        max_brightness: float | None = None,
        min_confidence: float | None = None,
        max_yaw_ratio: float | None = None,
    ) -> None:
        s = get_settings()
        self.min_pixels = s.face_min_pixels if min_pixels is None else min_pixels
        self.min_blur = s.face_min_blur if min_blur is None else min_blur
        self.min_brightness = s.face_min_brightness if min_brightness is None else min_brightness
        self.max_brightness = s.face_max_brightness if max_brightness is None else max_brightness
        self.min_confidence = s.face_min_confidence if min_confidence is None else min_confidence
        self.max_yaw_ratio = s.face_max_yaw_ratio if max_yaw_ratio is None else max_yaw_ratio

    # ------------------------------------------------------------- metrics
    @staticmethod
    def blur_score(gray: np.ndarray) -> float:
        """Variance of the Laplacian: higher means sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))

    @staticmethod
    def yaw_ratio(landmarks: np.ndarray | None) -> float:
        """Rough head-yaw proxy from the 5 YuNet landmarks.

        0.0 means the nose sits midway between the eyes (frontal); values
        approaching 1.0 mean a strong profile view.
        """
        if landmarks is None or len(landmarks) < 3:
            return 0.0
        right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_center_x = (right_eye[0] + left_eye[0]) / 2.0
        eye_distance = abs(left_eye[0] - right_eye[0])
        if eye_distance < 1e-3:
            return 1.0
        return float(min(1.0, abs(nose[0] - eye_center_x) / (eye_distance / 2.0)))

    # -------------------------------------------------------------- assess
    def assess(self, face: DetectedFace, image: np.ndarray) -> FaceQuality:
        crop = face.crop
        if crop is None or crop.size == 0:
            return FaceQuality(
                ok=False, score=0.0, reason="no_crop", face_pixels=0,
                blur=0.0, brightness=0.0, detection_confidence=face.confidence,
            )

        face_pixels = int(min(face.bbox.width, face.bbox.height))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blur = self.blur_score(gray)
        bright = self.brightness(gray)
        yaw = self.yaw_ratio(face.landmarks)

        reasons: list[str] = []
        if face.confidence < self.min_confidence:
            reasons.append("low_detection_confidence")
        if face_pixels < self.min_pixels:
            reasons.append("face_too_small")
        if blur < self.min_blur:
            reasons.append("too_blurry")
        if bright < self.min_brightness:
            reasons.append("too_dark")
        elif bright > self.max_brightness:
            reasons.append("overexposed")
        if yaw > self.max_yaw_ratio:
            reasons.append("extreme_pose")

        # Normalised sub-scores, combined into a single 0..1 quality score.
        size_score = min(1.0, face_pixels / max(1.0, self.min_pixels * 2.0))
        blur_norm = min(1.0, blur / max(1.0, self.min_blur * 4.0))
        mid = (self.min_brightness + self.max_brightness) / 2.0
        bright_norm = max(0.0, 1.0 - abs(bright - mid) / mid)
        pose_norm = max(0.0, 1.0 - yaw)
        score = (
            0.35 * size_score
            + 0.30 * blur_norm
            + 0.15 * bright_norm
            + 0.10 * pose_norm
            + 0.10 * min(1.0, face.confidence)
        )

        return FaceQuality(
            ok=not reasons,
            score=round(float(score), 4),
            reason="ok" if not reasons else ",".join(reasons),
            face_pixels=face_pixels,
            blur=blur,
            brightness=bright,
            detection_confidence=face.confidence,
        )
