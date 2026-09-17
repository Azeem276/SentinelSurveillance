"""Face quality gate.

This is what keeps the system honest. A face that is too small, too blurry or
badly exposed is reported as FACE_UNRECOGNIZABLE - it is never pushed through
the embedder and guessed at. "We could not evaluate this face" and "we
evaluated it and it matches nobody" are different security facts, and only the
second one justifies an unfamiliar-person alarm.

Pose is treated differently from the other metrics. A steeply angled face is
still *evaluable* - it is simply harder - so under the soft gate it is graded
rather than binned: the crop is embedded, contributes to the track's face
profile, and carries a ``threshold_penalty`` that the matcher adds to the
similarity bar it has to clear. Throwing angled faces away was the single
biggest reason the same person, seen from a new angle, became a stranger.
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
        soft_pose_gate: bool | None = None,
        pose_threshold_penalty: float | None = None,
    ) -> None:
        s = get_settings()
        self.min_pixels = s.face_min_pixels if min_pixels is None else min_pixels
        self.min_blur = s.face_min_blur if min_blur is None else min_blur
        self.min_brightness = s.face_min_brightness if min_brightness is None else min_brightness
        self.max_brightness = s.face_max_brightness if max_brightness is None else max_brightness
        self.min_confidence = s.face_min_confidence if min_confidence is None else min_confidence
        self.max_yaw_ratio = s.face_max_yaw_ratio if max_yaw_ratio is None else max_yaw_ratio
        self.soft_pose_gate = (
            s.face_pose_soft_gate if soft_pose_gate is None else soft_pose_gate
        )
        self.pose_threshold_penalty = (
            s.face_pose_threshold_penalty
            if pose_threshold_penalty is None
            else pose_threshold_penalty
        )

    # ------------------------------------------------------------- metrics
    @staticmethod
    def blur_score(gray: np.ndarray) -> float:
        """Variance of the Laplacian: higher means sharper."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def brightness(gray: np.ndarray) -> float:
        return float(np.mean(gray))

    @staticmethod
    def signed_yaw(landmarks: np.ndarray | None) -> float:
        """Head yaw in [-1, 1] from the 5 YuNet landmarks.

        0.0 means the nose sits midway between the eyes (frontal). The sign
        tells us *which* way the head is turned, which is what lets the face
        profile deliberately collect left, frontal and right views of the same
        person rather than a hundred near-duplicates of whichever side the
        camera happened to favour.
        """
        if landmarks is None or len(landmarks) < 3:
            return 0.0
        right_eye, left_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_center_x = (right_eye[0] + left_eye[0]) / 2.0
        eye_distance = abs(left_eye[0] - right_eye[0])
        if eye_distance < 1e-3:
            return 1.0
        ratio = (nose[0] - eye_center_x) / (eye_distance / 2.0)
        return float(max(-1.0, min(1.0, ratio)))

    @classmethod
    def yaw_ratio(cls, landmarks: np.ndarray | None) -> float:
        """Magnitude of the yaw, 0.0 (frontal) .. 1.0 (strong profile)."""
        return abs(cls.signed_yaw(landmarks))

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
        signed = self.signed_yaw(face.landmarks)
        yaw = abs(signed)

        # Hard failures: the crop genuinely cannot be evaluated.
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

        # Pose: graded under the soft gate, a hard failure otherwise.
        steep = yaw > self.max_yaw_ratio
        penalty = 0.0
        if steep:
            if self.soft_pose_gate:
                # Scale the extra similarity required with how far past the
                # comfortable range this face is.
                over = (yaw - self.max_yaw_ratio) / max(1e-6, 1.0 - self.max_yaw_ratio)
                penalty = round(self.pose_threshold_penalty * min(1.0, over), 4)
            else:
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

        if not reasons:
            reason = "soft_pose" if steep and self.soft_pose_gate else "ok"
        else:
            reason = ",".join(reasons)

        return FaceQuality(
            ok=not reasons,
            score=round(float(score), 4),
            reason=reason,
            face_pixels=face_pixels,
            blur=blur,
            brightness=bright,
            detection_confidence=face.confidence,
            yaw=round(signed, 4),
            threshold_penalty=penalty,
        )
