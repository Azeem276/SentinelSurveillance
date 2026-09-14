"""YuNet face detector (OpenCV Zoo, MIT) running fully locally."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import ModelNotAvailableError
from app.core.logging import FACE_RECOGNITION_ERROR, get_logger
from app.intelligence.face.base import FaceDetector
from app.intelligence.model_registry import ensure_model, log_model_loaded
from app.intelligence.types import BBox, DetectedFace

log = get_logger(__name__)


class YuNetFaceDetector(FaceDetector):
    """Detects any number of faces in an image.

    YuNet is input-size sensitive, so the input size is re-set whenever the
    frame shape changes. Very small crops are upscaled first: a 40px face that
    would be missed at native size is usually found at 2x.
    """

    name = "yunet"

    def __init__(
        self,
        *,
        score_threshold: float | None = None,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        min_upscale_width: int = 200,
    ) -> None:
        settings = get_settings()
        self.score_threshold = (
            settings.face_min_confidence if score_threshold is None else score_threshold
        )
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.min_upscale_width = min_upscale_width
        self._model = None
        self._input_size: tuple[int, int] | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        weights = ensure_model("face_detector")
        try:
            model = cv2.FaceDetectorYN.create(
                model=str(weights),
                config="",
                input_size=(320, 320),
                score_threshold=float(self.score_threshold),
                nms_threshold=float(self.nms_threshold),
                top_k=int(self.top_k),
            )
        except Exception as exc:
            raise ModelNotAvailableError(
                f"failed to load face detector {weights.name}: {exc}"
            ) from exc
        self._model = model
        self._input_size = (320, 320)
        log_model_loaded("face_detector", "cpu", weights)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        if image is None or image.size == 0:
            return []
        if self._model is None:
            self.load()

        work = image
        scale = 1.0
        h, w = work.shape[:2]
        if w < self.min_upscale_width and w > 0:
            scale = min(4.0, self.min_upscale_width / float(w))
            work = cv2.resize(work, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)
            h, w = work.shape[:2]

        with self._lock:
            if self._input_size != (w, h):
                self._model.setInputSize((w, h))
                self._input_size = (w, h)
            try:
                _retval, faces = self._model.detect(work)
            except Exception as exc:
                log.warning(FACE_RECOGNITION_ERROR, stage="detect", error=str(exc))
                return []

        if faces is None:
            return []

        inv = 1.0 / scale
        results: list[DetectedFace] = []
        for row in faces:
            x, y, bw, bh = (float(v) for v in row[:4])
            confidence = float(row[14]) if len(row) > 14 else 1.0
            bbox = BBox(x * inv, y * inv, (x + bw) * inv, (y + bh) * inv).clipped(
                image.shape[1], image.shape[0]
            )
            if bbox.width < 4 or bbox.height < 4:
                continue
            # 5 landmarks: right eye, left eye, nose, right mouth, left mouth.
            landmarks = np.array(row[4:14], dtype=np.float32).reshape(5, 2) * inv
            crop = self._crop(image, bbox)

            # Rescale geometry back to full-frame coordinates but leave the
            # score column untouched; SFace.alignCrop consumes this row.
            raw_row = np.array(row, dtype=np.float32).copy()
            if scale != 1.0:
                raw_row[:14] *= inv

            results.append(
                DetectedFace(
                    bbox=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                    crop=crop,
                    raw_row=raw_row,
                )
            )
        return results

    @staticmethod
    def _crop(image: np.ndarray, bbox: BBox, margin: float = 0.15) -> np.ndarray | None:
        h, w = image.shape[:2]
        mx = bbox.width * margin
        my = bbox.height * margin
        x1 = int(max(0, bbox.x1 - mx))
        y1 = int(max(0, bbox.y1 - my))
        x2 = int(min(w, bbox.x2 + mx))
        y2 = int(min(h, bbox.y2 + my))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()
