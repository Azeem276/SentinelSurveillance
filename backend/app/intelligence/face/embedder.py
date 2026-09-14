"""SFace embedder (OpenCV Zoo, Apache-2.0).

Produces a 128-d descriptor per face, entirely on this machine. Embeddings are
L2-normalised so cosine similarity is a plain dot product, which makes the
in-memory index a single matrix multiply.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from app.core.exceptions import ModelNotAvailableError
from app.core.logging import FACE_RECOGNITION_ERROR, get_logger
from app.intelligence.face.base import FaceEmbedder
from app.intelligence.model_registry import ensure_model, log_model_loaded
from app.intelligence.types import DetectedFace

log = get_logger(__name__)

SFACE_INPUT = (112, 112)


class SFaceEmbedder(FaceEmbedder):
    name = "sface"
    dim = 128

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        weights = ensure_model("face_embedder")
        try:
            model = cv2.FaceRecognizerSF.create(model=str(weights), config="")
        except Exception as exc:
            raise ModelNotAvailableError(
                f"failed to load face embedder {weights.name}: {exc}"
            ) from exc
        self._model = model
        log_model_loaded("face_embedder", "cpu", weights)

    # ----------------------------------------------------------- embedding
    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        """Align the face using its landmarks, then encode it."""
        if self._model is None:
            self.load()
        if image is None or image.size == 0:
            return None

        try:
            with self._lock:
                aligned = self._align(image, face)
                if aligned is None:
                    return None
                vector = self._model.feature(aligned)
        except Exception as exc:
            log.warning(FACE_RECOGNITION_ERROR, stage="embed", error=str(exc))
            return None

        return normalise(np.asarray(vector, dtype=np.float32).reshape(-1))

    def embed_image(self, face_image: np.ndarray) -> np.ndarray | None:
        """Encode an already-cropped face (used for dataset enrolment)."""
        if self._model is None:
            self.load()
        if face_image is None or face_image.size == 0:
            return None
        try:
            resized = cv2.resize(face_image, SFACE_INPUT, interpolation=cv2.INTER_AREA)
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            with self._lock:
                vector = self._model.feature(resized)
        except Exception as exc:
            log.warning(FACE_RECOGNITION_ERROR, stage="embed_image", error=str(exc))
            return None
        return normalise(np.asarray(vector, dtype=np.float32).reshape(-1))

    def _align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        """Landmark-based alignment; falls back to a plain resized crop."""
        if face.raw_row is not None:
            try:
                row = np.asarray(face.raw_row, dtype=np.float32).reshape(1, -1)
                aligned = self._model.alignCrop(image, row)
                if aligned is not None and aligned.size:
                    return aligned
            except Exception:
                pass  # fall through to the crop-based path
        crop = face.crop
        if crop is None or crop.size == 0:
            return None
        aligned = cv2.resize(crop, SFACE_INPUT, interpolation=cv2.INTER_AREA)
        if aligned.ndim == 2:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
        return aligned


def normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def to_bytes(vector: np.ndarray) -> bytes:
    """Serialise an embedding for the ``face_embeddings.vector`` column."""
    return np.asarray(vector, dtype="<f4").tobytes()


def from_bytes(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.size != dim:
        raise ValueError(f"embedding dimension mismatch: expected {dim}, got {vector.size}")
    return vector.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; both inputs are assumed normalised."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)
