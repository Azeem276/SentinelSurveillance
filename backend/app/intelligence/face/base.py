"""Face pipeline interfaces: detection, quality, embedding, recognition."""
from __future__ import annotations

import abc

import numpy as np

from app.intelligence.types import DetectedFace, FaceQuality, RecognitionMatch


class FaceDetector(abc.ABC):
    name = "face-detector"

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a BGR image (multiple faces supported)."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...


class FaceQualityAssessor(abc.ABC):
    @abc.abstractmethod
    def assess(self, face: DetectedFace, image: np.ndarray) -> FaceQuality:
        """Decide whether this crop is good enough to attempt recognition."""


class FaceEmbedder(abc.ABC):
    name = "face-embedder"
    dim = 0

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray | None:
        """Return an L2-normalised embedding, or None on failure."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...


class FaceRecognizer(abc.ABC):
    @abc.abstractmethod
    def recognize(self, embedding: np.ndarray, *, threshold: float | None = None
                  ) -> RecognitionMatch:
        """Compare an embedding against the active identity index."""
