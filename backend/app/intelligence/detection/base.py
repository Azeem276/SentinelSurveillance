"""ObjectDetector interface.

Swapping YOLO for another detector means implementing this class - nothing in
the pipeline, event engine or API refers to Ultralytics.
"""
from __future__ import annotations

import abc

import numpy as np

from app.intelligence.types import ObjectDetection


class ObjectDetector(abc.ABC):
    name: str = "object-detector"
    supports_native_tracking: bool = False

    @abc.abstractmethod
    def load(self) -> None:
        """Load weights. Must be idempotent and safe to call lazily."""

    @abc.abstractmethod
    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Detect objects in a BGR frame."""

    def track(self, frame: np.ndarray) -> list[ObjectDetection]:
        """Detect and assign tracker ids in one pass.

        Detectors that cannot do this fall back to plain detection, and the
        pipeline then uses an external tracker.
        """
        return self.detect(frame)

    def reset_tracker(self) -> None:
        """Drop any tracker state (called when a source restarts)."""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    def class_names(self) -> list[str]:
        return []

    def warmup(self) -> None:
        """Run one throwaway inference so the first real frame is not slow."""
        if not self.is_loaded:
            self.load()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.detect(blank)
        except Exception:  # pragma: no cover - warmup must never break startup
            pass
