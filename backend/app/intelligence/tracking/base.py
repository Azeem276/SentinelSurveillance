"""ObjectTracker interface."""
from __future__ import annotations

import abc

from app.intelligence.types import ObjectDetection, TrackedObject


class ObjectTracker(abc.ABC):
    name: str = "tracker"

    @abc.abstractmethod
    def update(self, detections: list[ObjectDetection]) -> list[TrackedObject]:
        """Associate this frame's detections with persistent track ids."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget all state (source restarted)."""

    @property
    @abc.abstractmethod
    def active_tracks(self) -> list[TrackedObject]: ...
