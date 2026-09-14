"""Motion detection with hysteresis.

The raw frame-differencing signal is noisy, so the detector applies start/end
frame thresholds: motion must persist for ``start_frames`` to begin and be
absent for ``end_frames`` to end. That debouncing is what keeps one walk-past
from producing hundreds of motion events.
"""
from __future__ import annotations

import abc

import cv2
import numpy as np

from app.core.config import get_settings
from app.intelligence.types import BBox, MotionResult


class MotionDetector(abc.ABC):
    name = "motion"

    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> MotionResult: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_active(self) -> bool:
        """True while a debounced motion episode is open."""


class MOG2MotionDetector(MotionDetector):
    name = "mog2"

    def __init__(
        self,
        *,
        min_area_ratio: float | None = None,
        start_frames: int | None = None,
        end_frames: int | None = None,
        history: int = 300,
        var_threshold: float = 32.0,
        detect_shadows: bool = True,
        working_width: int = 480,
    ) -> None:
        settings = get_settings()
        self.min_area_ratio = (
            settings.motion_min_area_ratio if min_area_ratio is None else min_area_ratio
        )
        self.start_frames = settings.motion_start_frames if start_frames is None else start_frames
        self.end_frames = settings.motion_end_frames if end_frames is None else end_frames
        self.working_width = working_width
        self._history = history
        self._var_threshold = var_threshold
        self._detect_shadows = detect_shadows
        self._bg = self._new_subtractor()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._above = 0
        self._below = 0
        self._active = False
        self._warmup_frames = 0

    def _new_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=self._detect_shadows,
        )

    def process(self, frame: np.ndarray) -> MotionResult:
        if frame is None or frame.size == 0:
            return MotionResult(self._active, 0.0, [])

        h, w = frame.shape[:2]
        scale = 1.0
        work = frame
        if w > self.working_width:
            scale = self.working_width / float(w)
            work = cv2.resize(frame, (self.working_width, max(1, int(h * scale))))

        mask = self._bg.apply(work)
        # MOG2 marks shadows as 127; only hard foreground (255) counts.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        self._warmup_frames += 1
        total = float(mask.shape[0] * mask.shape[1])
        moving = float(np.count_nonzero(mask))
        area_ratio = moving / total if total else 0.0

        regions: list[BBox] = []
        min_pixels = self.min_area_ratio * total
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / scale if scale else 1.0
        for contour in contours:
            if cv2.contourArea(contour) < min_pixels:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            regions.append(BBox(x * inv, y * inv, (x + cw) * inv, (y + ch) * inv))

        # The background model needs a few frames before its output is usable.
        raw_motion = bool(regions) and self._warmup_frames > 5

        if raw_motion:
            self._above += 1
            self._below = 0
            if not self._active and self._above >= self.start_frames:
                self._active = True
        else:
            self._below += 1
            self._above = 0
            if self._active and self._below >= self.end_frames:
                self._active = False

        return MotionResult(is_motion=self._active, area_ratio=area_ratio, regions=regions)

    def reset(self) -> None:
        self._bg = self._new_subtractor()
        self._above = 0
        self._below = 0
        self._active = False
        self._warmup_frames = 0

    @property
    def is_active(self) -> bool:
        return self._active
