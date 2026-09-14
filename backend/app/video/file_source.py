"""FileVideoSource - reads frames from a video file on disk.

This is the MVP source. It is deliberately thin: all it does is turn a file
into frames at a controlled cadence, exactly like the RTSP source will.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from app.core.logging import get_logger
from app.video.base import SourceCapabilities, SourceUnavailableError, VideoSourceAdapter

log = get_logger(__name__)


def _fourcc_to_str(value: float) -> str | None:
    try:
        code = int(value)
        if code <= 0:
            return None
        return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()
    except Exception:  # pragma: no cover - defensive
        return None


class FileVideoSource(VideoSourceAdapter):
    """A pre-recorded clip presented as a generic video source.

    ``realtime`` paces reads to the clip's native FPS so downstream timing,
    recording and event durations behave like a live camera. ``loop`` restarts
    the clip when it ends, which keeps a demo/monitoring session going.
    """

    source_type = "FILE"

    def __init__(
        self,
        uid: str,
        path: Path | str,
        *,
        name: str | None = None,
        loop: bool = False,
        realtime: bool = True,
        max_fps: float | None = None,
    ) -> None:
        super().__init__(uid, name)
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime
        self.max_fps = max_fps
        self._cap: cv2.VideoCapture | None = None
        self._next_deadline: float | None = None
        self._frame_interval: float = 0.0
        self._loop_count = 0

    # ------------------------------------------------------------- lifecycle
    def _open(self) -> SourceCapabilities:
        if not self.path.exists():
            raise SourceUnavailableError(f"video file not found: {self.path}")
        if not self.path.is_file():
            raise SourceUnavailableError(f"not a file: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise SourceUnavailableError(f"OpenCV could not open: {self.path.name}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0 < fps < 240):
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        caps = SourceCapabilities(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=fps,
            frame_count=frame_count,
            is_live=False,
            seekable=True,
            codec=_fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
            extra={"path": self.path.name, "loop": self.loop},
        )
        if caps.width <= 0 or caps.height <= 0:
            cap.release()
            raise SourceUnavailableError(f"malformed video (no frame size): {self.path.name}")

        self._cap = cap
        effective_fps = min(fps, self.max_fps) if self.max_fps else fps
        self._frame_interval = 1.0 / effective_fps if effective_fps > 0 else 0.0
        self._next_deadline = None
        self._loop_count = 0
        log.info("source_opened", source=self.uid, type=self.source_type,
                 fps=fps, width=caps.width, height=caps.height, frames=frame_count)
        return caps

    def _pace(self) -> None:
        """Sleep so playback approximates real time."""
        if not self.realtime or self._frame_interval <= 0:
            return
        now = time.perf_counter()
        if self._next_deadline is None:
            self._next_deadline = now + self._frame_interval
            return
        delay = self._next_deadline - now
        if delay > 0:
            time.sleep(min(delay, 1.0))
            self._next_deadline += self._frame_interval
        else:
            # We fell behind; resynchronise instead of accumulating debt.
            self._next_deadline = now + self._frame_interval

    def _read(self) -> np.ndarray | None:
        cap = self._cap
        if cap is None:
            raise SourceUnavailableError("source is not open")
        self._pace()
        ok, frame = cap.read()
        if not ok or frame is None:
            if self.loop and self.reset():
                self._loop_count += 1
                ok, frame = cap.read()
                if ok and frame is not None:
                    return frame
            return None
        return frame

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._next_deadline = None
        log.info("source_closed", source=self.uid, type=self.source_type)

    # -------------------------------------------------------------- queries
    def position_seconds(self) -> float | None:
        if self._cap is None:
            return None
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        return (ms / 1000.0) if ms and ms > 0 else None

    def reset(self) -> bool:
        if self._cap is None:
            return False
        return bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0))

    @property
    def loop_count(self) -> int:
        return self._loop_count
