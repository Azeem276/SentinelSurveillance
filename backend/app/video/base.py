"""The VideoSource abstraction.

Nothing above this layer may know whether frames come from an MP4 file, an
RTSP camera or a webcam. Implementations only have to provide ``open``,
``read``, ``release`` and expose :class:`SourceCapabilities`.
"""
from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass(slots=True)
class Frame:
    """A single decoded frame plus the metadata the pipeline needs."""

    image: np.ndarray
    frame_number: int
    timestamp: datetime
    source_uid: str
    # Position in the underlying media, seconds. None for live sources.
    position_seconds: float | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(slots=True)
class SourceCapabilities:
    """What the concrete source can tell us about itself."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int | None = None      # None for live/unbounded sources
    is_live: bool = False
    seekable: bool = False
    codec: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        if self.frame_count and self.fps > 0:
            return self.frame_count / self.fps
        return None


class SourceUnavailableError(RuntimeError):
    """The source could not be opened or has stopped delivering frames."""


class VideoSourceAdapter(abc.ABC):
    """Base class for every concrete video source.

    Thread-safety: ``read`` is called from a single capture thread per source,
    but ``release``/``is_open`` may be called from the control thread, so the
    underlying handle is guarded by a lock.
    """

    source_type: str = "GENERIC"

    def __init__(self, uid: str, name: str | None = None) -> None:
        self.uid = uid
        self.name = name or uid
        self._lock = threading.RLock()
        self._opened = False
        self._frame_number = 0
        self._capabilities = SourceCapabilities()
        self._last_error: str | None = None

    # ------------------------------------------------------------- lifecycle
    @abc.abstractmethod
    def _open(self) -> SourceCapabilities:
        """Open the underlying handle and return its capabilities."""

    @abc.abstractmethod
    def _read(self) -> np.ndarray | None:
        """Return the next BGR frame or None when exhausted/unavailable."""

    @abc.abstractmethod
    def _release(self) -> None:
        """Close the underlying handle."""

    def open(self) -> SourceCapabilities:
        with self._lock:
            if self._opened:
                return self._capabilities
            try:
                self._capabilities = self._open()
            except Exception as exc:
                self._last_error = str(exc)
                raise SourceUnavailableError(f"{self.uid}: {exc}") from exc
            self._opened = True
            self._frame_number = 0
            self._last_error = None
            return self._capabilities

    def read(self) -> Frame | None:
        """Return the next frame, or None if the source is exhausted."""
        with self._lock:
            if not self._opened:
                raise SourceUnavailableError(f"{self.uid}: source is not open")
            try:
                image = self._read()
            except Exception as exc:
                self._last_error = str(exc)
                raise SourceUnavailableError(f"{self.uid}: {exc}") from exc
            if image is None:
                return None
            self._frame_number += 1
            return Frame(
                image=image,
                frame_number=self._frame_number,
                timestamp=datetime.now(timezone.utc),
                source_uid=self.uid,
                position_seconds=self.position_seconds(),
            )

    def release(self) -> None:
        with self._lock:
            if not self._opened:
                return
            try:
                self._release()
            finally:
                self._opened = False

    # -------------------------------------------------------------- queries
    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def capabilities(self) -> SourceCapabilities:
        return self._capabilities

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def frame_number(self) -> int:
        return self._frame_number

    def position_seconds(self) -> float | None:
        return None

    def reset(self) -> bool:
        """Rewind a seekable source. Returns False if unsupported."""
        return False

    def __enter__(self) -> "VideoSourceAdapter":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.uid} open={self._opened}>"
