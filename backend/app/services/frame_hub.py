"""Latest-frame hub backing the MJPEG preview streams.

Video pixels never go through the WebSocket - that channel carries only
structured overlay/event data. Each source worker publishes its most recent
frame here as JPEG bytes, and HTTP stream endpoints serve them. Only the
newest frame is kept, so a slow viewer can never back-pressure a capture
thread.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FrameSlot:
    jpeg: bytes
    frame_number: int
    published_at: float
    width: int
    height: int


class FrameHub:
    def __init__(self, *, jpeg_quality: int = 72) -> None:
        self.jpeg_quality = jpeg_quality
        self._frames: dict[str, FrameSlot] = {}
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def _event_for(self, source_uid: str) -> threading.Event:
        with self._lock:
            event = self._events.get(source_uid)
            if event is None:
                event = threading.Event()
                self._events[source_uid] = event
            return event

    def publish(self, source_uid: str, image: np.ndarray, frame_number: int = 0) -> None:
        if image is None or image.size == 0:
            return
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        slot = FrameSlot(
            jpeg=buffer.tobytes(),
            frame_number=frame_number,
            published_at=time.time(),
            width=int(image.shape[1]),
            height=int(image.shape[0]),
        )
        with self._lock:
            self._frames[source_uid] = slot
        event = self._event_for(source_uid)
        event.set()
        event.clear()

    def latest(self, source_uid: str) -> FrameSlot | None:
        with self._lock:
            return self._frames.get(source_uid)

    def wait_for_frame(self, source_uid: str, *, timeout: float = 2.0) -> FrameSlot | None:
        """Block until a newer frame arrives (or the timeout expires)."""
        self._event_for(source_uid).wait(timeout)
        return self.latest(source_uid)

    def clear(self, source_uid: str | None = None) -> None:
        with self._lock:
            if source_uid is None:
                self._frames.clear()
            else:
                self._frames.pop(source_uid, None)

    def active_sources(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())

    def is_fresh(self, source_uid: str, *, max_age: float = 5.0) -> bool:
        slot = self.latest(source_uid)
        return slot is not None and (time.time() - slot.published_at) <= max_age


_hub = FrameHub()


def get_frame_hub() -> FrameHub:
    return _hub
