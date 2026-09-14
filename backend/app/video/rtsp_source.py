"""RTSPCameraSource - the future CCTV path, implemented against the same ABC.

It is intentionally minimal but real: it opens an RTSP/HTTP stream with
OpenCV/FFmpeg, keeps only the newest frame (live sources must not build a
backlog), and reconnects with backoff. Nothing in the intelligence, recording
or event layers needs to change to use it - register a source with
``type=RTSP`` and ``uri=rtsp://...``.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from app.core.logging import get_logger
from app.video.base import SourceCapabilities, SourceUnavailableError, VideoSourceAdapter

log = get_logger(__name__)


class RTSPCameraSource(VideoSourceAdapter):
    source_type = "RTSP"

    def __init__(
        self,
        uid: str,
        uri: str,
        *,
        name: str | None = None,
        transport: str = "tcp",
        open_timeout_ms: int = 8000,
        read_timeout_ms: int = 8000,
        max_reconnect_attempts: int = 5,
    ) -> None:
        super().__init__(uid, name)
        self.uri = uri
        self.transport = transport
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.max_reconnect_attempts = max_reconnect_attempts
        self._cap: cv2.VideoCapture | None = None
        self._reconnects = 0

    def _apply_ffmpeg_options(self) -> None:
        opts = [
            f"rtsp_transport;{self.transport}",
            f"stimeout;{self.open_timeout_ms * 1000}",
            "reorder_queue_size;0",
        ]
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)

    def _connect(self) -> cv2.VideoCapture:
        self._apply_ffmpeg_options()
        cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            raise SourceUnavailableError(f"cannot connect to stream: {self.uid}")
        # A live source must never queue frames: always serve the newest one.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - backend dependent
            pass
        return cap

    def _open(self) -> SourceCapabilities:
        cap = self._connect()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0 < fps < 240):
            fps = 25.0
        self._cap = cap
        self._reconnects = 0
        caps = SourceCapabilities(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=fps,
            frame_count=None,
            is_live=True,
            seekable=False,
            extra={"transport": self.transport},
        )
        log.info("source_opened", source=self.uid, type=self.source_type, fps=fps)
        return caps

    def _read(self) -> np.ndarray | None:
        if self._cap is None:
            raise SourceUnavailableError("source is not open")
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._reconnects = 0
            return frame
        return self._reconnect_and_read()

    def _reconnect_and_read(self) -> np.ndarray | None:
        while self._reconnects < self.max_reconnect_attempts:
            self._reconnects += 1
            backoff = min(2.0 ** self._reconnects, 15.0)
            log.warning("source_reconnecting", source=self.uid,
                        attempt=self._reconnects, backoff=backoff)
            time.sleep(backoff)
            try:
                if self._cap is not None:
                    self._cap.release()
                self._cap = self._connect()
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    return frame
            except SourceUnavailableError:
                continue
        return None

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("source_closed", source=self.uid, type=self.source_type)
