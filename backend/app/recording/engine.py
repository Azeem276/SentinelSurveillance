"""Recording engine.

One source produces one file per uninterrupted session. Sessions close safely
on stop, on source exhaustion, on error and on process shutdown; a resumed
recording always opens a NEW file. Nothing is ever appended to a file that
was interrupted, so a crashed session leaves a short-but-valid clip rather
than a corrupt one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logging import (
    RECORDING_INTERRUPTED, RECORDING_STARTED, RECORDING_STOPPED, get_logger,
)
from app.storage.paths import ensure_parent, recording_root, slugify

log = get_logger(__name__)


def session_filename(when: datetime) -> str:
    return f"recording_{when.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"


def source_directory(source_uid: str) -> str:
    return slugify(source_uid, fallback="source")


@dataclass(slots=True)
class RecordingResult:
    """What a finished session produced, for the repository to persist."""

    relative_path: str
    absolute_path: Path
    started_at: datetime
    ended_at: datetime
    frame_count: int
    fps: float
    width: int
    height: int
    file_size_bytes: int | None
    interrupted: bool = False
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


@dataclass
class RecordingSessionWriter:
    """A single open video file.

    Thread-safety: ``write`` is called from the owning capture thread, while
    ``close`` may come from the control thread, so both take the lock.
    """

    source_uid: str
    width: int
    height: int
    fps: float
    relative_path: str
    absolute_path: Path
    started_at: datetime
    fourcc: str = "mp4v"
    frame_count: int = 0
    error: str | None = None
    _writer: cv2.VideoWriter | None = field(default=None, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _closed: bool = False

    @property
    def is_open(self) -> bool:
        return self._writer is not None and not self._closed

    def open(self) -> None:
        ensure_parent(self.absolute_path)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        writer = cv2.VideoWriter(
            str(self.absolute_path), fourcc, float(self.fps), (self.width, self.height)
        )
        if not writer.isOpened():
            writer.release()
            raise OSError(
                f"could not open video writer for {self.absolute_path.name} "
                f"(codec {self.fourcc}); check FFmpeg/codec availability"
            )
        self._writer = writer
        log.info(
            RECORDING_STARTED,
            source=self.source_uid,
            file=self.relative_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    def write(self, frame: np.ndarray) -> bool:
        with self._lock:
            if self._writer is None or self._closed:
                return False
            try:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self._writer.write(frame)
                self.frame_count += 1
                return True
            except Exception as exc:
                # A write failure must not kill the capture thread; the
                # session is marked failed and a new one can be started.
                self.error = str(exc)
                log.error("recording_write_failed", source=self.source_uid, error=str(exc))
                self._safe_release()
                return False

    def _safe_release(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:  # pragma: no cover - defensive
                pass
            self._writer = None

    def close(self, *, interrupted: bool = False, error: str | None = None) -> RecordingResult:
        with self._lock:
            if not self._closed:
                self._safe_release()
                self._closed = True
            ended = datetime.now(timezone.utc)
            size: int | None
            try:
                size = self.absolute_path.stat().st_size
            except OSError:
                size = None

            result = RecordingResult(
                relative_path=self.relative_path,
                absolute_path=self.absolute_path,
                started_at=self.started_at,
                ended_at=ended,
                frame_count=self.frame_count,
                fps=self.fps,
                width=self.width,
                height=self.height,
                file_size_bytes=size,
                interrupted=interrupted,
                error=error or self.error,
            )
        if interrupted or result.error:
            log.warning(
                RECORDING_INTERRUPTED,
                source=self.source_uid,
                file=self.relative_path,
                frames=self.frame_count,
                error=result.error,
            )
        else:
            log.info(
                RECORDING_STOPPED,
                source=self.source_uid,
                file=self.relative_path,
                frames=self.frame_count,
                seconds=round(result.duration_seconds, 2),
            )
        return result


class RecordingEngine:
    """Creates recording sessions. Owns no database state by design.

    The caller (the source worker) persists RecordingSession rows; keeping
    the engine persistence-free makes the file lifecycle independently
    testable.
    """

    def __init__(self, *, fourcc: str | None = None, max_minutes: int | None = None) -> None:
        settings = get_settings()
        self.fourcc = fourcc or settings.recording_fourcc
        self.max_minutes = (
            settings.recording_max_minutes if max_minutes is None else max_minutes
        )

    def start_session(
        self,
        *,
        source_uid: str,
        width: int,
        height: int,
        fps: float | None = None,
        when: datetime | None = None,
    ) -> RecordingSessionWriter:
        settings = get_settings()
        started = when or datetime.now(timezone.utc)
        fps = fps if fps and fps > 0 else settings.recording_fps

        directory = source_directory(source_uid)
        filename = session_filename(started.astimezone())
        relative = f"{directory}/{filename}"
        absolute = recording_root() / directory / filename

        # A second session started inside the same second must not collide.
        counter = 1
        while absolute.exists():
            counter += 1
            stem = filename[:-4]
            relative = f"{directory}/{stem}_{counter}.mp4"
            absolute = recording_root() / directory / f"{stem}_{counter}.mp4"

        writer = RecordingSessionWriter(
            source_uid=source_uid,
            width=int(width),
            height=int(height),
            fps=float(fps),
            relative_path=relative,
            absolute_path=absolute,
            started_at=started,
            fourcc=self.fourcc,
        )
        writer.open()
        return writer

    def should_rotate(self, writer: RecordingSessionWriter, *, now: datetime | None = None
                      ) -> bool:
        """True when the open session has reached its maximum length."""
        if self.max_minutes <= 0:
            return False
        now = now or datetime.now(timezone.utc)
        started = writer.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return (now - started).total_seconds() >= self.max_minutes * 60
