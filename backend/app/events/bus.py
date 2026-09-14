"""In-process pub/sub bridging worker threads to async WebSocket clients.

The intelligence pipeline runs on plain threads; WebSocket clients live on the
asyncio loop. ``publish_threadsafe`` is the crossing point. Slow or dead
subscribers are dropped rather than allowed to stall a capture thread.
"""
from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger

log = get_logger(__name__)

MAX_QUEUE = 256
RECENT_BUFFER = 200


@dataclass(slots=True)
class BusMessage:
    type: str
    payload: dict[str, Any]
    source_uid: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source_id": self.source_uid,
            "timestamp": self.timestamp.isoformat(),
            **self.payload,
        }


class Subscriber:
    """One WebSocket connection's bounded mailbox."""

    def __init__(self, topics: set[str] | None = None, sources: set[str] | None = None) -> None:
        self.queue: asyncio.Queue[BusMessage] = asyncio.Queue(maxsize=MAX_QUEUE)
        self.topics = topics
        self.sources = sources
        self.dropped = 0

    def accepts(self, message: BusMessage) -> bool:
        if self.topics is not None and message.type not in self.topics:
            return False
        if self.sources is not None and message.source_uid not in self.sources:
            return False
        return True

    def offer(self, message: BusMessage) -> None:
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Never block the producer: drop the oldest, keep the newest.
            self.dropped += 1
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(message)
            except Exception:
                pass


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[Subscriber] = set()
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recent: deque[BusMessage] = deque(maxlen=RECENT_BUFFER)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once during app startup."""
        self._loop = loop

    # ---------------------------------------------------------- subscribe
    def subscribe(
        self, *, topics: Iterable[str] | None = None, sources: Iterable[str] | None = None
    ) -> Subscriber:
        sub = Subscriber(
            topics=set(topics) if topics else None,
            sources=set(sources) if sources else None,
        )
        with self._lock:
            self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(sub)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ------------------------------------------------------------ publish
    def _deliver(self, message: BusMessage) -> None:
        with self._lock:
            targets = [s for s in self._subscribers if s.accepts(message)]
        for sub in targets:
            sub.offer(message)

    def publish(self, message: BusMessage) -> None:
        """Publish from the asyncio loop thread."""
        self._recent.append(message)
        self._deliver(message)

    def publish_threadsafe(self, message: BusMessage) -> None:
        """Publish from a worker thread."""
        self._recent.append(message)
        loop = self._loop
        if loop is None or loop.is_closed():
            # No loop yet (startup/tests): deliver synchronously.
            self._deliver(message)
            return
        try:
            loop.call_soon_threadsafe(self._deliver, message)
        except RuntimeError:  # pragma: no cover - loop shutting down
            self._deliver(message)

    def emit(self, type_: str, source_uid: str | None = None, **payload: Any) -> None:
        self.publish_threadsafe(
            BusMessage(type=type_, payload=payload, source_uid=source_uid)
        )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [m.to_dict() for m in list(self._recent)[-limit:]]

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()
        self._recent.clear()


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus
