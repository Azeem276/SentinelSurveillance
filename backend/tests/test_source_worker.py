"""SourceWorker thread lifecycle.

Regression: the worker once stored its stop flag as ``self._stop``, which
shadows ``threading.Thread._stop()``. ``Thread.join()`` calls that method once
the thread has finished, so every *successful* stop raised
``TypeError: 'Event' object is not callable`` and ``/api/surveillance/stop``
returned 500 even though the sources had stopped.
"""
from __future__ import annotations

import threading

from app.services.source_worker import SourceWorker


def _worker() -> SourceWorker:
    return SourceWorker(
        source_id=1,
        source_uid="camera_01",
        source_name="Camera 01",
        intelligence_enabled=False,
        global_intelligence=False,
        recording_enabled=False,
    )


class TestSourceWorkerThreadContract:
    def test_worker_does_not_shadow_thread_internals(self):
        worker = _worker()
        # Thread._stop must remain the base-class method, not our Event.
        assert callable(worker._stop)
        assert not isinstance(worker._stop, threading.Event)

    def test_stop_flag_is_observable(self):
        worker = _worker()
        assert worker.stopping is False
        worker.stop()
        assert worker.stopping is True

    def test_join_after_run_completes_does_not_raise(self):
        """A finished thread must be joinable; this is the path the API takes."""
        worker = _worker()
        # Run the thread body as a no-op so join() reaches Thread._stop().
        worker.run = lambda: None  # type: ignore[method-assign]
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
