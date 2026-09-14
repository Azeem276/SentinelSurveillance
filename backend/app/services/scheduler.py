"""Backend-enforced background jobs.

Temporary-identity expiry must never depend on a browser being open, so it
runs here on a server-side scheduler with a persistent database check.
"""
from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.identity_service import run_expiration_scan

log = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def _expiration_job() -> None:
    try:
        expired = run_expiration_scan()
        if expired:
            log.info("expiration_sweep", expired=len(expired))
    except Exception as exc:  # pragma: no cover - the job must never die
        log.error("expiration_sweep_failed", error=str(exc))


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    settings = get_settings()
    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    scheduler.add_job(
        _expiration_job,
        trigger=IntervalTrigger(seconds=max(10, settings.expiration_scan_seconds)),
        id="temporary_identity_expiration",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    log.info("scheduler_started", interval_seconds=settings.expiration_scan_seconds)
    return scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:  # pragma: no cover
            pass
        _scheduler = None
        log.info("scheduler_stopped")
