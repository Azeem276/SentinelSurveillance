"""Structured logging.

Every notable lifecycle transition is logged with a stable ``event`` key so the
logs can be grepped/shipped without parsing free-form English.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.core.config import get_settings

# Stable structured-log event names (see README "Observability").
SOURCE_STARTED = "SOURCE_STARTED"
SOURCE_STOPPED = "SOURCE_STOPPED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
RECORDING_STARTED = "RECORDING_STARTED"
RECORDING_STOPPED = "RECORDING_STOPPED"
RECORDING_INTERRUPTED = "RECORDING_INTERRUPTED"
INTELLIGENCE_STARTED = "INTELLIGENCE_STARTED"
INTELLIGENCE_STOPPED = "INTELLIGENCE_STOPPED"
MODEL_LOADED = "MODEL_LOADED"
MODEL_MISSING = "MODEL_MISSING"
DETECTION_ERROR = "DETECTION_ERROR"
FACE_RECOGNITION_ERROR = "FACE_RECOGNITION_ERROR"
ALARM_STARTED = "ALARM_STARTED"
ALARM_STOPPED = "ALARM_STOPPED"
IDENTITY_UPDATED = "IDENTITY_UPDATED"
TEMPORARY_IDENTITY_EXPIRED = "TEMPORARY_IDENTITY_EXPIRED"

_configured = False


def configure_logging(level: str | None = None, as_json: bool | None = None) -> None:
    global _configured
    settings = get_settings()
    lvl = (level or settings.log_level).upper()
    use_json = settings.log_json if as_json is None else as_json

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, lvl, logging.INFO),
        force=True,
    )
    # OpenCV/ultralytics are chatty; keep their noise out of the operator log.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, lvl, logging.INFO)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str = "sentinel") -> Any:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
