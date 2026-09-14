"""Sentinel FastAPI application.

Startup sequence:
  1. configure logging and ensure storage directories exist
  2. detect hardware (GPU if present, CPU otherwise - never fatal)
  3. verify the database and recover from an unclean shutdown
  4. rebuild the face recognition index
  5. start the expiration scheduler
  6. restore the operator's persisted surveillance intent
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import (
    routes_archive, routes_events, routes_identities, routes_sources,
    routes_surveillance, routes_system, websocket,
)
from app.core.config import get_settings
from app.core.exceptions import SentinelError
from app.core.hardware import get_hardware
from app.core.logging import configure_logging, get_logger
from app.db.session import check_connection, dispose_engine
from app.events.bus import get_bus
from app.services.scheduler import start_scheduler, stop_scheduler

log = get_logger("sentinel.main")


def recover_after_restart() -> dict:
    """Close records left open by a crash or an unclean shutdown."""
    from app.db.session import session_scope
    from app.repositories.detection_repository import MotionRepository, TrackRepository
    from app.repositories.event_repository import EventRepository
    from app.repositories.recording_repository import RecordingRepository
    from app.repositories.source_repository import SourceRepository

    with session_scope() as session:
        result = {
            "recordings_interrupted": RecordingRepository(session).close_orphans(),
            "tracks_closed": TrackRepository(session).close_orphans(),
            "motion_events_closed": MotionRepository(session).close_orphans(),
            "events_closed": EventRepository(session).close_orphans(),
        }
        SourceRepository(session).reset_all_statuses()
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    settings.ensure_directories()

    hardware = get_hardware()
    log.info(
        "startup",
        app=settings.app_name,
        environment=settings.environment,
        device=hardware.device,
        cuda=hardware.cuda_available,
    )
    if hardware.reason:
        log.info("device_selection", reason=hardware.reason)

    get_bus().bind_loop(asyncio.get_running_loop())

    connected, detail = check_connection()
    if connected:
        try:
            recovered = recover_after_restart()
            if any(recovered.values()):
                log.info("restart_recovery", **recovered)
        except Exception as exc:
            log.error("restart_recovery_failed", error=str(exc))

        try:
            from app.alerts.engine import get_alert_engine
            from app.db.session import session_scope
            from app.services.identity_service import sync_recognition_index

            stats = sync_recognition_index()
            log.info("recognition_index_ready", **stats)
            with session_scope() as session:
                get_alert_engine().hydrate(session)
        except Exception as exc:
            log.error("index_bootstrap_failed", error=str(exc))

        try:
            start_scheduler()
        except Exception as exc:
            log.error("scheduler_start_failed", error=str(exc))

        try:
            from app.services.surveillance import get_manager

            get_manager().hydrate()
        except Exception as exc:
            log.error("surveillance_hydrate_failed", error=str(exc))
    else:
        # The API still starts so the operator can see *why* it is degraded.
        log.error(
            "database_unavailable",
            detail=detail,
            hint="Set DATABASE_URL in .env and apply backend/migrations/schema.sql",
        )

    try:
        yield
    finally:
        log.info("shutdown_started")
        try:
            from app.services.surveillance import get_manager

            get_manager().shutdown()
        except Exception as exc:
            log.warning("surveillance_shutdown_failed", error=str(exc))
        stop_scheduler()
        dispose_engine()
        log.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Sentinel",
        description=(
            "Intelligent Video Surveillance & Analysis Platform. "
            "All AI inference runs locally; no video, face or embedding data "
            "is sent to any external service."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------- error handling
    @app.exception_handler(SentinelError)
    async def sentinel_error_handler(_request: Request, exc: SentinelError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "validation_error",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_error", path=str(request.url.path), error=str(exc), exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "details": {},
            },
        )

    # ------------------------------------------------------------ routers
    app.include_router(routes_surveillance.router)
    app.include_router(routes_sources.router)
    app.include_router(routes_identities.router)
    app.include_router(routes_events.router)
    app.include_router(routes_archive.router)
    app.include_router(routes_system.router)
    app.include_router(websocket.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "name": "Sentinel",
            "version": "1.0.0",
            "docs": "/api/docs",
            "realtime": "/ws",
            "local_only": True,
        }

    return app


app = create_app()
