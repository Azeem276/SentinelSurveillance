"""Health, diagnostics, model inventory and runtime settings."""
from __future__ import annotations

import os
import platform
import time

from fastapi import APIRouter

from app.api.deps import DbSession, Manager
from app.core.config import get_settings
from app.core.hardware import get_hardware
from app.db.session import check_connection
from app.events.bus import get_bus
from app.intelligence.face.index import get_index
from app.intelligence.model_registry import registry_status
from app.repositories.event_repository import SettingsRepository
from app.schemas.common import OperationResult
from app.schemas.intelligence import DiagnosticsResponse, ModelInfo

router = APIRouter(prefix="/api/system", tags=["system"])

_STARTED_AT = time.time()


@router.get("/health")
def health():
    connected, detail = check_connection()
    return {
        "status": "ok" if connected else "degraded",
        "database": {"connected": connected, "detail": detail if not connected else "ok"},
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "version": "1.0.0",
    }


@router.get("/models", response_model=list[ModelInfo])
def models():
    """Model inventory, including licences and whether each is installed."""
    return [ModelInfo(**entry) for entry in registry_status()]


@router.get("/config")
def runtime_config():
    """Non-secret configuration the UI needs. Never exposes credentials."""
    s = get_settings()
    return {
        "face_recognition_threshold": s.face_recognition_threshold,
        "default_proximity_a": s.default_proximity_a,
        "default_proximity_b": s.default_proximity_b,
        "temporary_familiar_default_days": s.temporary_familiar_default_days,
        "detection_classes": s.detection_class_list,
        "target_fps": s.target_fps,
        "detection_interval": s.detection_interval,
        "device": get_hardware().device,
        "recording_enabled": s.recording_enabled,
        "temporary_familiar_alert": s.temporary_familiar_alert,
        "alarm_unrecognizable_policy": s.alarm_unrecognizable_policy,
        "environment": s.environment,
    }


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(session: DbSession, manager: Manager):
    """Measured performance, not claimed: real per-source latency and FPS."""
    hardware = get_hardware()
    runtime = manager.all_runtime_status()

    sources = []
    for data in runtime.values():
        stats = data.get("stats", {}) or {}
        processed = stats.get("frames_processed", 0) or 0
        inference_ms = stats.get("inference_latency_ms", 0.0) or 0.0
        sources.append(
            {
                "source_id": data.get("source_id"),
                "uid": data.get("uid"),
                "name": data.get("name"),
                "status": data.get("status"),
                "recording": data.get("recording"),
                "intelligence": data.get("intelligence"),
                "frames_read": data.get("frames_read"),
                "frames_processed": processed,
                "dropped_frames": stats.get("dropped_frames", 0),
                "active_tracks": data.get("active_tracks", 0),
                "processing_fps": (
                    round(1000.0 / inference_ms, 2) if inference_ms > 0 else None
                ),
                "inference_latency_ms": inference_ms,
                "detection_latency_ms": stats.get("detection_latency_ms", 0.0),
                "face_latency_ms": stats.get("face_latency_ms", 0.0),
                "detector_runs": stats.get("detector_runs", 0),
                "recognitions": stats.get("recognitions", 0),
                "errors": stats.get("errors", 0),
                "capabilities": data.get("capabilities"),
            }
        )

    process: dict = {"pid": os.getpid(), "python": platform.python_version()}
    try:
        import psutil

        proc = psutil.Process()
        with proc.oneshot():
            process["cpu_percent"] = proc.cpu_percent(interval=0.05)
            process["memory_mb"] = round(proc.memory_info().rss / 1e6, 1)
            process["threads"] = proc.num_threads()
        process["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        process["system_memory_percent"] = psutil.virtual_memory().percent
    except Exception:  # pragma: no cover - psutil optional at runtime
        process["note"] = "psutil unavailable; process metrics omitted"

    gpu: dict = {"available": hardware.cuda_available}
    if hardware.cuda_available:
        try:
            import torch

            gpu["name"] = hardware.cuda_device_name
            gpu["memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            gpu["memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1e6, 1)
        except Exception:  # pragma: no cover
            pass

    return DiagnosticsResponse(
        device=hardware.device,
        hardware={**hardware.as_dict(), "gpu": gpu},
        system=manager.system_state(),
        sources=sources,
        recognition_index=get_index().stats(),
        models=[ModelInfo(**m) for m in registry_status()],
        process=process,
        websocket_subscribers=get_bus().subscriber_count,
    )


@router.get("/settings")
def get_system_settings(session: DbSession):
    return SettingsRepository(session).all_values()


@router.put("/settings/{key}", response_model=OperationResult)
def set_system_setting(key: str, value: dict, session: DbSession):
    SettingsRepository(session).set_value(key, value.get("value"))
    session.commit()
    return OperationResult(data={"key": key, "value": value.get("value")})
