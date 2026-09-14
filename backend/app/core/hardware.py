"""Hardware capability detection.

CUDA is never mandatory: the resolver degrades to CPU and records *why*.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class HardwareInfo:
    device: str = "cpu"
    requested: str = "auto"
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_device_name: str | None = None
    cuda_device_count: int = 0
    mps_available: bool = False
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probe() -> HardwareInfo:
    settings = get_settings()
    info = HardwareInfo(requested=settings.ai_device)

    try:
        import torch  # noqa: PLC0415  (optional heavy import)

        info.torch_available = True
        info.torch_version = torch.__version__
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.cuda_device_count = torch.cuda.device_count()
            info.cuda_device_name = torch.cuda.get_device_name(0)
        info.mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception as exc:  # pragma: no cover - depends on install
        info.reason = f"torch unavailable: {exc}"

    requested = settings.ai_device
    if requested == "cuda":
        if info.cuda_available:
            info.device = "cuda"
        else:
            info.device = "cpu"
            info.reason = "AI_DEVICE=cuda requested but CUDA is not available; using CPU"
    elif requested == "mps":
        info.device = "mps" if info.mps_available else "cpu"
        if info.device == "cpu":
            info.reason = "AI_DEVICE=mps requested but MPS is not available; using CPU"
    elif requested == "cpu":
        info.device = "cpu"
    else:  # auto
        if info.cuda_available:
            info.device = "cuda"
        elif info.mps_available:
            info.device = "mps"
        else:
            info.device = "cpu"
            info.reason = info.reason or "no GPU detected; using CPU"
    return info


@lru_cache(maxsize=1)
def get_hardware() -> HardwareInfo:
    info = _probe()
    log.info("hardware_detected", **info.as_dict())
    return info


def resolve_device() -> str:
    return get_hardware().device
