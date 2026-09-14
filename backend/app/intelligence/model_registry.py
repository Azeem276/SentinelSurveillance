"""Model registry, provisioning and licence bookkeeping.

Sentinel never silently pulls large weights: every artefact is declared here
with its size, licence and origin, and ``ensure_model`` reports exactly what
it is doing. Provisioning is idempotent, so the same call works on a laptop
and on a fresh server (``AUTO_DOWNLOAD_MODELS=true`` by default).
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ModelNotAvailableError
from app.core.logging import MODEL_LOADED, MODEL_MISSING, get_logger

log = get_logger(__name__)

USER_AGENT = "Sentinel-ModelProvisioner/1.0"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    approx_mb: float
    licence: str
    origin: str
    purpose: str
    sha256: str | None = None
    # Ultralytics weights are fetched by the library itself on first use.
    managed_by_library: bool = False


OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODEL_SPECS: dict[str, ModelSpec] = {
    "object_detector": ModelSpec(
        key="object_detector",
        filename="yolo11n.pt",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        approx_mb=5.6,
        licence="AGPL-3.0 (Ultralytics)",
        origin="https://github.com/ultralytics/assets",
        purpose="Object detection (person, car, dog, ...) and ByteTrack tracking",
    ),
    "face_detector": ModelSpec(
        key="face_detector",
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        approx_mb=0.34,
        licence="MIT (OpenCV Zoo / YuNet)",
        origin="https://github.com/opencv/opencv_zoo",
        purpose="Face detection inside the Proximity A recognition zone",
    ),
    "face_embedder": ModelSpec(
        key="face_embedder",
        filename="face_recognition_sface_2021dec.onnx",
        url=f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        approx_mb=37.0,
        licence="Apache-2.0 (OpenCV Zoo / SFace)",
        origin="https://github.com/opencv/opencv_zoo",
        purpose="128-d face embeddings for local recognition",
    ),
}


def model_dir() -> Path:
    path = Path(get_settings().model_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(key: str) -> Path:
    spec = MODEL_SPECS[key]
    settings = get_settings()
    # Allow per-model overrides from configuration.
    override = {
        "object_detector": settings.object_model_name,
        "face_detector": settings.face_detector_model,
        "face_embedder": settings.face_embedder_model,
    }.get(key)
    filename = override or spec.filename
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return model_dir() / filename


def is_installed(key: str) -> bool:
    path = model_path(key)
    return path.exists() and path.stat().st_size > 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: ModelSpec, destination: Path) -> None:
    log.info(
        "model_download_started",
        model=spec.key,
        filename=destination.name,
        approx_mb=spec.approx_mb,
        licence=spec.licence,
        url=spec.url,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(spec.url, headers={"User-Agent": USER_AGENT})
    tmp_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            if getattr(response, "status", 200) >= 400:
                raise ModelNotAvailableError(
                    f"download failed for {spec.key}: HTTP {response.status}"
                )
            with tempfile.NamedTemporaryFile(
                delete=False, dir=destination.parent, suffix=".part"
            ) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copyfileobj(response, tmp, length=1 << 20)
    except urllib.error.URLError as exc:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise ModelNotAvailableError(
            f"Could not download model {spec.key!r} from {spec.url}: {exc}. "
            f"Download it manually and place it at {destination}."
        ) from exc

    assert tmp_path is not None
    if spec.sha256:
        actual = _sha256(tmp_path)
        if actual != spec.sha256:
            tmp_path.unlink(missing_ok=True)
            raise ModelNotAvailableError(
                f"checksum mismatch for {spec.key}: expected {spec.sha256}, got {actual}"
            )
    # Atomic publish so a half-written file is never treated as installed.
    tmp_path.replace(destination)
    log.info(
        "model_download_completed",
        model=spec.key,
        path=str(destination),
        size_bytes=destination.stat().st_size,
    )


def ensure_model(key: str, *, allow_download: bool | None = None) -> Path:
    """Return the local path to a model, downloading it if permitted."""
    if key not in MODEL_SPECS:
        raise ModelNotAvailableError(f"unknown model key: {key}")
    spec = MODEL_SPECS[key]
    path = model_path(key)

    if path.exists() and path.stat().st_size > 0:
        return path

    if spec.managed_by_library:
        return path

    settings = get_settings()
    permitted = settings.auto_download_models if allow_download is None else allow_download
    if not permitted:
        log.error(MODEL_MISSING, model=spec.key, expected_path=str(path))
        raise ModelNotAvailableError(
            f"Model not installed: {spec.filename}\n"
            f"Run:\n    python scripts/download_models.py\n"
            f"or set AUTO_DOWNLOAD_MODELS=true, or place the file at {path}"
        )

    _download(spec, path)
    return path


def ensure_all(*, allow_download: bool | None = None) -> dict[str, Path]:
    return {key: ensure_model(key, allow_download=allow_download) for key in MODEL_SPECS}


def registry_status() -> list[dict]:
    """Machine-readable inventory, surfaced by /api/system/models."""
    out = []
    for key, spec in MODEL_SPECS.items():
        path = model_path(key)
        installed = path.exists() and path.stat().st_size > 0
        out.append(
            {
                "key": key,
                "filename": path.name,
                "installed": installed,
                "path": str(path) if installed else None,
                "size_bytes": path.stat().st_size if installed else None,
                "approx_mb": spec.approx_mb,
                "licence": spec.licence,
                "origin": spec.origin,
                "purpose": spec.purpose,
            }
        )
    return out


def log_model_loaded(key: str, device: str, path: Path) -> None:
    log.info(MODEL_LOADED, model=key, device=device, path=str(path))
