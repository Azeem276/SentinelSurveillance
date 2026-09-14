"""Central application configuration.

All tunable behaviour is funnelled through :class:`Settings` so that no module
reads ``os.environ`` directly.  Values come from (in order of precedence):
environment variables, the ``.env`` file at the repository root, then defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../Sentinel  (this file lives at backend/app/core/config.py)
REPO_ROOT = Path(__file__).resolve().parents[3]

DeviceOption = Literal["auto", "cuda", "cpu", "mps"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Sentinel"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False

    # ------------------------------------------------------------- security
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    secret_key: str = "change-me-in-production"

    # ------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # -------------------------------------------------------------- storage
    storage_path: Path = REPO_ROOT / "storage"
    recording_path: Path | None = None
    face_storage_path: Path | None = None
    snapshot_path: Path | None = None
    event_asset_path: Path | None = None
    video_storage_path: Path = REPO_ROOT / "data" / "sample_videos"
    face_dataset_path: Path = REPO_ROOT / "data" / "familiar_faces"
    model_path: Path = REPO_ROOT / "models"

    # --------------------------------------------------------------- models
    ai_device: DeviceOption = "auto"
    auto_download_models: bool = True
    object_detector: str = "yolo"
    object_model_name: str = "yolo11n.pt"
    face_detector_model: str = "face_detection_yunet_2023mar.onnx"
    face_embedder_model: str = "face_recognition_sface_2021dec.onnx"
    detection_confidence: float = 0.35
    detection_iou: float = 0.5
    detection_classes: str = (
        "person,bicycle,car,motorcycle,bus,truck,cat,dog,horse,sheep,cow,bear,bird"
    )

    # ------------------------------------------------------------ inference
    detection_interval: int = 3          # run the detector every N frames
    target_fps: float = 12.0             # processing cadence per source
    max_inference_workers: int = 2
    frame_max_width: int = 960           # downscale before inference
    recognition_cooldown_frames: int = 45  # re-verify an identified track

    # ------------------------------------------------------------- tracking
    track_max_age: int = 30
    track_min_hits: int = 3
    track_iou_threshold: float = 0.3

    # --------------------------------------------------------------- motion
    motion_min_area_ratio: float = 0.0015
    motion_start_frames: int = 3
    motion_end_frames: int = 25

    # ----------------------------------------------------------------- face
    face_recognition_threshold: float = 0.55
    face_min_pixels: int = 48
    face_min_blur: float = 18.0
    face_min_brightness: float = 35.0
    face_max_brightness: float = 225.0
    face_min_confidence: float = 0.7
    face_max_yaw_ratio: float = 0.38
    face_votes_to_switch_identity: int = 3

    # ------------------------------------------------------------ proximity
    default_proximity_a: float = 10.0
    default_proximity_b: float = 3.0
    reference_person_height_m: float = 1.7
    default_vertical_fov_deg: float = 55.0

    # ------------------------------------------------------------- identity
    temporary_familiar_default_days: int = 7
    expiration_scan_seconds: int = 60

    # --------------------------------------------------------------- alerts
    alarm_unrecognizable_policy: Literal["ignore", "alarm"] = "ignore"
    temporary_familiar_alert: Literal["none", "beep", "continuous"] = "beep"
    alert_repeat_seconds: int = 30

    # ------------------------------------------------------------ recording
    recording_fps: float = 15.0
    recording_fourcc: str = "mp4v"
    recording_max_minutes: int = 30
    recording_enabled: bool = True

    # ------------------------------------------------------------ retention
    event_retention_days: int = 90

    # -------------------------------------------------------------- helpers
    @field_validator("cors_origins")
    @classmethod
    def _strip_origins(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        # Relative paths in .env are interpreted against the repository root,
        # not the process working directory, so the backend behaves the same
        # whether it is started from the repo root, backend/, or a service
        # manager with an arbitrary cwd.
        for field in (
            "storage_path", "recording_path", "face_storage_path", "snapshot_path",
            "event_asset_path", "video_storage_path", "face_dataset_path", "model_path",
        ):
            value = getattr(self, field, None)
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            object.__setattr__(self, field, path)

        base = self.storage_path
        object.__setattr__(self, "recording_path", self.recording_path or base / "recordings")
        object.__setattr__(self, "face_storage_path", self.face_storage_path or base / "faces")
        object.__setattr__(self, "snapshot_path", self.snapshot_path or base / "snapshots")
        object.__setattr__(self, "event_asset_path", self.event_asset_path or base / "events")
        if self.default_proximity_a <= self.default_proximity_b:
            raise ValueError(
                "DEFAULT_PROXIMITY_A must be greater than DEFAULT_PROXIMITY_B "
                "(A is the farther recognition zone, B is the closer alarm zone)"
            )
        if self.default_proximity_b <= 0:
            raise ValueError("DEFAULT_PROXIMITY_B must be > 0")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def detection_class_list(self) -> list[str]:
        return [c.strip() for c in self.detection_classes.split(",") if c.strip()]

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy URL guaranteed to use an installed driver."""
        return self.database_url

    def ensure_directories(self) -> None:
        for p in (
            self.storage_path,
            self.recording_path,
            self.face_storage_path,
            self.snapshot_path,
            self.event_asset_path,
            self.video_storage_path,
            self.face_dataset_path,
            self.model_path,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)
        for sub in ("permanent", "temporary", "unfamiliar"):
            (Path(self.face_storage_path) / sub).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache (used by tests that patch environment variables)."""
    get_settings.cache_clear()
    return get_settings()


settings = get_settings()
