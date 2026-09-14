"""Video source request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.enums import SourceStatus, SourceType
from app.schemas.common import ORMModel


class ProximitySettings(BaseModel):
    """Proximity A must always be farther from the camera than Proximity B."""

    proximity_a: float = Field(..., gt=0, le=500, description="Recognition zone, metres")
    proximity_b: float = Field(..., gt=0, le=500, description="Alarm zone, metres")

    @model_validator(mode="after")
    def _check_order(self) -> "ProximitySettings":
        if self.proximity_a <= self.proximity_b:
            raise ValueError(
                "Proximity A (recognition zone) must be greater than "
                "Proximity B (alarm zone)"
            )
        return self


class AlertPolicySchema(BaseModel):
    temporary_familiar: str = Field("beep", pattern="^(none|beep|continuous)$")
    unrecognizable_policy: str = Field("ignore", pattern="^(ignore|alarm)$")
    permanent_familiar: str = Field("none", pattern="^(none|beep)$")
    beep_on_recognition: bool = True


class CalibrationSchema(BaseModel):
    vertical_fov_deg: float = Field(55.0, gt=1, lt=179)
    reference_height_m: float = Field(1.7, gt=0.2, lt=3.0)
    distance_scale: float = Field(1.0, gt=0.05, lt=20.0)
    visible_height_fraction: float = Field(1.0, gt=0.05, le=1.0)


class SourceCreate(BaseModel):
    uid: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")
    name: str = Field(..., min_length=1, max_length=128)
    type: SourceType = SourceType.FILE
    uri: str = Field(..., min_length=1, max_length=1024)
    location: str | None = Field(None, max_length=255)
    description: str | None = None
    enabled: bool = True
    surveillance_enabled: bool = True
    intelligence_enabled: bool = True
    recognition_enabled: bool = True
    recording_enabled: bool = True
    loop_playback: bool = True
    proximity_a: float = Field(10.0, gt=0, le=500)
    proximity_b: float = Field(3.0, gt=0, le=500)
    recognition_threshold: float | None = Field(None, gt=0, le=1)
    detection_confidence: float | None = Field(None, gt=0, le=1)
    temporary_retention_days: int | None = Field(None, gt=0, le=3650)
    alert_policy: AlertPolicySchema | None = None
    calibration: CalibrationSchema | None = None

    @model_validator(mode="after")
    def _check_proximity(self) -> "SourceCreate":
        if self.proximity_a <= self.proximity_b:
            raise ValueError("Proximity A must be greater than Proximity B")
        return self

    @field_validator("uri")
    @classmethod
    def _check_uri(cls, value: str, info) -> str:
        return value.strip()


class SourceUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=128)
    location: str | None = Field(None, max_length=255)
    description: str | None = None
    enabled: bool | None = None
    surveillance_enabled: bool | None = None
    intelligence_enabled: bool | None = None
    recognition_enabled: bool | None = None
    recording_enabled: bool | None = None
    loop_playback: bool | None = None
    proximity_a: float | None = Field(None, gt=0, le=500)
    proximity_b: float | None = Field(None, gt=0, le=500)
    recognition_threshold: float | None = Field(None, gt=0, le=1)
    detection_confidence: float | None = Field(None, gt=0, le=1)
    temporary_retention_days: int | None = Field(None, gt=0, le=3650)
    alert_policy: AlertPolicySchema | None = None
    calibration: CalibrationSchema | None = None
    display_order: int | None = None


class SourceRuntime(BaseModel):
    running: bool = False
    status: str = SourceStatus.IDLE.value
    recording: bool = False
    recording_id: int | None = None
    intelligence: bool = False
    motion: bool = False
    active_tracks: int = 0
    frames_read: int = 0
    error: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    capabilities: dict[str, Any] | None = None


class SourceRead(ORMModel):
    id: int
    uid: str
    name: str
    type: SourceType
    uri: str
    location: str | None
    description: str | None
    enabled: bool
    surveillance_enabled: bool
    intelligence_enabled: bool
    recognition_enabled: bool
    recording_enabled: bool
    loop_playback: bool
    status: SourceStatus
    last_error: str | None
    last_seen_at: datetime | None
    proximity_a: float
    proximity_b: float
    recognition_threshold: float | None
    detection_confidence: float | None
    temporary_retention_days: int | None
    alert_policy: dict[str, Any]
    calibration: dict[str, Any]
    display_order: int
    created_at: datetime
    updated_at: datetime


class SourceDetail(SourceRead):
    runtime: SourceRuntime = Field(default_factory=SourceRuntime)
    stream_url: str | None = None


class SourceListItem(SourceRead):
    runtime: SourceRuntime = Field(default_factory=SourceRuntime)
    unread_alerts: int = 0
