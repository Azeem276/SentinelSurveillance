"""Identity, face, event, alert, recording and analysis schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.enums import (
    AlertState, AlertType, EventSeverity, EventType, FaceReviewStatus, IdentityCategory,
    IdentityStatus, ProximityZone, RecognitionState, RecordingStatus, TrackStatus,
)
from app.schemas.common import ORMModel

RETENTION_PRESETS = {"24h": 1, "3d": 3, "7d": 7, "30d": 30}


# --------------------------------------------------------------- identities
class IdentityCreate(BaseModel):
    display_name: str | None = Field(None, max_length=128)
    category: IdentityCategory
    retention_days: int | None = Field(None, gt=0, le=3650)
    notes: str | None = None
    source_id: int | None = None
    first_detected_at: datetime | None = None


class IdentityUpdate(BaseModel):
    display_name: str | None = Field(None, max_length=128)
    category: IdentityCategory | None = None
    retention_days: int | None = Field(None, gt=0, le=3650)
    notes: str | None = None


class IdentityRead(ORMModel):
    id: int
    generated_identifier: str
    display_name: str | None
    category: IdentityCategory
    status: IdentityStatus
    first_detected_at: datetime
    last_seen_at: datetime | None
    expires_at: datetime | None
    retention_days: int | None
    notes: str | None
    thumbnail_path: str | None
    source_id: int | None
    created_at: datetime

    @property
    def label(self) -> str:
        return self.display_name or self.generated_identifier


class IdentityDetail(IdentityRead):
    embedding_count: int = 0
    track_count: int = 0
    label: str = ""
    thumbnail_url: str | None = None


# --------------------------------------------------------------------- faces
class FaceRead(ORMModel):
    id: int
    source_id: int
    track_id: int | None
    identity_id: int | None
    detected_at: datetime
    frame_number: int
    image_path: str | None
    detection_confidence: float
    quality_score: float
    quality_ok: bool
    quality_reason: str | None
    face_pixels: int | None
    recognition_state: RecognitionState
    match_score: float | None
    review_status: FaceReviewStatus


class UnfamiliarFace(FaceRead):
    source_uid: str | None = None
    source_name: str | None = None
    image_url: str | None = None
    suggested_identifier: str | None = None
    # How many extra angles of this person are waiting to be enrolled with
    # them. A high number means classifying this face teaches the system a
    # lot, not just one more picture.
    profile_samples: int = 0


class FaceClassifyRequest(BaseModel):
    """Classify one unfamiliar face.

    A blank ``display_name`` is valid and intentional: the backend then
    generates ``Unknown_YYYYMMDD_HHMMSS`` from the original detection time.
    """

    category: IdentityCategory
    display_name: str | None = Field(None, max_length=128)
    retention_days: int | None = Field(None, gt=0, le=3650)
    retention_preset: str | None = Field(None, pattern="^(24h|3d|7d|30d|custom)$")
    identity_id: int | None = None

    @model_validator(mode="after")
    def _resolve_retention(self) -> "FaceClassifyRequest":
        if self.retention_preset and self.retention_preset != "custom":
            self.retention_days = RETENTION_PRESETS[self.retention_preset]
        if self.category is IdentityCategory.TEMPORARY and self.retention_days is None:
            self.retention_days = 7
        return self


class BulkClassifyRequest(FaceClassifyRequest):
    face_ids: list[int] = Field(..., min_length=1, max_length=200)
    # Off by default: distinct people must not silently merge into one identity.
    merge_into_one_identity: bool = False


class FaceMergeRequest(BaseModel):
    """Attach a reviewed face to an identity that already exists."""

    identity_id: int


class IdentityMergeRequest(BaseModel):
    """Fold this identity into another one, which survives."""

    into_identity_id: int


class MergeCandidate(BaseModel):
    """A "this might already be somebody you know" suggestion."""

    identity_id: int
    label: str
    display_name: str | None = None
    generated_identifier: str
    category: str
    thumbnail_path: str | None = None
    thumbnail_url: str | None = None
    mean_score: float
    best_score: float
    samples_compared: int
    embedding_count: int


class ClassificationResponse(BaseModel):
    identity_id: int
    identifier: str
    display_name: str | None
    category: str
    expires_at: str | None
    embeddings_added: int
    created: bool


# -------------------------------------------------------------------- tracks
class TrackRead(ORMModel):
    id: int
    source_id: int
    track_key: int
    object_class: str
    status: TrackStatus
    first_seen_at: datetime
    last_seen_at: datetime
    duration_seconds: float
    detection_count: int
    max_confidence: float
    identity_id: int | None
    recognition_state: RecognitionState
    recognition_confidence: float | None
    proximity_zone: ProximityZone
    min_distance_m: float | None
    entered_zone_a: bool
    entered_zone_b: bool
    recording_id: int | None


class DetectionRead(ORMModel):
    id: int
    source_id: int
    track_id: int | None
    recording_id: int | None
    timestamp: datetime
    frame_number: int
    object_class: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    distance_m: float | None
    proximity_zone: ProximityZone | None


class MotionEventRead(ORMModel):
    id: int
    source_id: int
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: float | None
    peak_area_ratio: float
    recording_id: int | None


# -------------------------------------------------------------------- events
class EventRead(ORMModel):
    id: int
    source_id: int
    event_type: EventType
    severity: EventSeverity
    track_id: int | None
    identity_id: int | None
    face_id: int | None
    recording_id: int | None
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: float | None
    is_open: bool
    occurrence_count: int
    confidence: float | None
    label: str | None
    message: str | None
    event_metadata: dict[str, Any] = Field(default_factory=dict)


class TimelineEntry(BaseModel):
    id: int
    timestamp: datetime
    event_type: str
    severity: str
    label: str | None
    message: str | None
    duration_seconds: float | None
    track_id: int | None
    identity_id: int | None
    identity_label: str | None
    recording_id: int | None
    object_class: str | None = None
    is_open: bool = False


# -------------------------------------------------------------------- alerts
class AlertRead(ORMModel):
    id: int
    source_id: int
    alert_type: AlertType
    state: AlertState
    reason: str
    started_at: datetime
    stopped_at: datetime | None
    stopped_by: str | None
    track_id: int | None
    identity_id: int | None
    security_event_id: int | None
    acknowledged: bool


# ---------------------------------------------------------------- recordings
class RecordingRead(ORMModel):
    id: int
    source_id: int
    file_path: str
    status: RecordingStatus
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: float | None
    frame_count: int
    fps: float | None
    width: int | None
    height: int | None
    file_size_bytes: int | None
    error: str | None


class RecordingDetail(RecordingRead):
    source_uid: str | None = None
    source_name: str | None = None
    playback_url: str | None = None
    event_count: int = 0
    exists_on_disk: bool = True


# ------------------------------------------------------------------ analysis
class ObjectCount(BaseModel):
    object_class: str
    count: int


class IdentityAppearance(BaseModel):
    identity_id: int | None
    label: str
    category: str | None
    recognition_state: str
    appearances: int
    last_seen_at: datetime | None


class SourceAnalysis(BaseModel):
    source_id: int
    source_uid: str
    source_name: str
    window_hours: int
    objects: list[ObjectCount]
    total_objects: int
    people: int
    identities: list[IdentityAppearance]
    permanent_count: int
    temporary_count: int
    unfamiliar_count: int
    unrecognizable_count: int
    motion_events: int
    recognition_events: int
    active_alerts: int
    event_counts: dict[str, int]
    timeline: list[TimelineEntry]
    pending_review: int
    runtime: dict[str, Any] = Field(default_factory=dict)


# -------------------------------------------------------------------- system
class SystemState(BaseModel):
    surveillance_active: bool
    intelligence_active: bool
    intelligence_switch: bool
    running_sources: int
    recording_sources: int
    active_tracks: int
    active_alerts: int = 0
    pending_review: int = 0
    database_connected: bool = True
    device: str = "cpu"


class ModelInfo(BaseModel):
    key: str
    filename: str
    installed: bool
    path: str | None
    size_bytes: int | None
    approx_mb: float
    licence: str
    origin: str
    purpose: str


class DiagnosticsResponse(BaseModel):
    device: str
    hardware: dict[str, Any]
    system: dict[str, Any]
    sources: list[dict[str, Any]]
    recognition_index: dict[str, Any]
    models: list[ModelInfo]
    process: dict[str, Any]
    websocket_subscribers: int
