"""Tracks, detections and motion events."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigIntPK, JSONType, TimestampMixin
from app.models.enums import ProximityZone, RecognitionState, TrackStatus


class Track(Base, TimestampMixin):
    """A tracking session for one object on one source.

    track_key is the tracker-local integer id and is NOT an identity.
    """

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="SET NULL")
    )
    track_key: Mapped[int] = mapped_column(Integer, nullable=False)
    object_class: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[TrackStatus] = mapped_column(
        Enum(TrackStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=TrackStatus.ACTIVE,
    )
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    detection_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Person-only enrichment; stays NO_FACE for dogs, cars and other objects.
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL")
    )
    recognition_state: Mapped[RecognitionState] = mapped_column(
        Enum(RecognitionState, native_enum=False, length=32, validate_strings=True),
        nullable=False,
        default=RecognitionState.NO_FACE,
    )
    recognition_confidence: Mapped[float | None] = mapped_column(Float)
    proximity_zone: Mapped[ProximityZone] = mapped_column(
        Enum(ProximityZone, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=ProximityZone.UNKNOWN,
    )
    min_distance_m: Mapped[float | None] = mapped_column(Float)
    last_distance_m: Mapped[float | None] = mapped_column(Float)
    entered_zone_a: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    entered_zone_b: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trajectory: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    snapshot_path: Mapped[str | None] = mapped_column(String(512))

    identity = relationship("Identity", lazy="joined")

    __table_args__ = (
        Index("ix_tracks_source_track_key", "source_id", "track_key"),
        Index("ix_tracks_source_first_seen", "source_id", "first_seen_at"),
        Index("ix_tracks_identity_id", "identity_id"),
        Index("ix_tracks_status", "status"),
        Index("ix_tracks_object_class", "object_class"),
    )


class Detection(Base):
    """One object observation in one frame."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"))
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="SET NULL")
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    object_class: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)
    distance_m: Mapped[float | None] = mapped_column(Float)
    proximity_zone: Mapped[ProximityZone | None] = mapped_column(
        Enum(ProximityZone, native_enum=False, length=16, validate_strings=True)
    )

    __table_args__ = (
        Index("ix_detections_source_timestamp", "source_id", "timestamp"),
        Index("ix_detections_track_id", "track_id"),
        Index("ix_detections_object_class", "object_class"),
        Index("ix_detections_recording_id", "recording_id"),
    )


class MotionEvent(Base):
    """Aggregated motion activity: debounced, not one row per frame."""

    __tablename__ = "motion_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="SET NULL")
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    peak_area_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    frame_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frame_end: Mapped[int | None] = mapped_column(Integer)
    snapshot_path: Mapped[str | None] = mapped_column(String(512))

    __table_args__ = (Index("ix_motion_events_source_started", "source_id", "started_at"),)
