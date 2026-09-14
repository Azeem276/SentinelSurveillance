"""Video source configuration (file today, RTSP tomorrow)."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Enum, Float, Index, Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JSONType, TimestampMixin
from app.models.enums import SourceStatus, SourceType


class VideoSource(Base, TimestampMixin):
    __tablename__ = "video_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=SourceType.FILE,
    )
    # For FILE sources this is a path *relative to VIDEO_STORAGE_PATH*; never an
    # absolute path supplied by an API client (see app.storage.paths).
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    surveillance_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    intelligence_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    recognition_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    recording_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    loop_playback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    status: Mapped[SourceStatus] = mapped_column(
        Enum(SourceStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=SourceStatus.IDLE,
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- per-source intelligence tuning -------------------------------------
    proximity_a: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    proximity_b: Mapped[float] = mapped_column(Float, nullable=False, default=3.0)
    recognition_threshold: Mapped[float | None] = mapped_column(Float)
    detection_confidence: Mapped[float | None] = mapped_column(Float)
    temporary_retention_days: Mapped[int | None] = mapped_column(Integer)
    alert_policy: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    # Calibration used by the proximity estimator (fov, reference height, ...).
    calibration: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    recordings = relationship("RecordingSession", back_populates="source", lazy="selectin")

    __table_args__ = (
        CheckConstraint("proximity_a > proximity_b", name="proximity_a_gt_b"),
        CheckConstraint("proximity_b > 0", name="proximity_b_positive"),
        CheckConstraint(
            "recognition_threshold IS NULL OR "
            "(recognition_threshold > 0 AND recognition_threshold <= 1)",
            name="recognition_threshold_range",
        ),
        Index("ix_video_sources_status", "status"),
        Index("ix_video_sources_enabled", "enabled"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<VideoSource {self.uid} {self.name}>"
