"""Recording sessions: one row per uninterrupted capture of one source."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import RecordingStatus


class RecordingSession(Base, TimestampMixin):
    __tablename__ = "recording_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    # Path relative to RECORDING_PATH, e.g. "source_01/recording_...mp4".
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[RecordingStatus] = mapped_column(
        Enum(RecordingStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=RecordingStatus.ACTIVE,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    codec: Mapped[str | None] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)

    source = relationship("VideoSource", back_populates="recordings", lazy="joined")

    __table_args__ = (
        Index("ix_recording_sessions_source_started", "source_id", "started_at"),
        Index("ix_recording_sessions_status", "status"),
    )
