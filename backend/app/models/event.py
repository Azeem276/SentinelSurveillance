"""Security events, alerts and key/value system settings."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigIntPK, JSONType, TimestampMixin
from app.models.enums import AlertState, AlertType, EventSeverity, EventType


class SecurityEvent(Base):
    """A structured, de-duplicated event.

    Continuing conditions (motion, an unknown person in frame, a proximity
    breach) are represented as ONE row whose ended_at/duration is updated,
    never one row per frame.
    """

    __tablename__ = "security_events"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[EventType] = mapped_column(
        Enum(EventType, native_enum=False, length=48, validate_strings=True), nullable=False
    )
    severity: Mapped[EventSeverity] = mapped_column(
        Enum(EventSeverity, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=EventSeverity.INFO,
    )
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="SET NULL"))
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL")
    )
    face_id: Mapped[int | None] = mapped_column(ForeignKey("faces.id", ondelete="SET NULL"))
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="SET NULL")
    )
    motion_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("motion_events.id", ondelete="SET NULL")
    )

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    confidence: Mapped[float | None] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(255))
    message: Mapped[str | None] = mapped_column(Text)
    snapshot_path: Mapped[str | None] = mapped_column(String(512))
    # Dedup key: identical open events with the same key are merged.
    dedup_key: Mapped[str | None] = mapped_column(String(255))
    event_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONType, nullable=False, default=dict
    )

    identity = relationship("Identity", lazy="joined")

    __table_args__ = (
        Index("ix_security_events_source_started", "source_id", "started_at"),
        Index("ix_security_events_event_type", "event_type"),
        Index("ix_security_events_track_id", "track_id"),
        Index("ix_security_events_identity_id", "identity_id"),
        Index("ix_security_events_open", "is_open"),
        Index("ix_security_events_dedup", "source_id", "dedup_key", "is_open"),
        Index("ix_security_events_recording_id", "recording_id"),
    )


class Alert(Base):
    """An audible/visual alert raised by the security rule engine."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    security_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("security_events.id", ondelete="SET NULL")
    )
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="SET NULL"))
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL")
    )
    alert_type: Mapped[AlertType] = mapped_column(
        Enum(AlertType, native_enum=False, length=24, validate_strings=True), nullable=False
    )
    state: Mapped[AlertState] = mapped_column(
        Enum(AlertState, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=AlertState.ACTIVE,
    )
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_metadata: Mapped[dict] = mapped_column("metadata", JSONType, nullable=False, default=dict)

    identity = relationship("Identity", lazy="joined")

    __table_args__ = (
        Index("ix_alerts_source_started", "source_id", "started_at"),
        Index("ix_alerts_state", "state"),
        Index("ix_alerts_track_id", "track_id"),
    )


class SystemSetting(Base, TimestampMixin):
    """Runtime system state and operator preferences (surveillance on/off, ...)."""

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    description: Mapped[str | None] = mapped_column(Text)
