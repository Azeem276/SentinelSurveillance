"""Identity, face crop and face-embedding models.

Deliberately three separate concepts:

    Identity        - a person known to the system (permanent or temporary)
    Face            - one detected face crop observed at a point in time
    FaceEmbedding   - one vector belonging to an identity (many per identity)

A Track (see app.models.detection) references an Identity but is never the
same thing as one.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index,
    Integer, LargeBinary, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BigIntPK, JSONType, TimestampMixin
from app.models.enums import (
    FaceReviewStatus, IdentityCategory, IdentityStatus, RecognitionState,
)


class Identity(Base, TimestampMixin):
    __tablename__ = "identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Human-readable, stable, always populated. When the operator supplies no
    # name this is derived from the first detection time: Unknown_YYYYMMDD_HHMMSS
    generated_identifier: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[IdentityCategory] = mapped_column(
        Enum(IdentityCategory, native_enum=False, length=16, validate_strings=True),
        nullable=False,
    )
    status: Mapped[IdentityStatus] = mapped_column(
        Enum(IdentityStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=IdentityStatus.ACTIVE,
    )
    first_detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retention_days: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512))
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("video_sources.id", ondelete="SET NULL")
    )
    extra: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)

    embeddings = relationship(
        "FaceEmbedding", back_populates="identity", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint(
            "(category <> 'TEMPORARY') OR (expires_at IS NOT NULL)",
            name="temporary_requires_expiry",
        ),
        Index("ix_identities_category_status", "category", "status"),
        Index("ix_identities_expires_at", "expires_at"),
        Index("ix_identities_display_name", "display_name"),
    )

    @property
    def label(self) -> str:
        return self.display_name or self.generated_identifier


class Face(Base):
    """A stored face crop plus its quality metrics and review status."""

    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("video_sources.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="SET NULL"))
    identity_id: Mapped[int | None] = mapped_column(
        ForeignKey("identities.id", ondelete="SET NULL")
    )
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="SET NULL")
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    image_path: Mapped[str | None] = mapped_column(String(512))

    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    blur_score: Mapped[float | None] = mapped_column(Float)
    brightness: Mapped[float | None] = mapped_column(Float)
    face_pixels: Mapped[int | None] = mapped_column(Integer)
    quality_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_reason: Mapped[str | None] = mapped_column(String(255))

    recognition_state: Mapped[RecognitionState] = mapped_column(
        Enum(RecognitionState, native_enum=False, length=32, validate_strings=True),
        nullable=False,
        default=RecognitionState.FACE_UNRECOGNIZABLE,
    )
    match_score: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[FaceReviewStatus] = mapped_column(
        Enum(FaceReviewStatus, native_enum=False, length=16, validate_strings=True),
        nullable=False,
        default=FaceReviewStatus.PENDING,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    identity = relationship("Identity", lazy="joined")

    __table_args__ = (
        Index("ix_faces_source_detected", "source_id", "detected_at"),
        Index("ix_faces_review_status", "review_status"),
        Index("ix_faces_identity_id", "identity_id"),
        Index("ix_faces_track_id", "track_id"),
        Index("ix_faces_recognition_state", "recognition_state"),
    )


class FaceEmbedding(Base):
    """A face descriptor vector belonging to an identity.

    Stored as raw little-endian float32 bytes plus its dimensionality, so the
    schema does not require the pgvector extension. The similarity search runs
    in the in-memory recognition index (app.intelligence.face.index) which is
    rebuilt whenever the identity dataset changes.
    """

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"), nullable=False
    )
    face_id: Mapped[int | None] = mapped_column(ForeignKey("faces.id", ondelete="SET NULL"))
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    origin: Mapped[str] = mapped_column(String(32), nullable=False, default="classification")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    identity = relationship("Identity", back_populates="embeddings")

    __table_args__ = (
        Index("ix_face_embeddings_identity_id", "identity_id"),
        Index("ix_face_embeddings_model_name", "model_name"),
        UniqueConstraint("face_id", "identity_id", name="uq_face_embeddings_face_identity"),
    )


class TemporaryIdentityExpiration(Base):
    """Audit trail of scheduled and performed temporary-identity expirations."""

    __tablename__ = "temporary_identity_expirations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identities.id", ondelete="CASCADE"), nullable=False
    )
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retention_days: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_temp_identity_exp_scheduled", "scheduled_for"),
        Index("ix_temp_identity_exp_identity", "identity_id"),
    )
