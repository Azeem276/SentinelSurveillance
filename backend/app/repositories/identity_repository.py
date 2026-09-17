"""Identity, face and embedding persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import and_, desc, func, or_, select, update

from app.intelligence.face.embedder import from_bytes, to_bytes
from app.intelligence.face.index import IndexEntry
from app.models.enums import (
    FaceReviewStatus, IdentityCategory, IdentityStatus, RecognitionState,
)
from app.models.identity import (
    Face, FaceEmbedding, FaceProfileSample, Identity, TemporaryIdentityExpiration,
)
from app.repositories.base import BaseRepository


def generate_identifier(first_detected_at: datetime, *, prefix: str = "Unknown") -> str:
    """Stable, human-readable fallback identifier: Unknown_YYYYMMDD_HHMMSS.

    Derived from the original detection time, never random, so an operator can
    correlate it with the event timeline.
    """
    ts = first_detected_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return f"{prefix}_{ts.astimezone().strftime('%Y%m%d_%H%M%S')}"


class IdentityRepository(BaseRepository[Identity]):
    model = Identity

    # ------------------------------------------------------------- create
    def unique_identifier(self, base: str) -> str:
        """Ensure the generated identifier does not collide."""
        candidate = base
        suffix = 1
        while self.session.execute(
            select(Identity.id).where(Identity.generated_identifier == candidate)
        ).first():
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    def create(
        self,
        *,
        category: IdentityCategory,
        first_detected_at: datetime,
        display_name: str | None = None,
        retention_days: int | None = None,
        source_id: int | None = None,
        notes: str | None = None,
        thumbnail_path: str | None = None,
    ) -> Identity:
        identifier = self.unique_identifier(generate_identifier(first_detected_at))
        expires_at: datetime | None = None
        if category is IdentityCategory.TEMPORARY:
            days = retention_days or 7
            expires_at = datetime.now(timezone.utc) + timedelta(days=days)

        identity = Identity(
            generated_identifier=identifier,
            display_name=(display_name or "").strip() or None,
            category=category,
            status=IdentityStatus.ACTIVE,
            first_detected_at=first_detected_at,
            expires_at=expires_at,
            retention_days=retention_days if category is IdentityCategory.TEMPORARY else None,
            source_id=source_id,
            notes=notes,
            thumbnail_path=thumbnail_path,
            extra={},
        )
        self.add(identity)
        if expires_at is not None:
            self.session.add(
                TemporaryIdentityExpiration(
                    identity_id=identity.id,
                    scheduled_for=expires_at,
                    retention_days=retention_days,
                    created_at=datetime.now(timezone.utc),
                )
            )
            self.session.flush()
        return identity

    # ------------------------------------------------------------ queries
    def get_by_identifier(self, identifier: str) -> Identity | None:
        return self.session.execute(
            select(Identity).where(Identity.generated_identifier == identifier)
        ).scalar_one_or_none()

    def find_by_name(self, name: str) -> Identity | None:
        return self.session.execute(
            select(Identity).where(
                func.lower(Identity.display_name) == name.strip().lower(),
                Identity.status == IdentityStatus.ACTIVE,
            )
        ).scalar_one_or_none()

    def list_identities(
        self,
        *,
        category: IdentityCategory | None = None,
        status: IdentityStatus | None = IdentityStatus.ACTIVE,
        search: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Identity]:
        stmt = select(Identity)
        if category is not None:
            stmt = stmt.where(Identity.category == category)
        if status is not None:
            stmt = stmt.where(Identity.status == status)
        if search:
            pattern = f"%{search.strip().lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(Identity.display_name).like(pattern),
                    func.lower(Identity.generated_identifier).like(pattern),
                )
            )
        stmt = stmt.order_by(desc(Identity.created_at)).limit(limit).offset(offset)
        return list(self.session.execute(stmt).scalars().all())

    def active_index_records(
        self, *, model_name: str | None = None, dim: int | None = None
    ) -> list[tuple[IndexEntry, np.ndarray]]:
        """Everything the recognition index needs, in one query.

        Embeddings are filtered to the *current* embedder. Vectors produced by
        a different model are not comparable with today's queries, and letting
        them sit in the matcher produces silently meaningless similarities
        rather than an error anybody would notice.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            select(
                Identity.id,
                Identity.display_name,
                Identity.generated_identifier,
                Identity.category,
                Identity.expires_at,
                FaceEmbedding.vector,
                FaceEmbedding.dim,
            )
            .join(FaceEmbedding, FaceEmbedding.identity_id == Identity.id)
            .where(
                Identity.status == IdentityStatus.ACTIVE,
                or_(Identity.expires_at.is_(None), Identity.expires_at > now),
            )
        )
        if model_name:
            stmt = stmt.where(FaceEmbedding.model_name == model_name)
        if dim:
            stmt = stmt.where(FaceEmbedding.dim == dim)
        rows = self.session.execute(stmt).all()

        records: list[tuple[IndexEntry, np.ndarray]] = []
        for ident_id, name, identifier, category, expires_at, blob, dim in rows:
            try:
                vector = from_bytes(blob, int(dim))
            except ValueError:
                continue
            records.append(
                (
                    IndexEntry(
                        identity_id=int(ident_id),
                        label=name or identifier,
                        category=IdentityCategory(str(getattr(category, "value", category))),
                        expires_at=expires_at,
                    ),
                    vector,
                )
            )
        return records

    # ------------------------------------------------------------ updates
    def rename(self, identity_id: int, display_name: str | None) -> None:
        self.session.execute(
            update(Identity)
            .where(Identity.id == identity_id)
            .values(display_name=(display_name or "").strip() or None)
        )

    def set_category(
        self, identity_id: int, category: IdentityCategory, *, retention_days: int | None = None
    ) -> None:
        values: dict = {"category": category}
        if category is IdentityCategory.PERMANENT:
            values["expires_at"] = None
            values["retention_days"] = None
        else:
            days = retention_days or 7
            values["retention_days"] = days
            values["expires_at"] = datetime.now(timezone.utc) + timedelta(days=days)
        self.session.execute(update(Identity).where(Identity.id == identity_id).values(**values))
        if category is IdentityCategory.TEMPORARY:
            self.session.add(
                TemporaryIdentityExpiration(
                    identity_id=identity_id,
                    scheduled_for=values["expires_at"],
                    retention_days=values["retention_days"],
                    created_at=datetime.now(timezone.utc),
                )
            )
        self.session.flush()

    def touch_seen(self, identity_id: int, when: datetime) -> None:
        self.session.execute(
            update(Identity).where(Identity.id == identity_id).values(last_seen_at=when)
        )

    def soft_delete(self, identity_id: int) -> None:
        self.session.execute(
            update(Identity)
            .where(Identity.id == identity_id)
            .values(status=IdentityStatus.DELETED)
        )

    # --------------------------------------------------------- expiration
    def due_for_expiry(self, *, now: datetime | None = None) -> list[Identity]:
        now = now or datetime.now(timezone.utc)
        return list(
            self.session.execute(
                select(Identity).where(
                    Identity.category == IdentityCategory.TEMPORARY,
                    Identity.status == IdentityStatus.ACTIVE,
                    Identity.expires_at.is_not(None),
                    Identity.expires_at <= now,
                )
            )
            .scalars()
            .all()
        )

    def mark_expired(self, identity_id: int, *, when: datetime | None = None) -> None:
        """Expire an identity. Historical events and faces are left untouched."""
        when = when or datetime.now(timezone.utc)
        self.session.execute(
            update(Identity)
            .where(Identity.id == identity_id)
            .values(status=IdentityStatus.EXPIRED)
        )
        self.session.execute(
            update(TemporaryIdentityExpiration)
            .where(
                and_(
                    TemporaryIdentityExpiration.identity_id == identity_id,
                    TemporaryIdentityExpiration.expired_at.is_(None),
                )
            )
            .values(expired_at=when)
        )

    def counts_by_category(self) -> dict[str, int]:
        rows = self.session.execute(
            select(Identity.category, func.count(Identity.id))
            .where(Identity.status == IdentityStatus.ACTIVE)
            .group_by(Identity.category)
        ).all()
        out = {"PERMANENT": 0, "TEMPORARY": 0}
        for category, count in rows:
            out[str(getattr(category, "value", category))] = int(count)
        return out


class FaceRepository(BaseRepository[Face]):
    model = Face

    def list_unfamiliar(
        self,
        *,
        review_status: FaceReviewStatus | None = FaceReviewStatus.PENDING,
        source_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Face]:
        stmt = select(Face).where(Face.recognition_state == RecognitionState.UNFAMILIAR)
        if review_status is not None:
            stmt = stmt.where(Face.review_status == review_status)
        if source_id is not None:
            stmt = stmt.where(Face.source_id == source_id)
        return list(
            self.session.execute(
                stmt.order_by(desc(Face.detected_at)).limit(limit).offset(offset)
            )
            .scalars()
            .all()
        )

    def pending_review_count(self) -> int:
        return int(
            self.session.execute(
                select(func.count(Face.id)).where(
                    Face.recognition_state == RecognitionState.UNFAMILIAR,
                    Face.review_status == FaceReviewStatus.PENDING,
                )
            ).scalar()
            or 0
        )

    def list_for_source(
        self, source_id: int, *, limit: int = 100, since: datetime | None = None
    ) -> list[Face]:
        stmt = select(Face).where(Face.source_id == source_id)
        if since:
            stmt = stmt.where(Face.detected_at >= since)
        return list(
            self.session.execute(stmt.order_by(desc(Face.detected_at)).limit(limit))
            .scalars()
            .all()
        )

    def list_for_identity(self, identity_id: int, *, limit: int = 50) -> list[Face]:
        return list(
            self.session.execute(
                select(Face)
                .where(Face.identity_id == identity_id)
                .order_by(desc(Face.detected_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )

    def classify(
        self,
        face_id: int,
        *,
        identity_id: int,
        recognition_state: RecognitionState,
        when: datetime | None = None,
    ) -> None:
        self.session.execute(
            update(Face)
            .where(Face.id == face_id)
            .values(
                identity_id=identity_id,
                recognition_state=recognition_state,
                review_status=FaceReviewStatus.CLASSIFIED,
                reviewed_at=when or datetime.now(timezone.utc),
            )
        )

    def dismiss(self, face_id: int) -> None:
        self.session.execute(
            update(Face)
            .where(Face.id == face_id)
            .values(
                review_status=FaceReviewStatus.DISMISSED,
                reviewed_at=datetime.now(timezone.utc),
            )
        )

    def reassign(self, from_identity_id: int, to_identity_id: int) -> int:
        """Move stored face crops to the surviving identity after a merge."""
        result = self.session.execute(
            update(Face)
            .where(Face.identity_id == from_identity_id)
            .values(identity_id=to_identity_id)
        )
        self.session.flush()
        return int(result.rowcount or 0)

    def best_for_track(self, track_id: int) -> Face | None:
        return self.session.execute(
            select(Face)
            .where(Face.track_id == track_id)
            .order_by(desc(Face.quality_score))
            .limit(1)
        ).scalar_one_or_none()


class FaceEmbeddingRepository(BaseRepository[FaceEmbedding]):
    model = FaceEmbedding

    def add_embedding(
        self,
        *,
        identity_id: int,
        vector: np.ndarray,
        model_name: str,
        face_id: int | None = None,
        origin: str = "classification",
    ) -> FaceEmbedding:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        return self.add(
            FaceEmbedding(
                identity_id=identity_id,
                face_id=face_id,
                vector=to_bytes(array),
                dim=int(array.size),
                model_name=model_name,
                origin=origin,
                created_at=datetime.now(timezone.utc),
            )
        )

    def list_for_identity(self, identity_id: int) -> list[FaceEmbedding]:
        return list(
            self.session.execute(
                select(FaceEmbedding).where(FaceEmbedding.identity_id == identity_id)
            )
            .scalars()
            .all()
        )

    def count_for_identity(self, identity_id: int) -> int:
        return int(
            self.session.execute(
                select(func.count(FaceEmbedding.id)).where(
                    FaceEmbedding.identity_id == identity_id
                )
            ).scalar()
            or 0
        )

    def has_origin_key(self, identity_id: int, origin: str) -> bool:
        """Has this exact enrolment source already been ingested?

        Used to stop ``enrol-dataset`` re-embedding the same photo every time
        it is run, which previously grew an identity's gallery with duplicates
        on every sweep.
        """
        return (
            self.session.execute(
                select(FaceEmbedding.id).where(
                    FaceEmbedding.identity_id == identity_id,
                    FaceEmbedding.origin == origin,
                )
            ).first()
            is not None
        )

    def reassign(self, from_identity_id: int, to_identity_id: int) -> int:
        """Move every embedding from one identity to another (merge)."""
        result = self.session.execute(
            update(FaceEmbedding)
            .where(FaceEmbedding.identity_id == from_identity_id)
            .values(identity_id=to_identity_id)
        )
        self.session.flush()
        return int(result.rowcount or 0)


class FaceProfileSampleRepository(BaseRepository[FaceProfileSample]):
    """The multi-angle gallery collected for one unreviewed person.

    These rows exist only between "the pipeline decided this person is
    unknown" and "an operator classified or dismissed them". They are the
    reason a newly created identity knows what somebody looks like from every
    angle the camera saw, rather than from the one crop that happened to be
    saved first.
    """

    model = FaceProfileSample

    def add_sample(
        self,
        *,
        face_id: int,
        source_id: int,
        vector: np.ndarray,
        model_name: str,
        track_id: int | None = None,
        quality_score: float = 0.0,
        yaw: float = 0.0,
        brightness: float = 0.0,
        face_pixels: int | None = None,
        image_path: str | None = None,
        frame_number: int = 0,
    ) -> FaceProfileSample:
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        return self.add(
            FaceProfileSample(
                face_id=face_id,
                source_id=source_id,
                track_id=track_id,
                vector=to_bytes(array),
                dim=int(array.size),
                model_name=model_name,
                quality_score=float(quality_score),
                yaw=float(yaw),
                brightness=float(brightness),
                face_pixels=face_pixels,
                image_path=image_path,
                frame_number=int(frame_number),
                created_at=datetime.now(timezone.utc),
            )
        )

    def list_for_face(self, face_id: int) -> list[FaceProfileSample]:
        return list(
            self.session.execute(
                select(FaceProfileSample)
                .where(FaceProfileSample.face_id == face_id)
                .order_by(desc(FaceProfileSample.quality_score))
            )
            .scalars()
            .all()
        )

    def count_for_face(self, face_id: int) -> int:
        return int(
            self.session.execute(
                select(func.count(FaceProfileSample.id)).where(
                    FaceProfileSample.face_id == face_id
                )
            ).scalar()
            or 0
        )

    def delete_for_face(self, face_id: int) -> int:
        samples = self.list_for_face(face_id)
        for sample in samples:
            self.session.delete(sample)
        self.session.flush()
        return len(samples)
