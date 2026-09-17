"""Identity lifecycle: classification, naming, enrolment, expiry, sync.

The invariant this service protects: whenever the identity dataset changes -
created, renamed, re-categorised, deleted, expired, or given a new embedding -
the recognition index is rebuilt before the call returns. Subsequent frames
therefore see the new dataset immediately.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import (
    IDENTITY_UPDATED, TEMPORARY_IDENTITY_EXPIRED, get_logger,
)
from app.db.session import session_scope
from app.events.bus import get_bus
from app.events.engine import EventContext, EventEmitter
from app.intelligence.face.embedder import SFaceEmbedder, from_bytes
from app.intelligence.face.index import get_index
from app.models.enums import (
    EventType, FaceReviewStatus, IdentityCategory, IdentityStatus, RecognitionState,
)
from app.models.identity import Identity
from app.repositories.detection_repository import TrackRepository
from app.repositories.identity_repository import (
    FaceEmbeddingRepository, FaceProfileSampleRepository, FaceRepository,
    IdentityRepository, generate_identifier,
)
from app.storage.paths import ensure_parent, face_root, relative_to_root, slugify

log = get_logger(__name__)

VALID_RETENTIONS = {1, 3, 7, 30}
CATEGORY_DIRECTORY = {
    IdentityCategory.PERMANENT: "permanent",
    IdentityCategory.TEMPORARY: "temporary",
}


def _enrolment_key(folder: str, person: str, filename: str) -> str:
    """Stable per-file provenance key that fits the 32-char origin column."""
    digest = hashlib.sha1(
        f"{folder}/{person}/{filename}".encode("utf-8")
    ).hexdigest()[:24]
    return f"enrol:{digest}"


@dataclass(slots=True)
class ClassificationResult:
    identity_id: int
    identifier: str
    display_name: str | None
    category: str
    expires_at: str | None
    embeddings_added: int
    created: bool


_embedder: SFaceEmbedder | None = None


def get_embedder() -> SFaceEmbedder:
    """Shared embedder used for enrolment (the pipeline owns its own)."""
    global _embedder
    if _embedder is None:
        _embedder = SFaceEmbedder()
    return _embedder


class IdentityService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.identities = IdentityRepository(session)
        self.faces = FaceRepository(session)
        self.embeddings = FaceEmbeddingRepository(session)
        self.profile_samples = FaceProfileSampleRepository(session)
        self.tracks = TrackRepository(session)
        self.index = get_index()
        self.bus = get_bus()

    # ---------------------------------------------------------- index sync
    def sync_index(self) -> dict:
        """Rebuild the recognition index from the current active dataset.

        Only embeddings from the embedder currently in use are loaded. Vectors
        from an older model are left in the database for provenance but kept
        out of the matcher, where they would otherwise produce meaningless
        similarity scores that nothing would flag.
        """
        embedder = get_embedder()
        records = self.identities.active_index_records(
            model_name=embedder.name, dim=embedder.dim
        )
        self.index.rebuild(records)
        stats = self.index.stats()
        self.bus.emit("identity_index", source_uid=None, **stats)
        return stats

    # ------------------------------------------------------------ creation
    def _resolve_retention(self, category: IdentityCategory, days: int | None) -> int | None:
        if category is IdentityCategory.PERMANENT:
            return None
        settings = get_settings()
        value = days or settings.temporary_familiar_default_days
        if value <= 0:
            raise ValidationError("Retention must be a positive number of days")
        if value > 3650:
            raise ValidationError("Retention cannot exceed 10 years")
        return int(value)

    def create_identity(
        self,
        *,
        category: IdentityCategory,
        display_name: str | None = None,
        first_detected_at: datetime | None = None,
        retention_days: int | None = None,
        source_id: int | None = None,
        notes: str | None = None,
    ) -> Identity:
        detected = first_detected_at or datetime.now(timezone.utc)
        identity = self.identities.create(
            category=category,
            first_detected_at=detected,
            display_name=display_name,
            retention_days=self._resolve_retention(category, retention_days),
            source_id=source_id,
            notes=notes,
        )
        log.info(
            IDENTITY_UPDATED, action="created", identity_id=identity.id,
            identifier=identity.generated_identifier, category=category.value,
        )
        return identity

    # ------------------------------------------------------ classification
    def classify_face(
        self,
        face_id: int,
        *,
        category: IdentityCategory,
        display_name: str | None = None,
        retention_days: int | None = None,
        identity_id: int | None = None,
    ) -> ClassificationResult:
        """Promote one reviewed unfamiliar face into a familiar identity.

        A blank name yields the timestamp-derived identifier
        ``Unknown_YYYYMMDD_HHMMSS`` built from the ORIGINAL detection time.
        """
        face = self.faces.get(face_id)
        if face is None:
            raise NotFoundError(f"face {face_id} not found")

        name = (display_name or "").strip() or None
        created = False

        if identity_id is not None:
            identity = self.identities.get(identity_id)
            if identity is None:
                raise NotFoundError(f"identity {identity_id} not found")
        else:
            identity = self.identities.create(
                category=category,
                first_detected_at=face.detected_at,
                display_name=name,
                retention_days=self._resolve_retention(category, retention_days),
                source_id=face.source_id,
            )
            created = True

        if not created:
            if name:
                self.identities.rename(identity.id, name)
            self.identities.set_category(
                identity.id, category,
                retention_days=self._resolve_retention(category, retention_days),
            )

        added = self._enrol_face(identity, face, category)

        self.faces.classify(
            face_id,
            identity_id=identity.id,
            recognition_state=(
                RecognitionState.PERMANENT_FAMILIAR
                if category is IdentityCategory.PERMANENT
                else RecognitionState.TEMPORARY_FAMILIAR
            ),
        )
        # Historical tracks for this face gain the identity too, so the
        # timeline reads correctly after review.
        if face.track_id is not None:
            self.tracks.update_state(
                face.track_id,
                identity_id=identity.id,
                recognition_state=(
                    RecognitionState.PERMANENT_FAMILIAR
                    if category is IdentityCategory.PERMANENT
                    else RecognitionState.TEMPORARY_FAMILIAR
                ),
            )

        self.session.flush()
        self.session.refresh(identity)

        EventEmitter(
            self.session, EventContext(face.source_id, str(face.source_id))
        ).emit(
            EventType.IDENTITY_CLASSIFIED,
            track_id=face.track_id,
            identity_id=identity.id,
            face_id=face.id,
            label=identity.display_name or identity.generated_identifier,
            message=(
                f"Classified as {category.value} familiar: "
                f"{identity.display_name or identity.generated_identifier}"
            ),
            metadata={"category": category.value, "created": created},
        )

        self.session.flush()
        self.sync_index()
        log.info(
            IDENTITY_UPDATED, action="classified", identity_id=identity.id,
            identifier=identity.generated_identifier, category=category.value,
            embeddings_added=added,
        )
        return ClassificationResult(
            identity_id=identity.id,
            identifier=identity.generated_identifier,
            display_name=identity.display_name,
            category=category.value,
            expires_at=identity.expires_at.isoformat() if identity.expires_at else None,
            embeddings_added=added,
            created=created,
        )

    def bulk_classify(
        self,
        face_ids: list[int],
        *,
        category: IdentityCategory,
        display_name: str | None = None,
        retention_days: int | None = None,
        merge_into_one_identity: bool = False,
    ) -> list[ClassificationResult]:
        """Classify several faces at once.

        Distinct people are NOT merged unless the caller explicitly asks:
        by default each face becomes its own identity, each with its own
        timestamp-derived identifier. A shared name is applied to all of them.
        """
        if not face_ids:
            raise ValidationError("No faces selected")

        results: list[ClassificationResult] = []
        shared_identity_id: int | None = None

        for face_id in face_ids:
            result = self.classify_face(
                face_id,
                category=category,
                display_name=display_name,
                retention_days=retention_days,
                identity_id=shared_identity_id,
            )
            if merge_into_one_identity and shared_identity_id is None:
                shared_identity_id = result.identity_id
            results.append(result)
        return results

    def _enrol_face(
        self, identity: Identity, face, category: IdentityCategory
    ) -> int:
        """Enrol a reviewed person: the primary crop plus their whole profile.

        Two things matter here and both were previously wrong.

        First, the embedding is produced by the *same* detect-then-align chain
        the live pipeline uses. The old code resized the stored crop straight
        to 112x112, so every gallery vector was computed differently from
        every query vector - a systematic mismatch that depressed similarity
        for everybody and looked exactly like angle sensitivity.

        Second, the multi-angle samples the pipeline collected for this person
        are enrolled alongside it. An identity therefore starts life knowing
        several views of the face rather than the single crop that happened to
        be saved, which is what lets it survive being seen from a new angle.
        """
        embedder = get_embedder()
        added = 0

        # --- the reviewed crop ------------------------------------------
        if face.image_path is not None:
            try:
                source_path = face_root() / face.image_path
                if not source_path.exists():
                    log.warning("face_crop_missing", face_id=face.id, path=str(source_path))
                else:
                    image = cv2.imread(str(source_path))
                    if image is not None:
                        vector = self._embed_dataset_image(image, embedder)
                        if vector is not None:
                            self.embeddings.add_embedding(
                                identity_id=identity.id,
                                vector=vector,
                                model_name=embedder.name,
                                face_id=face.id,
                                origin="classification",
                            )
                            added += 1

                        # File the crop under the category for the archive.
                        target_dir = face_root() / CATEGORY_DIRECTORY[category]
                        target = target_dir / (
                            f"{slugify(identity.generated_identifier)}_{face.id}"
                            f"{source_path.suffix}"
                        )
                        ensure_parent(target)
                        shutil.copy2(source_path, target)
                        relative = relative_to_root(target, face_root())
                        face.image_path = relative
                        if not identity.thumbnail_path:
                            identity.thumbnail_path = relative
            except Exception as exc:
                log.warning("face_enrolment_failed", face_id=face.id, error=str(exc))

        added += self._enrol_profile_samples(identity, face, embedder)
        self.session.flush()
        return added

    def _enrol_profile_samples(self, identity: Identity, face, embedder) -> int:
        """Enrol the multi-angle gallery collected while this person was live.

        These vectors were produced by the live pipeline, so they already went
        through the aligned path and need no re-embedding - they are exactly
        what the matcher will be comparing against in future.
        """
        try:
            samples = self.profile_samples.list_for_face(face.id)
        except Exception as exc:  # pragma: no cover - table may predate upgrade
            log.warning("profile_samples_unavailable", face_id=face.id, error=str(exc))
            return 0

        added = 0
        for sample in samples:
            if sample.model_name != embedder.name or sample.dim != embedder.dim:
                # Produced by a different embedder; not comparable.
                continue
            try:
                vector = from_bytes(sample.vector, int(sample.dim))
            except ValueError:
                continue
            self.embeddings.add_embedding(
                identity_id=identity.id,
                vector=vector,
                model_name=embedder.name,
                face_id=None,
                origin="profile",
            )
            added += 1

        if samples:
            self.profile_samples.delete_for_face(face.id)
        if added:
            log.info(
                "profile_gallery_enrolled",
                identity_id=identity.id, face_id=face.id, embeddings=added,
            )
        return added

    # ---------------------------------------------------------------- merge
    def merge_face_into_identity(
        self, face_id: int, identity_id: int
    ) -> ClassificationResult:
        """Attach a reviewed face to an identity that already exists.

        This is the answer to the commonest review mistake: the same person
        being detected as somebody new because the camera saw them from an
        angle their gallery did not cover. Merging adds this appearance -
        crop plus the whole multi-angle profile behind it - to the identity
        the operator knows it is, which both fixes the record and makes the
        next sighting from that angle recognisable.

        The identity keeps its own category and retention; merging is not a
        back door for silently promoting a temporary person to permanent.
        """
        identity = self.identities.get(identity_id)
        if identity is None:
            raise NotFoundError(f"identity {identity_id} not found")
        if identity.status is not IdentityStatus.ACTIVE:
            raise ValidationError(
                "Cannot merge into an expired or deleted identity. "
                "Restore it first, or classify this face as a new identity."
            )
        return self.classify_face(
            face_id,
            category=identity.category,
            display_name=None,
            retention_days=identity.retention_days,
            identity_id=identity.id,
        )

    def merge_identities(self, source_identity_id: int, target_identity_id: int) -> dict:
        """Fold one identity into another, keeping the target.

        Everything the source knew - embeddings, face crops, historical tracks
        - moves across, then the source is removed from recognition. Use this
        when the same person has accumulated two entries; the merged gallery
        is strictly better than either half, because it covers more angles.
        """
        if source_identity_id == target_identity_id:
            raise ValidationError("An identity cannot be merged into itself")

        source = self.identities.get(source_identity_id)
        target = self.identities.get(target_identity_id)
        if source is None:
            raise NotFoundError(f"identity {source_identity_id} not found")
        if target is None:
            raise NotFoundError(f"identity {target_identity_id} not found")
        if target.status is not IdentityStatus.ACTIVE:
            raise ValidationError("The target identity must be active")

        moved_embeddings = self.embeddings.reassign(source.id, target.id)
        moved_faces = self.faces.reassign(source.id, target.id)
        moved_tracks = self.tracks.reassign_identity(source.id, target.id)

        if not target.thumbnail_path and source.thumbnail_path:
            target.thumbnail_path = source.thumbnail_path
        # The earliest sighting of either record is the person's first sighting.
        if source.first_detected_at < target.first_detected_at:
            target.first_detected_at = source.first_detected_at
        if source.last_seen_at and (
            target.last_seen_at is None or source.last_seen_at > target.last_seen_at
        ):
            target.last_seen_at = source.last_seen_at

        self.identities.soft_delete(source.id)
        self.session.flush()

        EventEmitter(
            self.session,
            EventContext(target.source_id or 0, str(target.source_id or "")),
        ).emit(
            EventType.IDENTITY_CLASSIFIED,
            identity_id=target.id,
            label=target.display_name or target.generated_identifier,
            message=(
                f"Merged {source.display_name or source.generated_identifier} into "
                f"{target.display_name or target.generated_identifier}"
            ),
            metadata={
                "merged_from": source.id,
                "embeddings_moved": moved_embeddings,
                "faces_moved": moved_faces,
                "tracks_moved": moved_tracks,
            },
        )

        self.session.flush()
        self.sync_index()

        from app.services.surveillance import get_manager

        # Live tracks still pointing at the absorbed identity must let go.
        get_manager().invalidate_identity(source.id)

        log.info(
            IDENTITY_UPDATED, action="merged", identity_id=target.id,
            merged_from=source.id, embeddings_moved=moved_embeddings,
        )
        return {
            "target_identity_id": target.id,
            "source_identity_id": source.id,
            "embeddings_moved": moved_embeddings,
            "faces_moved": moved_faces,
            "tracks_moved": moved_tracks,
            "embedding_count": self.embeddings.count_for_identity(target.id),
        }

    def merge_candidates(self, face_id: int, *, limit: int = 8) -> list[dict]:
        """Rank existing identities by how well they match a reviewed face.

        Deliberately runs at a *lower* bar than live recognition. The point is
        to surface "this might be Azeem" for a human to confirm, not to make
        the call automatically - the operator has context the matcher does not.
        """
        face = self.faces.get(face_id)
        if face is None:
            raise NotFoundError(f"face {face_id} not found")

        embedder = get_embedder()
        vectors: list[np.ndarray] = []

        for sample in self.profile_samples.list_for_face(face.id):
            if sample.model_name == embedder.name and sample.dim == embedder.dim:
                try:
                    vectors.append(from_bytes(sample.vector, int(sample.dim)))
                except ValueError:
                    continue

        if not vectors and face.image_path:
            path = face_root() / face.image_path
            if path.exists():
                image = cv2.imread(str(path))
                if image is not None:
                    vector = self._embed_dataset_image(image, embedder)
                    if vector is not None:
                        vectors.append(vector)

        if not vectors:
            return []

        matches = self.index.recognize_batch(vectors, threshold=0.0)
        tally: dict[int, list[float]] = {}
        labels: dict[int, str] = {}
        for match in matches:
            if match.identity_id is None:
                continue
            tally.setdefault(match.identity_id, []).append(match.score)
            labels[match.identity_id] = match.label or ""

        ranked = sorted(
            tally.items(), key=lambda kv: float(np.mean(kv[1])), reverse=True
        )[:limit]

        out: list[dict] = []
        for identity_id, scores in ranked:
            identity = self.identities.get(identity_id)
            if identity is None:
                continue
            out.append(
                {
                    "identity_id": identity_id,
                    "label": labels.get(identity_id) or identity.label,
                    "display_name": identity.display_name,
                    "generated_identifier": identity.generated_identifier,
                    "category": identity.category.value,
                    "thumbnail_path": identity.thumbnail_path,
                    "mean_score": round(float(np.mean(scores)), 4),
                    "best_score": round(float(max(scores)), 4),
                    "samples_compared": len(vectors),
                    "embedding_count": self.embeddings.count_for_identity(identity_id),
                }
            )
        return out

    # ---------------------------------------------------- dataset enrolment
    def enrol_from_directory(self, directory: Path | None = None) -> dict:
        """Enrol the on-disk familiar-face dataset.

        Layout (see README):

            data/familiar_faces/
                permanent/Azeem/*.jpg
                temporary/John/*.jpg
        """
        base = Path(directory or get_settings().face_dataset_path)
        if not base.exists():
            return {"enrolled": 0, "identities": 0, "skipped": 0,
                    "message": f"dataset directory not found: {base}"}

        embedder = get_embedder()
        enrolled = 0
        skipped = 0
        identities_touched: set[int] = set()

        for category, folder in (
            (IdentityCategory.PERMANENT, "permanent"),
            (IdentityCategory.TEMPORARY, "temporary"),
        ):
            category_dir = base / folder
            if not category_dir.is_dir():
                continue
            for person_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                name = person_dir.name.strip()
                identity = self.identities.find_by_name(name)
                if identity is None:
                    identity = self.identities.create(
                        category=category,
                        first_detected_at=datetime.now(timezone.utc),
                        display_name=name,
                        retention_days=(
                            None if category is IdentityCategory.PERMANENT
                            else get_settings().temporary_familiar_default_days
                        ),
                        notes=f"enrolled from {folder}/{name}",
                    )
                identities_touched.add(identity.id)

                images = [
                    p for p in sorted(person_dir.iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ]

                for image_path in images:
                    # Dedup per FILE, not per count. The old check compared an
                    # embedding count against an image count, so adding one
                    # photo to a folder re-embedded every photo in it and the
                    # gallery filled with duplicates on each sweep.
                    origin = _enrolment_key(folder, name, image_path.name)
                    if self.embeddings.has_origin_key(identity.id, origin):
                        skipped += 1
                        continue
                    image = cv2.imread(str(image_path))
                    if image is None:
                        skipped += 1
                        continue
                    vector = self._embed_dataset_image(image, embedder)
                    if vector is None:
                        log.warning(
                            "enrolment_image_rejected",
                            identity=name, image=image_path.name,
                            reason="no face detected in the photo",
                        )
                        skipped += 1
                        continue
                    self.embeddings.add_embedding(
                        identity_id=identity.id,
                        vector=vector,
                        model_name=embedder.name,
                        origin=origin,
                    )
                    enrolled += 1
                    if not identity.thumbnail_path:
                        target = face_root() / CATEGORY_DIRECTORY[category] / (
                            f"{slugify(identity.generated_identifier)}{image_path.suffix}"
                        )
                        ensure_parent(target)
                        shutil.copy2(image_path, target)
                        identity.thumbnail_path = relative_to_root(target, face_root())

        self.session.flush()
        stats = self.sync_index()
        log.info(
            "dataset_enrolled", enrolled=enrolled, skipped=skipped,
            identities=len(identities_touched),
        )
        return {
            "enrolled": enrolled,
            "skipped": skipped,
            "identities": len(identities_touched),
            "index": stats,
        }

    @staticmethod
    def _embed_dataset_image(image: np.ndarray, embedder: SFaceEmbedder) -> np.ndarray | None:
        """Detect the face in an enrolment photo, align it, then embed it.

        This is the *same* chain the live pipeline uses, and that is the whole
        point: a gallery vector and a query vector have to be produced
        identically or their cosine similarity means nothing.

        There is deliberately no fallback to embedding the raw image. The old
        code did that when no face was found, which quietly enrolled whole
        rooms and full-body shots as if they were faces and left junk vectors
        competing for matches. A photo with no detectable face is rejected.
        """
        from app.intelligence.face.detector import YuNetFaceDetector

        detector = YuNetFaceDetector()
        try:
            faces = detector.detect(image)
        except Exception:
            faces = []
        if not faces:
            return None
        face = max(faces, key=lambda f: f.bbox.area)
        return embedder.embed(image, face)

    # ---------------------------------------------------------- management
    def update_identity(
        self,
        identity_id: int,
        *,
        display_name: str | None = None,
        category: IdentityCategory | None = None,
        retention_days: int | None = None,
        notes: str | None = None,
    ) -> Identity:
        identity = self.identities.get(identity_id)
        if identity is None:
            raise NotFoundError(f"identity {identity_id} not found")

        if display_name is not None:
            self.identities.rename(identity_id, display_name)
        if category is not None:
            self.identities.set_category(
                identity_id, category,
                retention_days=self._resolve_retention(category, retention_days),
            )
        elif retention_days is not None and identity.category is IdentityCategory.TEMPORARY:
            self.identities.set_category(
                identity_id, IdentityCategory.TEMPORARY,
                retention_days=self._resolve_retention(
                    IdentityCategory.TEMPORARY, retention_days
                ),
            )
        if notes is not None:
            identity.notes = notes

        self.session.flush()
        self.session.refresh(identity)
        self.sync_index()
        log.info(IDENTITY_UPDATED, action="updated", identity_id=identity_id)
        return identity

    def delete_identity(self, identity_id: int, *, hard: bool = False) -> dict:
        """Remove an identity from recognition.

        Historical detections, faces and events are preserved; only the
        identity's ability to match future faces is removed.
        """
        identity = self.identities.get(identity_id)
        if identity is None:
            raise NotFoundError(f"identity {identity_id} not found")

        detached = self.tracks.clear_identity(identity_id)
        if hard:
            self.session.delete(identity)   # cascades to its embeddings
        else:
            self.identities.soft_delete(identity_id)
            for embedding in self.embeddings.list_for_identity(identity_id):
                self.session.delete(embedding)
        self.session.flush()
        self.sync_index()

        from app.services.surveillance import get_manager

        get_manager().invalidate_identity(identity_id)
        log.info(IDENTITY_UPDATED, action="deleted", identity_id=identity_id, hard=hard)
        return {"identity_id": identity_id, "deleted": True, "tracks_detached": detached}

    # ---------------------------------------------------------- expiration
    def expire_due(self, *, now: datetime | None = None) -> list[dict]:
        """Expire temporary identities whose retention has elapsed.

        History is untouched: only ACTIVE -> EXPIRED and removal from the
        recognition index.
        """
        now = now or datetime.now(timezone.utc)
        due = self.identities.due_for_expiry(now=now)
        expired: list[dict] = []

        for identity in due:
            self.identities.mark_expired(identity.id, when=now)
            label = identity.display_name or identity.generated_identifier
            # The event is anchored to the source the identity was first seen
            # on; identities enrolled from disk have none, so they are only
            # broadcast, not persisted against a source.
            if identity.source_id:
                EventEmitter(
                    self.session, EventContext(identity.source_id, str(identity.source_id))
                ).emit(
                    EventType.IDENTITY_EXPIRED,
                    when=now,
                    identity_id=identity.id,
                    label=label,
                    message=f"Temporary identity expired: {label}",
                )
            expired.append(
                {
                    "identity_id": identity.id,
                    "identifier": identity.generated_identifier,
                    "display_name": identity.display_name,
                    "expired_at": now.isoformat(),
                }
            )
            log.info(
                TEMPORARY_IDENTITY_EXPIRED,
                identity_id=identity.id,
                identifier=identity.generated_identifier,
            )

        if expired:
            self.session.flush()
            self.sync_index()
            from app.services.surveillance import get_manager

            manager = get_manager()
            for item in expired:
                manager.invalidate_identity(item["identity_id"])
            self.bus.emit("identities_expired", source_uid=None, identities=expired)
        return expired

    # ------------------------------------------------------------- queries
    def summary(self) -> dict:
        counts = self.identities.counts_by_category()
        return {
            **counts,
            "pending_review": self.faces.pending_review_count(),
            "index": self.index.stats(),
        }


# ------------------------------------------------------------------ helpers
def sync_recognition_index() -> dict:
    """Rebuild the index using a fresh session (startup and scheduler)."""
    with session_scope() as session:
        return IdentityService(session).sync_index()


def run_expiration_scan() -> list[dict]:
    with session_scope() as session:
        return IdentityService(session).expire_due()


def preview_identifier(when: datetime | None = None) -> str:
    return generate_identifier(when or datetime.now(timezone.utc))
