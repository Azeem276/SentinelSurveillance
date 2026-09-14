"""Identity creation, naming, classification, expiry and index synchronisation."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.core.exceptions import NotFoundError
from app.intelligence.face.index import FaceRecognitionIndex, IndexEntry, get_index
from app.models.enums import (
    FaceReviewStatus, IdentityCategory, IdentityStatus, RecognitionState,
)
from app.models.identity import Face
from app.repositories.identity_repository import (
    FaceEmbeddingRepository, FaceRepository, IdentityRepository, generate_identifier,
)
from app.services.identity_service import IdentityService


# --------------------------------------------------------------- identifier
class TestGeneratedIdentifier:
    def test_format_is_timestamp_derived(self):
        when = datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc)
        assert re.fullmatch(r"Unknown_\d{8}_\d{6}", generate_identifier(when))

    def test_is_stable_for_the_same_timestamp(self):
        when = datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc)
        assert generate_identifier(when) == generate_identifier(when)

    def test_differs_for_different_timestamps(self):
        a = generate_identifier(datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc))
        b = generate_identifier(datetime(2026, 9, 14, 2, 13, 16, tzinfo=timezone.utc))
        assert a != b

    def test_is_not_a_random_uuid(self):
        value = generate_identifier(datetime.now(timezone.utc))
        assert value.startswith("Unknown_")
        assert "-" not in value

    def test_collisions_get_a_suffix(self, session):
        repo = IdentityRepository(session)
        when = datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc)
        first = repo.create(category=IdentityCategory.PERMANENT, first_detected_at=when)
        second = repo.create(category=IdentityCategory.PERMANENT, first_detected_at=when)
        assert first.generated_identifier != second.generated_identifier


# ------------------------------------------------------------------ create
class TestIdentityCreation:
    def test_named_permanent(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.PERMANENT, display_name="Azeem"
        )
        assert identity.display_name == "Azeem"
        assert identity.expires_at is None
        assert identity.label == "Azeem"

    def test_unnamed_permanent_falls_back_to_the_identifier(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.PERMANENT, display_name=None
        )
        assert identity.display_name is None
        assert identity.label == identity.generated_identifier

    def test_blank_name_is_treated_as_unnamed(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.PERMANENT, display_name="   "
        )
        assert identity.display_name is None

    def test_named_temporary_gets_an_expiry(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.TEMPORARY, display_name="John", retention_days=3
        )
        assert identity.expires_at is not None
        assert identity.retention_days == 3
        delta = identity.expires_at - datetime.now(timezone.utc)
        assert 2.9 < delta.total_seconds() / 86400 < 3.1

    def test_temporary_defaults_to_seven_days(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.TEMPORARY
        )
        assert identity.retention_days == 7

    @pytest.mark.parametrize("days", [1, 3, 7, 30])
    def test_supported_retention_presets(self, session, days):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.TEMPORARY, retention_days=days
        )
        assert identity.retention_days == days

    def test_custom_retention_is_allowed(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.TEMPORARY, retention_days=45
        )
        assert identity.retention_days == 45

    def test_permanent_never_gets_a_retention(self, session):
        identity = IdentityService(session).create_identity(
            category=IdentityCategory.PERMANENT, retention_days=7
        )
        assert identity.retention_days is None
        assert identity.expires_at is None


# -------------------------------------------------------------- expiration
class TestExpiration:
    def _expired_identity(self, session, name=None):
        repo = IdentityRepository(session)
        identity = repo.create(
            category=IdentityCategory.TEMPORARY,
            first_detected_at=datetime.now(timezone.utc) - timedelta(days=10),
            display_name=name,
            retention_days=1,
        )
        identity.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        session.flush()
        return identity

    def test_due_identities_are_found(self, session):
        self._expired_identity(session)
        assert len(IdentityRepository(session).due_for_expiry()) == 1

    def test_unexpired_identities_are_not_due(self, session):
        IdentityService(session).create_identity(
            category=IdentityCategory.TEMPORARY, retention_days=7
        )
        assert IdentityRepository(session).due_for_expiry() == []

    def test_permanent_identities_are_never_due(self, session):
        IdentityService(session).create_identity(category=IdentityCategory.PERMANENT)
        assert IdentityRepository(session).due_for_expiry() == []

    def test_expiry_marks_the_identity_expired(self, session):
        identity = self._expired_identity(session, "Courier")
        expired = IdentityService(session).expire_due()
        session.refresh(identity)
        assert len(expired) == 1
        assert identity.status is IdentityStatus.EXPIRED

    def test_expiry_is_enforced_by_the_backend_not_a_timer(self, session):
        """expire_due is a pure DB query against the clock."""
        self._expired_identity(session)
        assert len(IdentityService(session).expire_due()) == 1
        # Idempotent: a second sweep finds nothing.
        assert IdentityService(session).expire_due() == []

    def test_history_survives_expiry(self, session, make_source):
        source = make_source()
        identity = self._expired_identity(session, "Courier")
        face = Face(
            source_id=source.id, identity_id=identity.id,
            detected_at=datetime.now(timezone.utc) - timedelta(days=2),
            bbox_x1=0, bbox_y1=0, bbox_x2=10, bbox_y2=10,
            recognition_state=RecognitionState.TEMPORARY_FAMILIAR,
        )
        session.add(face)
        session.flush()

        IdentityService(session).expire_due()
        session.refresh(face)
        # The face record and its identity link are untouched.
        assert FaceRepository(session).get(face.id) is not None
        assert face.identity_id == identity.id

    def test_expired_identity_leaves_the_recognition_index(self, session, unit_vector):
        """Live -> expired must remove the identity from matching."""
        service = IdentityService(session)
        identity = service.create_identity(
            category=IdentityCategory.TEMPORARY, display_name="Courier",
            retention_days=7,
        )
        vec = unit_vector(1)
        FaceEmbeddingRepository(session).add_embedding(
            identity_id=identity.id, vector=vec, model_name="sface"
        )
        service.sync_index()
        assert get_index().identity_count == 1
        assert get_index().recognize(vec).identity_id == identity.id

        # Retention elapses.
        identity.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.flush()

        assert len(service.expire_due()) == 1
        assert get_index().identity_count == 0
        assert get_index().recognize(vec).state is RecognitionState.UNFAMILIAR


# ------------------------------------------------------------ index syncing
class TestRecognitionIndex:
    def test_empty_index_reports_unfamiliar(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        result = index.recognize(unit_vector(1))
        assert result.state is RecognitionState.UNFAMILIAR
        assert result.identity_id is None

    def test_exact_match_is_recognised(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        vec = unit_vector(1)
        index.rebuild([(IndexEntry(1, "Azeem", IdentityCategory.PERMANENT, None), vec)])
        result = index.recognize(vec)
        assert result.state is RecognitionState.PERMANENT_FAMILIAR
        assert result.identity_id == 1
        assert result.score == pytest.approx(1.0, abs=1e-4)

    def test_dissimilar_face_is_unfamiliar(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        index.rebuild(
            [(IndexEntry(1, "Azeem", IdentityCategory.PERMANENT, None), unit_vector(1))]
        )
        assert index.recognize(unit_vector(999)).state is RecognitionState.UNFAMILIAR

    def test_threshold_is_respected(self, unit_vector):
        base = unit_vector(1)
        nudged = base + np.random.default_rng(5).normal(0, 0.35, base.shape)
        nudged = (nudged / np.linalg.norm(nudged)).astype(np.float32)
        entry = IndexEntry(1, "Azeem", IdentityCategory.PERMANENT, None)

        lenient = FaceRecognitionIndex(threshold=0.2)
        lenient.rebuild([(entry, base)])
        strict = FaceRecognitionIndex(threshold=0.99)
        strict.rebuild([(entry, base)])

        assert lenient.recognize(nudged).state is RecognitionState.PERMANENT_FAMILIAR
        assert strict.recognize(nudged).state is RecognitionState.UNFAMILIAR

    def test_temporary_identity_yields_the_temporary_state(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        future = datetime.now(timezone.utc) + timedelta(days=1)
        vec = unit_vector(2)
        index.rebuild(
            [(IndexEntry(7, "John", IdentityCategory.TEMPORARY, future), vec)]
        )
        assert index.recognize(vec).state is RecognitionState.TEMPORARY_FAMILIAR

    def test_expired_temporary_is_filtered_at_build_time(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        past = datetime.now(timezone.utc) - timedelta(days=1)
        vec = unit_vector(2)
        index.rebuild([(IndexEntry(7, "John", IdentityCategory.TEMPORARY, past), vec)])
        assert index.size == 0
        assert index.recognize(vec).state is RecognitionState.UNFAMILIAR

    def test_many_embeddings_per_identity_and_best_wins(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        entry = IndexEntry(1, "Azeem", IdentityCategory.PERMANENT, None)
        vectors = [unit_vector(i) for i in (1, 2, 3)]
        index.rebuild([(entry, v) for v in vectors])
        assert index.size == 3
        assert index.identity_count == 1
        result = index.recognize(vectors[2])
        assert result.identity_id == 1
        assert result.score == pytest.approx(1.0, abs=1e-4)

    def test_rebuilding_replaces_the_dataset(self, unit_vector):
        index = FaceRecognitionIndex(threshold=0.5)
        vec = unit_vector(1)
        index.rebuild([(IndexEntry(1, "Azeem", IdentityCategory.PERMANENT, None), vec)])
        assert index.recognize(vec).identity_id == 1
        index.rebuild([])
        assert index.recognize(vec).state is RecognitionState.UNFAMILIAR
        assert index.version == 2

    def test_service_sync_loads_from_the_database(self, session, unit_vector):
        service = IdentityService(session)
        identity = service.create_identity(
            category=IdentityCategory.PERMANENT, display_name="Azeem"
        )
        vec = unit_vector(4)
        FaceEmbeddingRepository(session).add_embedding(
            identity_id=identity.id, vector=vec, model_name="sface"
        )
        stats = service.sync_index()
        assert stats["identities"] == 1
        assert get_index().recognize(vec).identity_id == identity.id

    def test_deleting_an_identity_removes_it_from_recognition(self, session, unit_vector):
        service = IdentityService(session)
        identity = service.create_identity(
            category=IdentityCategory.PERMANENT, display_name="Azeem"
        )
        vec = unit_vector(4)
        FaceEmbeddingRepository(session).add_embedding(
            identity_id=identity.id, vector=vec, model_name="sface"
        )
        service.sync_index()
        assert get_index().recognize(vec).identity_id == identity.id

        service.delete_identity(identity.id)
        assert get_index().recognize(vec).state is RecognitionState.UNFAMILIAR


# ----------------------------------------------------------- classification
@pytest.fixture()
def pending_face(session, make_source):
    source = make_source()
    face = Face(
        source_id=source.id,
        detected_at=datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc),
        frame_number=42,
        bbox_x1=10, bbox_y1=10, bbox_x2=70, bbox_y2=80,
        detection_confidence=0.9,
        quality_score=0.8,
        quality_ok=True,
        recognition_state=RecognitionState.UNFAMILIAR,
        review_status=FaceReviewStatus.PENDING,
    )
    session.add(face)
    session.flush()
    return face


class TestClassification:
    def test_named_permanent_classification(self, session, pending_face):
        result = IdentityService(session).classify_face(
            pending_face.id, category=IdentityCategory.PERMANENT, display_name="Mike"
        )
        assert result.created
        assert result.display_name == "Mike"
        assert result.category == "PERMANENT"
        assert result.expires_at is None

    def test_unnamed_classification_uses_the_detection_timestamp(
        self, session, pending_face
    ):
        result = IdentityService(session).classify_face(
            pending_face.id, category=IdentityCategory.TEMPORARY, display_name=None,
            retention_days=7,
        )
        assert result.display_name is None
        assert result.identifier == generate_identifier(pending_face.detected_at)

    def test_classification_marks_the_face_reviewed(self, session, pending_face):
        IdentityService(session).classify_face(
            pending_face.id, category=IdentityCategory.PERMANENT, display_name="Mike"
        )
        session.refresh(pending_face)
        assert pending_face.review_status is FaceReviewStatus.CLASSIFIED
        assert pending_face.recognition_state is RecognitionState.PERMANENT_FAMILIAR
        assert pending_face.identity_id is not None

    def test_classifying_into_an_existing_identity(self, session, pending_face):
        service = IdentityService(session)
        existing = service.create_identity(
            category=IdentityCategory.PERMANENT, display_name="Azeem"
        )
        result = service.classify_face(
            pending_face.id, category=IdentityCategory.PERMANENT,
            identity_id=existing.id,
        )
        assert not result.created
        assert result.identity_id == existing.id

    def test_unknown_face_id_raises(self, session):
        with pytest.raises(NotFoundError):
            IdentityService(session).classify_face(
                999999, category=IdentityCategory.PERMANENT
            )

    def test_pending_review_count_tracks_classification(self, session, pending_face):
        repo = FaceRepository(session)
        assert repo.pending_review_count() == 1
        IdentityService(session).classify_face(
            pending_face.id, category=IdentityCategory.PERMANENT, display_name="Mike"
        )
        assert repo.pending_review_count() == 0

    def test_dismissing_a_face_clears_it_from_review(self, session, pending_face):
        repo = FaceRepository(session)
        repo.dismiss(pending_face.id)
        assert repo.pending_review_count() == 0


class TestBulkClassification:
    @pytest.fixture()
    def faces(self, session, make_source):
        source = make_source()
        out = []
        for i in range(3):
            face = Face(
                source_id=source.id,
                detected_at=datetime(2026, 9, 14, 2, 13, 15 + i, tzinfo=timezone.utc),
                bbox_x1=0, bbox_y1=0, bbox_x2=50, bbox_y2=60,
                detection_confidence=0.9, quality_score=0.8, quality_ok=True,
                recognition_state=RecognitionState.UNFAMILIAR,
                review_status=FaceReviewStatus.PENDING,
            )
            session.add(face)
            out.append(face)
        session.flush()
        return out

    def test_each_face_becomes_its_own_identity_by_default(self, session, faces):
        results = IdentityService(session).bulk_classify(
            [f.id for f in faces], category=IdentityCategory.TEMPORARY,
            retention_days=7,
        )
        assert len(results) == 3
        assert len({r.identity_id for r in results}) == 3, "distinct people must not merge"

    def test_unnamed_bulk_gets_per_detection_identifiers(self, session, faces):
        results = IdentityService(session).bulk_classify(
            [f.id for f in faces], category=IdentityCategory.TEMPORARY
        )
        identifiers = {r.identifier for r in results}
        assert len(identifiers) == 3

    def test_shared_name_is_applied_to_all(self, session, faces):
        results = IdentityService(session).bulk_classify(
            [f.id for f in faces], category=IdentityCategory.PERMANENT,
            display_name="Night Shift",
        )
        assert all(r.display_name == "Night Shift" for r in results)

    def test_explicit_merge_produces_one_identity(self, session, faces):
        results = IdentityService(session).bulk_classify(
            [f.id for f in faces], category=IdentityCategory.PERMANENT,
            display_name="Mike", merge_into_one_identity=True,
        )
        assert len({r.identity_id for r in results}) == 1

    def test_empty_selection_is_rejected(self, session):
        from app.core.exceptions import ValidationError

        with pytest.raises(ValidationError):
            IdentityService(session).bulk_classify([], category=IdentityCategory.PERMANENT)


class TestIdentityUpdates:
    def test_renaming_works(self, session):
        service = IdentityService(session)
        identity = service.create_identity(category=IdentityCategory.PERMANENT)
        service.update_identity(identity.id, display_name="Renamed")
        assert identity.display_name == "Renamed"

    def test_promoting_temporary_to_permanent_clears_the_expiry(self, session):
        service = IdentityService(session)
        identity = service.create_identity(
            category=IdentityCategory.TEMPORARY, retention_days=7
        )
        assert identity.expires_at is not None
        service.update_identity(identity.id, category=IdentityCategory.PERMANENT)
        session.refresh(identity)
        assert identity.expires_at is None
        assert identity.category is IdentityCategory.PERMANENT

    def test_demoting_permanent_to_temporary_sets_an_expiry(self, session):
        service = IdentityService(session)
        identity = service.create_identity(category=IdentityCategory.PERMANENT)
        service.update_identity(
            identity.id, category=IdentityCategory.TEMPORARY, retention_days=3
        )
        session.refresh(identity)
        assert identity.expires_at is not None
        assert identity.retention_days == 3

    def test_deleting_detaches_tracks_but_keeps_them(self, session):
        service = IdentityService(session)
        identity = service.create_identity(category=IdentityCategory.PERMANENT)
        result = service.delete_identity(identity.id)
        assert result["deleted"]
        session.refresh(identity)
        assert identity.status is IdentityStatus.DELETED
