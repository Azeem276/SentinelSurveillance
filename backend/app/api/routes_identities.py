"""Identity management, unfamiliar-face review and classification."""
from __future__ import annotations

from fastapi import APIRouter, Query, Response, status

from app.api.deps import DbSession, PaginationDep
from app.core.exceptions import NotFoundError, StorageError
from app.models.enums import FaceReviewStatus, IdentityCategory, IdentityStatus
from app.repositories.detection_repository import TrackRepository
from app.repositories.identity_repository import (
    FaceEmbeddingRepository, FaceRepository, IdentityRepository, generate_identifier,
)
from app.repositories.source_repository import SourceRepository
from app.schemas.common import OperationResult
from app.schemas.intelligence import (
    BulkClassifyRequest, ClassificationResponse, FaceClassifyRequest, IdentityCreate,
    IdentityDetail, IdentityRead, IdentityUpdate, TrackRead, UnfamiliarFace,
)
from app.services.identity_service import IdentityService
from app.storage.paths import face_root, resolve_face

router = APIRouter(prefix="/api", tags=["identities"])


def _detail(session, identity) -> IdentityDetail:
    detail = IdentityDetail.model_validate(identity)
    detail.embedding_count = FaceEmbeddingRepository(session).count_for_identity(identity.id)
    detail.track_count = len(TrackRepository(session).list_for_identity(identity.id, limit=500))
    detail.label = identity.display_name or identity.generated_identifier
    detail.thumbnail_url = (
        f"/api/faces/image/{identity.thumbnail_path}" if identity.thumbnail_path else None
    )
    return detail


# ------------------------------------------------------------- identities
@router.get("/identities", response_model=list[IdentityDetail])
def list_identities(
    session: DbSession,
    page: PaginationDep,
    category: IdentityCategory | None = None,
    status_filter: IdentityStatus | None = Query(IdentityStatus.ACTIVE, alias="status"),
    search: str | None = None,
):
    identities = IdentityRepository(session).list_identities(
        category=category,
        status=status_filter,
        search=search,
        limit=page.limit,
        offset=page.offset,
    )
    return [_detail(session, i) for i in identities]


@router.post("/identities", response_model=IdentityDetail,
             status_code=status.HTTP_201_CREATED)
def create_identity(payload: IdentityCreate, session: DbSession):
    service = IdentityService(session)
    identity = service.create_identity(
        category=payload.category,
        display_name=payload.display_name,
        first_detected_at=payload.first_detected_at,
        retention_days=payload.retention_days,
        source_id=payload.source_id,
        notes=payload.notes,
    )
    session.commit()
    session.refresh(identity)
    return _detail(session, identity)


@router.get("/identities/{identity_id}", response_model=IdentityDetail)
def get_identity(identity_id: int, session: DbSession):
    identity = IdentityRepository(session).get(identity_id)
    if identity is None:
        raise NotFoundError(f"identity {identity_id} not found")
    return _detail(session, identity)


@router.patch("/identities/{identity_id}", response_model=IdentityDetail)
def update_identity(identity_id: int, payload: IdentityUpdate, session: DbSession):
    service = IdentityService(session)
    identity = service.update_identity(
        identity_id,
        display_name=payload.display_name,
        category=payload.category,
        retention_days=payload.retention_days,
        notes=payload.notes,
    )
    session.commit()
    session.refresh(identity)
    return _detail(session, identity)


@router.delete("/identities/{identity_id}", response_model=OperationResult)
def delete_identity(identity_id: int, session: DbSession, hard: bool = False):
    """Remove an identity from recognition. History is preserved."""
    result = IdentityService(session).delete_identity(identity_id, hard=hard)
    session.commit()
    return OperationResult(data=result)


@router.get("/identities/{identity_id}/tracks", response_model=list[TrackRead])
def identity_tracks(identity_id: int, session: DbSession, limit: int = Query(100, le=500)):
    return TrackRepository(session).list_for_identity(identity_id, limit=limit)


@router.post("/identities/enrol-dataset", response_model=OperationResult)
def enrol_dataset(session: DbSession):
    """Enrol data/familiar_faces/{permanent,temporary}/<Name>/*.jpg."""
    result = IdentityService(session).enrol_from_directory()
    session.commit()
    return OperationResult(data=result)


@router.post("/identities/sync-index", response_model=OperationResult)
def sync_index(session: DbSession):
    return OperationResult(data=IdentityService(session).sync_index())


@router.post("/identities/expire-now", response_model=OperationResult)
def expire_now(session: DbSession):
    """Force an expiration sweep (the scheduler also does this periodically)."""
    expired = IdentityService(session).expire_due()
    session.commit()
    return OperationResult(data={"expired": expired, "count": len(expired)})


# ------------------------------------------------------------------ faces
@router.get("/faces/unfamiliar", response_model=list[UnfamiliarFace])
def list_unfamiliar_faces(
    session: DbSession,
    page: PaginationDep,
    source_id: int | None = None,
    review_status: FaceReviewStatus | None = FaceReviewStatus.PENDING,
):
    faces = FaceRepository(session).list_unfamiliar(
        review_status=review_status,
        source_id=source_id,
        limit=page.limit,
        offset=page.offset,
    )
    source_repo = SourceRepository(session)
    cache: dict[int, tuple[str, str]] = {}
    out: list[UnfamiliarFace] = []
    for face in faces:
        if face.source_id not in cache:
            source = source_repo.get(face.source_id)
            cache[face.source_id] = (
                (source.uid, source.name) if source else ("unknown", "Unknown source")
            )
        uid, name = cache[face.source_id]
        item = UnfamiliarFace.model_validate(face)
        item.source_uid = uid
        item.source_name = name
        item.image_url = f"/api/faces/image/{face.image_path}" if face.image_path else None
        item.suggested_identifier = generate_identifier(face.detected_at)
        out.append(item)
    return out


@router.get("/faces/pending-count", response_model=OperationResult)
def pending_review_count(session: DbSession):
    return OperationResult(data={"pending": FaceRepository(session).pending_review_count()})


@router.post("/faces/{face_id}/classify", response_model=ClassificationResponse)
def classify_face(face_id: int, payload: FaceClassifyRequest, session: DbSession):
    """Classify one unfamiliar face as permanent or temporary familiar.

    Leaving ``display_name`` blank is supported: the identity then uses the
    timestamp-derived identifier from its first detection.
    """
    result = IdentityService(session).classify_face(
        face_id,
        category=payload.category,
        display_name=payload.display_name,
        retention_days=payload.retention_days,
        identity_id=payload.identity_id,
    )
    session.commit()
    return ClassificationResponse(**result.__dict__)


@router.post("/faces/classify-bulk", response_model=list[ClassificationResponse])
def bulk_classify(payload: BulkClassifyRequest, session: DbSession):
    results = IdentityService(session).bulk_classify(
        payload.face_ids,
        category=payload.category,
        display_name=payload.display_name,
        retention_days=payload.retention_days,
        merge_into_one_identity=payload.merge_into_one_identity,
    )
    session.commit()
    return [ClassificationResponse(**r.__dict__) for r in results]


@router.post("/faces/{face_id}/dismiss", response_model=OperationResult)
def dismiss_face(face_id: int, session: DbSession):
    repo = FaceRepository(session)
    if repo.get(face_id) is None:
        raise NotFoundError(f"face {face_id} not found")
    repo.dismiss(face_id)
    session.commit()
    return OperationResult(data={"face_id": face_id, "dismissed": True})


@router.get("/faces/image/{path:path}")
def face_image(path: str):
    """Serve a stored face crop.

    The path is resolved strictly inside FACE_STORAGE_PATH; traversal attempts
    raise a storage error rather than reading arbitrary files.
    """
    resolved = resolve_face(path)
    if not resolved.exists():
        raise NotFoundError("face image not found")
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise StorageError(f"could not read face image: {exc}") from exc
    media = "image/png" if resolved.suffix.lower() == ".png" else "image/jpeg"
    return Response(content=data, media_type=media,
                    headers={"Cache-Control": "private, max-age=3600"})
