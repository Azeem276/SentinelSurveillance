"""Analysis panel aggregation: objects, identities, events and timeline."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models.detection import Track
from app.models.enums import (
    AlertState, EventType, IdentityCategory, RecognitionState,
)
from app.models.event import Alert, SecurityEvent
from app.models.identity import Identity
from app.models.source import VideoSource
from app.repositories.detection_repository import MotionRepository, TrackRepository
from app.repositories.event_repository import EventRepository
from app.repositories.identity_repository import FaceRepository
from app.schemas.intelligence import (
    IdentityAppearance, ObjectCount, SourceAnalysis, TimelineEntry,
)

RECOGNITION_EVENT_TYPES = (
    EventType.FACE_RECOGNIZED,
    EventType.UNKNOWN_FACE,
    EventType.FACE_DETECTED,
    EventType.FACE_UNRECOGNIZABLE,
)


def build_timeline(
    session: Session, source_id: int, *, since: datetime, limit: int
) -> list[TimelineEntry]:
    """Chronological event feed, enriched with identity and object labels."""
    rows = session.execute(
        select(SecurityEvent, Identity, Track)
        .outerjoin(Identity, SecurityEvent.identity_id == Identity.id)
        .outerjoin(Track, SecurityEvent.track_id == Track.id)
        .where(SecurityEvent.source_id == source_id, SecurityEvent.started_at >= since)
        .order_by(desc(SecurityEvent.started_at))
        .limit(limit)
    ).all()

    entries: list[TimelineEntry] = []
    for event, identity, track in rows:
        entries.append(
            TimelineEntry(
                id=event.id,
                timestamp=event.started_at,
                event_type=str(getattr(event.event_type, "value", event.event_type)),
                severity=str(getattr(event.severity, "value", event.severity)),
                label=event.label,
                message=event.message,
                duration_seconds=event.duration_seconds,
                track_id=event.track_id,
                identity_id=event.identity_id,
                identity_label=(
                    (identity.display_name or identity.generated_identifier)
                    if identity else None
                ),
                recording_id=event.recording_id,
                object_class=track.object_class if track else None,
                is_open=bool(event.is_open),
            )
        )
    return entries


def build_source_analysis(
    session: Session,
    source: VideoSource,
    *,
    hours: int = 24,
    timeline_limit: int = 60,
    runtime: dict[str, Any] | None = None,
) -> SourceAnalysis:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    tracks = TrackRepository(session)
    events = EventRepository(session)

    class_counts = tracks.class_counts(source.id, since=since)
    recognition_counts = tracks.recognition_counts(source.id, since=since)

    # Distinct identities seen in the window, with appearance counts.
    identity_rows = session.execute(
        select(
            Identity.id,
            Identity.display_name,
            Identity.generated_identifier,
            Identity.category,
            func.count(Track.id),
            func.max(Track.last_seen_at),
        )
        .join(Track, Track.identity_id == Identity.id)
        .where(Track.source_id == source.id, Track.first_seen_at >= since)
        .group_by(
            Identity.id, Identity.display_name, Identity.generated_identifier,
            Identity.category,
        )
        .order_by(desc(func.max(Track.last_seen_at)))
    ).all()

    identities = [
        IdentityAppearance(
            identity_id=int(ident_id),
            label=name or identifier,
            category=str(getattr(category, "value", category)),
            recognition_state=(
                RecognitionState.PERMANENT_FAMILIAR.value
                if str(getattr(category, "value", category)) == IdentityCategory.PERMANENT.value
                else RecognitionState.TEMPORARY_FAMILIAR.value
            ),
            appearances=int(count),
            last_seen_at=last_seen,
        )
        for ident_id, name, identifier, category, count, last_seen in identity_rows
    ]

    # Unidentified people are summarised by recognition state, not merged.
    unfamiliar = recognition_counts.get(RecognitionState.UNFAMILIAR.value, 0)
    unrecognizable = recognition_counts.get(RecognitionState.FACE_UNRECOGNIZABLE.value, 0)
    if unfamiliar:
        identities.append(
            IdentityAppearance(
                identity_id=None,
                label="UNFAMILIAR",
                category=None,
                recognition_state=RecognitionState.UNFAMILIAR.value,
                appearances=unfamiliar,
                last_seen_at=None,
            )
        )
    if unrecognizable:
        identities.append(
            IdentityAppearance(
                identity_id=None,
                label="FACE UNRECOGNIZABLE",
                category=None,
                recognition_state=RecognitionState.FACE_UNRECOGNIZABLE.value,
                appearances=unrecognizable,
                last_seen_at=None,
            )
        )

    event_counts = events.count_by_type(source.id, since=since)
    motion_count = len(
        MotionRepository(session).list_for_source(source.id, limit=1000, since=since)
    )
    active_alerts = int(
        session.execute(
            select(func.count(Alert.id)).where(
                Alert.source_id == source.id, Alert.state == AlertState.ACTIVE
            )
        ).scalar()
        or 0
    )

    permanent_count = sum(
        1 for i in identities
        if i.category == IdentityCategory.PERMANENT.value
    )
    temporary_count = sum(
        1 for i in identities
        if i.category == IdentityCategory.TEMPORARY.value
    )

    objects = [
        ObjectCount(object_class=cls, count=count)
        for cls, count in sorted(class_counts.items(), key=lambda kv: -kv[1])
    ]

    return SourceAnalysis(
        source_id=source.id,
        source_uid=source.uid,
        source_name=source.name,
        window_hours=hours,
        objects=objects,
        total_objects=sum(class_counts.values()),
        people=class_counts.get("person", 0),
        identities=identities,
        permanent_count=permanent_count,
        temporary_count=temporary_count,
        unfamiliar_count=unfamiliar,
        unrecognizable_count=unrecognizable,
        motion_events=motion_count,
        recognition_events=sum(
            event_counts.get(t.value, 0) for t in RECOGNITION_EVENT_TYPES
        ),
        active_alerts=active_alerts,
        event_counts=event_counts,
        timeline=build_timeline(session, source.id, since=since, limit=timeline_limit),
        pending_review=FaceRepository(session).pending_review_count(),
        runtime=runtime or {},
    )
