#!/usr/bin/env python
"""Seed a fully populated demo dataset.

This exists so the entire UI can be exercised - sources, grid, analysis
panel, identities, unfamiliar-face review, bulk classification, alarms,
event timeline and archive - without waiting for hours of real footage.

It writes ONLY through the real repositories and models, so what you see in
the UI is the same shape of data the live pipeline produces. It seeds the
acceptance scenario from the brief:

    Azeem  -> PERMANENT familiar   (no alarm at Proximity B)
    John   -> TEMPORARY familiar   (beep on recognition)
    Unknown person -> UNFAMILIAR   (continuous alarm at Proximity B)
    Dog, Car -> detected and tracked, no face processing

    python scripts/seed_demo_data.py            # add demo data
    python scripts/seed_demo_data.py --reset    # wipe demo data first
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from sqlalchemy import delete, text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.models import (  # noqa: E402
    Alert, Detection, Face, FaceEmbedding, Identity, MotionEvent,
    RecordingSession, SecurityEvent, Track, VideoSource,
)
from app.models.enums import (  # noqa: E402
    AlertState, AlertType, EventSeverity, EventType, FaceReviewStatus,
    IdentityCategory, IdentityStatus, ProximityZone, RecognitionState,
    RecordingStatus, SourceStatus, SourceType, TrackStatus,
)
from app.repositories.identity_repository import generate_identifier  # noqa: E402
from app.storage.paths import ensure_parent, face_root, relative_to_root  # noqa: E402

rng = random.Random(20260914)
NOW = datetime.now(timezone.utc)

SOURCES = [
    dict(uid="camera_01", name="Front Door", uri="camera_01.mp4", location="Main entrance",
         proximity_a=10.0, proximity_b=3.0, fov=55.0),
    dict(uid="camera_02", name="Back Gate", uri="camera_02.mp4", location="Rear perimeter",
         proximity_a=15.0, proximity_b=5.0, fov=62.0),
    dict(uid="camera_03", name="Driveway", uri="camera_03.mp4", location="North driveway",
         proximity_a=12.0, proximity_b=4.0, fov=58.0),
]


# ----------------------------------------------------------------- helpers
def synthetic_face(seed: int, label: str) -> np.ndarray:
    """A distinct, deterministic placeholder face crop for the review UI.

    Clearly synthetic on purpose: this is demo imagery for exercising the
    review workflow, never presented as a real biometric sample.
    """
    r = random.Random(seed)
    size = 160
    skin = (r.randint(150, 205), r.randint(140, 185), r.randint(135, 175))
    img = np.full((size, size, 3), (38, 40, 46), dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2 + 6), (46, 60), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (size // 2, size // 2 - 46), (46, 30), 0, 180, 360,
                (r.randint(30, 90),) * 3, -1)
    for dx in (-18, 18):
        cv2.circle(img, (size // 2 + dx, size // 2 - 10), 7, (250, 250, 250), -1)
        cv2.circle(img, (size // 2 + dx, size // 2 - 10), 3,
                   (r.randint(40, 120), r.randint(40, 110), r.randint(30, 90)), -1)
    cv2.line(img, (size // 2, size // 2 - 4), (size // 2, size // 2 + 14),
             (max(0, skin[0] - 40),) * 3, 2)
    cv2.ellipse(img, (size // 2, size // 2 + 30), (16, 7), 0, 0, 180,
                (90, 60, 60), 2)
    cv2.putText(img, label[:10], (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)
    return img


def write_face_crop(subdir: str, name: str, image: np.ndarray) -> str:
    path = face_root() / subdir / name
    ensure_parent(path)
    cv2.imwrite(str(path), image)
    return relative_to_root(path, face_root())


def embedding_for(seed: int, dim: int = 128) -> np.ndarray:
    """A stable unit vector standing in for a real SFace descriptor."""
    vec = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vec / float(np.linalg.norm(vec))


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def bbox_for_distance(distance_m: float, frame_h: int = 540, fov_deg: float = 55.0,
                      person_h: float = 1.7) -> tuple[float, float, float, float]:
    """Invert the proximity model so seeded boxes match their stated distance."""
    focal = (frame_h / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    h = person_h * focal / max(0.5, distance_m)
    w = h * 0.38
    cx = rng.uniform(0.3, 0.7) * 960
    y2 = min(540.0, 300.0 + h * 0.7)
    return cx - w / 2, y2 - h, cx + w / 2, y2


# -------------------------------------------------------------------- reset
def reset(session) -> None:
    print("Removing existing demo data...")
    for model in (Alert, SecurityEvent, Detection, Face, FaceEmbedding, MotionEvent,
                  Track, RecordingSession, Identity, VideoSource):
        session.execute(delete(model))
    session.flush()


# --------------------------------------------------------------------- seed
def seed_sources(session) -> dict[str, VideoSource]:
    existing = {s.uid: s for s in session.query(VideoSource).all()}
    out: dict[str, VideoSource] = {}
    for order, spec in enumerate(SOURCES, start=1):
        if spec["uid"] in existing:
            out[spec["uid"]] = existing[spec["uid"]]
            continue
        source = VideoSource(
            uid=spec["uid"],
            name=spec["name"],
            type=SourceType.FILE,
            uri=spec["uri"],
            location=spec["location"],
            description=f"Demo source backed by {spec['uri']}",
            enabled=True,
            surveillance_enabled=True,
            intelligence_enabled=spec["uid"] != "camera_03",
            recognition_enabled=True,
            recording_enabled=True,
            loop_playback=True,
            status=SourceStatus.IDLE,
            proximity_a=spec["proximity_a"],
            proximity_b=spec["proximity_b"],
            alert_policy={"temporary_familiar": "beep", "unrecognizable_policy": "ignore",
                          "permanent_familiar": "none", "beep_on_recognition": True},
            calibration={"vertical_fov_deg": spec["fov"], "reference_height_m": 1.7,
                         "distance_scale": 1.0, "visible_height_fraction": 1.0},
            display_order=order,
        )
        session.add(source)
        out[spec["uid"]] = source
    session.flush()
    print(f"  sources:    {len(out)}")
    return out


def seed_identities(session) -> dict[str, Identity]:
    identities: dict[str, Identity] = {}
    specs = [
        ("Azeem", IdentityCategory.PERMANENT, None, 101),
        ("Sarah Khan", IdentityCategory.PERMANENT, None, 102),
        ("John Smith", IdentityCategory.TEMPORARY, 7, 103),
        ("Delivery Courier", IdentityCategory.TEMPORARY, 1, 104),
    ]
    for name, category, retention, seed in specs:
        found = session.query(Identity).filter(Identity.display_name == name).first()
        if found:
            identities[name] = found
            continue
        first_seen = NOW - timedelta(days=rng.randint(3, 30))
        identity = Identity(
            generated_identifier=generate_identifier(first_seen),
            display_name=name,
            category=category,
            status=IdentityStatus.ACTIVE,
            first_detected_at=first_seen,
            last_seen_at=NOW - timedelta(minutes=rng.randint(5, 300)),
            expires_at=(NOW + timedelta(days=retention)) if retention else None,
            retention_days=retention,
            notes="Seeded demo identity",
            thumbnail_path=write_face_crop(
                "permanent" if category is IdentityCategory.PERMANENT else "temporary",
                f"{name.replace(' ', '_').lower()}.jpg",
                synthetic_face(seed, name.split()[0]),
            ),
            extra={"demo": True},
        )
        session.add(identity)
        session.flush()
        # Several embeddings per identity, as real enrolment produces.
        for k in range(3):
            base = embedding_for(seed)
            jitter = np.random.default_rng(seed + k).normal(0, 0.04, base.shape)
            vec = (base + jitter).astype(np.float32)
            vec /= float(np.linalg.norm(vec))
            session.add(
                FaceEmbedding(
                    identity_id=identity.id, vector=to_bytes(vec), dim=128,
                    model_name="sface", origin="enrollment",
                    created_at=first_seen + timedelta(minutes=k),
                )
            )
        identities[name] = identity

    # An already-expired temporary identity, to prove history survives expiry.
    if not session.query(Identity).filter(
        Identity.status == IdentityStatus.EXPIRED
    ).first():
        expired_first = NOW - timedelta(days=12)
        expired = Identity(
            generated_identifier=generate_identifier(expired_first),
            display_name=None,
            category=IdentityCategory.TEMPORARY,
            status=IdentityStatus.EXPIRED,
            first_detected_at=expired_first,
            last_seen_at=NOW - timedelta(days=6),
            expires_at=NOW - timedelta(days=1),
            retention_days=7,
            notes="Expired automatically by the retention scheduler",
            extra={"demo": True},
        )
        session.add(expired)
        identities["_expired"] = expired
    session.flush()
    print(f"  identities: {len(identities)}")
    return identities


def seed_activity(session, sources: dict[str, VideoSource],
                  identities: dict[str, Identity]) -> None:
    total_events = total_tracks = total_alerts = 0

    for source in sources.values():
        fov = float(source.calibration.get("vertical_fov_deg", 55.0))

        # ---- recording sessions: completed, interrupted and one active ----
        recordings: list[RecordingSession] = []
        for idx in range(3):
            started = NOW - timedelta(hours=6 - idx * 2)
            status = [RecordingStatus.COMPLETED, RecordingStatus.INTERRUPTED,
                      RecordingStatus.COMPLETED][idx]
            ended = started + timedelta(minutes=30 if idx != 1 else 7)
            rec = RecordingSession(
                source_id=source.id,
                file_path=(
                    f"{source.uid}/recording_"
                    f"{started.astimezone().strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
                ),
                status=status,
                started_at=started,
                ended_at=ended,
                duration_seconds=(ended - started).total_seconds(),
                frame_count=int((ended - started).total_seconds() * 15),
                fps=15.0, width=960, height=540, codec="mp4v",
                error=("source became unavailable mid-session"
                       if status is RecordingStatus.INTERRUPTED else None),
            )
            session.add(rec)
            recordings.append(rec)
        session.flush()

        # ---- motion episodes (aggregated, not per-frame) ----
        for idx in range(4):
            started = NOW - timedelta(hours=5, minutes=idx * 37)
            ended = started + timedelta(seconds=rng.randint(12, 90))
            session.add(
                MotionEvent(
                    source_id=source.id, recording_id=recordings[0].id,
                    started_at=started, ended_at=ended,
                    duration_seconds=(ended - started).total_seconds(),
                    peak_area_ratio=round(rng.uniform(0.01, 0.22), 4),
                    frame_start=idx * 900, frame_end=idx * 900 + 420,
                )
            )
            session.add(
                SecurityEvent(
                    source_id=source.id, event_type=EventType.MOTION_STARTED,
                    severity=EventSeverity.INFO, started_at=started, ended_at=ended,
                    duration_seconds=(ended - started).total_seconds(),
                    recording_id=recordings[0].id, message="Motion started",
                    dedup_key="motion", event_metadata={"seeded": True},
                )
            )
            total_events += 1

        # ---- tracked objects, including the acceptance-scenario cast ----
        scenarios = [
            ("person", "Azeem", RecognitionState.PERMANENT_FAMILIAR, 2.4, True, False),
            ("person", "John Smith", RecognitionState.TEMPORARY_FAMILIAR, 2.8, True, True),
            ("person", None, RecognitionState.UNFAMILIAR, 2.1, True, True),
            ("person", None, RecognitionState.FACE_UNRECOGNIZABLE, 6.5, False, False),
            ("person", None, RecognitionState.UNKNOWN_PENDING_RECOGNITION, 18.0,
             False, False),
            ("dog", None, RecognitionState.NO_FACE, 5.0, False, False),
            ("car", None, RecognitionState.NO_FACE, 11.0, False, False),
            ("person", "Sarah Khan", RecognitionState.PERMANENT_FAMILIAR, 4.2, False, False),
        ]

        for n, (cls, ident_name, state, min_dist, entered_b, alarms) in enumerate(
            scenarios
        ):
            first_seen = NOW - timedelta(hours=4, minutes=n * 23)
            last_seen = first_seen + timedelta(seconds=rng.randint(20, 180))
            identity = identities.get(ident_name) if ident_name else None
            zone = (
                ProximityZone.ZONE_B if min_dist <= source.proximity_b
                else ProximityZone.ZONE_A if min_dist <= source.proximity_a
                else ProximityZone.FAR
            )
            x1, y1, x2, y2 = bbox_for_distance(min_dist, fov_deg=fov)

            track = Track(
                source_id=source.id,
                recording_id=recordings[0].id,
                track_key=100 + n,
                object_class=cls,
                status=TrackStatus.ENDED,
                first_seen_at=first_seen,
                last_seen_at=last_seen,
                first_frame=n * 500,
                last_frame=n * 500 + 300,
                duration_seconds=(last_seen - first_seen).total_seconds(),
                detection_count=rng.randint(20, 140),
                max_confidence=round(rng.uniform(0.72, 0.96), 3),
                identity_id=identity.id if identity else None,
                recognition_state=state,
                recognition_confidence=(
                    round(rng.uniform(0.62, 0.93), 3) if identity else None
                ),
                proximity_zone=zone,
                min_distance_m=min_dist,
                last_distance_m=min_dist + rng.uniform(0, 1.5),
                entered_zone_a=min_dist <= source.proximity_a,
                entered_zone_b=entered_b,
                trajectory=[
                    {"f": n * 500 + k * 30, "x": round(x1 + k * 12, 1),
                     "y": round(y2 - 10, 1),
                     "t": (first_seen + timedelta(seconds=k)).isoformat()}
                    for k in range(6)
                ],
            )
            session.add(track)
            session.flush()
            total_tracks += 1

            for k in range(6):
                ts = first_seen + timedelta(seconds=k * 3)
                session.add(
                    Detection(
                        source_id=source.id, track_id=track.id,
                        recording_id=recordings[0].id, timestamp=ts,
                        frame_number=n * 500 + k * 30, object_class=cls,
                        confidence=round(rng.uniform(0.6, 0.95), 3),
                        bbox_x1=x1 + k * 8, bbox_y1=y1, bbox_x2=x2 + k * 8, bbox_y2=y2,
                        distance_m=round(min_dist + (5 - k) * 0.8, 2),
                        proximity_zone=zone,
                    )
                )

            events: list[tuple[EventType, EventSeverity, str]] = [
                (EventType.OBJECT_TRACK_STARTED, EventSeverity.INFO,
                 f"{cls} detected (track #{track.track_key})")
            ]
            if track.entered_zone_a:
                events.append((EventType.PROXIMITY_A_ENTERED, EventSeverity.INFO,
                               "Entered recognition zone"))
            if entered_b:
                events.append((EventType.PROXIMITY_B_ENTERED, EventSeverity.NOTICE,
                               "Entered alarm zone"))

            face_record: Face | None = None
            if cls == "person" and state is not RecognitionState.UNKNOWN_PENDING_RECOGNITION:
                quality_ok = state not in (RecognitionState.FACE_UNRECOGNIZABLE,)
                face_record = Face(
                    source_id=source.id, track_id=track.id,
                    identity_id=identity.id if identity else None,
                    recording_id=recordings[0].id,
                    detected_at=first_seen + timedelta(seconds=4),
                    frame_number=n * 500 + 60,
                    image_path=(
                        write_face_crop(
                            "unfamiliar",
                            f"{source.uid}_track{track.track_key}_"
                            f"{first_seen.strftime('%Y%m%d_%H%M%S')}.jpg",
                            synthetic_face(source.id * 1000 + n, "UNKNOWN"),
                        )
                        if state is RecognitionState.UNFAMILIAR
                        else (identity.thumbnail_path if identity else None)
                    ),
                    bbox_x1=x1 + 10, bbox_y1=y1 + 6, bbox_x2=x1 + 60, bbox_y2=y1 + 66,
                    detection_confidence=round(rng.uniform(0.78, 0.97), 3),
                    quality_score=round(rng.uniform(0.62, 0.9) if quality_ok
                                        else rng.uniform(0.12, 0.34), 3),
                    blur_score=round(rng.uniform(40, 180) if quality_ok
                                     else rng.uniform(4, 14), 2),
                    brightness=round(rng.uniform(80, 170), 1),
                    face_pixels=rng.randint(60, 140) if quality_ok else rng.randint(18, 40),
                    quality_ok=quality_ok,
                    quality_reason="ok" if quality_ok else "face_too_small,too_blurry",
                    recognition_state=state,
                    match_score=(round(rng.uniform(0.6, 0.92), 3) if identity else
                                 (round(rng.uniform(0.1, 0.4), 3)
                                  if state is RecognitionState.UNFAMILIAR else None)),
                    review_status=(
                        FaceReviewStatus.PENDING
                        if state is RecognitionState.UNFAMILIAR
                        else FaceReviewStatus.CLASSIFIED
                    ),
                )
                session.add(face_record)
                session.flush()

                if state is RecognitionState.UNFAMILIAR:
                    events.append((EventType.UNKNOWN_FACE, EventSeverity.WARNING,
                                   "Unfamiliar person detected"))
                elif state is RecognitionState.FACE_UNRECOGNIZABLE:
                    events.append((EventType.FACE_UNRECOGNIZABLE, EventSeverity.NOTICE,
                                   "Face detected but unrecognizable (face_too_small)"))
                elif identity:
                    events.append((EventType.FACE_RECOGNIZED, EventSeverity.INFO,
                                   f"Recognised {identity.display_name}"))
                    events.append((
                        EventType.PERMANENT_FAMILIAR_DETECTED
                        if state is RecognitionState.PERMANENT_FAMILIAR
                        else EventType.TEMPORARY_FAMILIAR_DETECTED,
                        EventSeverity.INFO if
                        state is RecognitionState.PERMANENT_FAMILIAR
                        else EventSeverity.NOTICE,
                        f"{identity.display_name} recognised",
                    ))

            alarm_event: SecurityEvent | None = None
            for offset, (etype, severity, message) in enumerate(events):
                ts = first_seen + timedelta(seconds=offset * 2)
                ev = SecurityEvent(
                    source_id=source.id, event_type=etype, severity=severity,
                    track_id=track.id,
                    identity_id=identity.id if identity else None,
                    face_id=face_record.id if face_record else None,
                    recording_id=recordings[0].id,
                    started_at=ts, ended_at=last_seen,
                    duration_seconds=(last_seen - ts).total_seconds(),
                    label=(identity.display_name if identity
                           else ("UNKNOWN" if state is RecognitionState.UNFAMILIAR
                                 else cls)),
                    message=message,
                    event_metadata={"track_key": track.track_key,
                                    "distance_m": min_dist},
                )
                session.add(ev)
                total_events += 1

            if alarms and state in (RecognitionState.UNFAMILIAR,
                                    RecognitionState.TEMPORARY_FAMILIAR):
                is_unknown = state is RecognitionState.UNFAMILIAR
                alarm_time = first_seen + timedelta(seconds=12)
                alarm_event = SecurityEvent(
                    source_id=source.id, event_type=EventType.ALARM_STARTED,
                    severity=EventSeverity.CRITICAL if is_unknown
                    else EventSeverity.NOTICE,
                    track_id=track.id,
                    identity_id=identity.id if identity else None,
                    recording_id=recordings[0].id,
                    started_at=alarm_time, ended_at=alarm_time,
                    label="UNKNOWN" if is_unknown else identity.display_name,
                    message=("Alarm: unfamiliar_person_in_alarm_zone" if is_unknown
                             else "Alert: temporary_familiar_in_alarm_zone"),
                    event_metadata={"track_key": track.track_key},
                )
                session.add(alarm_event)
                session.flush()
                total_events += 1

                # The newest unfamiliar alarm on camera_01 is left ACTIVE so
                # the operator can exercise STOP ALARM in the UI.
                leave_active = is_unknown and source.uid == "camera_01"
                session.add(
                    Alert(
                        source_id=source.id,
                        security_event_id=alarm_event.id,
                        track_id=track.id,
                        identity_id=identity.id if identity else None,
                        alert_type=AlertType.CONTINUOUS_ALARM if is_unknown
                        else AlertType.BEEP,
                        state=AlertState.ACTIVE if leave_active else AlertState.STOPPED,
                        reason="unfamiliar_person_in_alarm_zone" if is_unknown
                        else "temporary_familiar_in_alarm_zone",
                        started_at=alarm_time,
                        stopped_at=None if leave_active
                        else alarm_time + timedelta(seconds=25),
                        stopped_by=None if leave_active else "operator",
                        acknowledged=not leave_active,
                        alert_metadata={"track_key": track.track_key,
                                        "identity": ident_name},
                    )
                )
                total_alerts += 1

    session.flush()
    print(f"  tracks:     {total_tracks}")
    print(f"  events:     {total_events}")
    print(f"  alerts:     {total_alerts}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Sentinel demo data")
    parser.add_argument("--reset", action="store_true",
                        help="delete existing data before seeding")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    print(f"Seeding demo data into {settings.database_url.split('@')[-1]}")

    with session_scope() as session:
        if args.reset:
            reset(session)
        sources = seed_sources(session)
        identities = seed_identities(session)
        seed_activity(session, sources, identities)

    # Make the seeded identities immediately matchable.
    from app.services.identity_service import sync_recognition_index

    stats = sync_recognition_index()
    print(f"  index:      {stats['identities']} identities, "
          f"{stats['embeddings']} embeddings")
    print("\nDemo data ready. Start the backend and frontend, then open the dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
