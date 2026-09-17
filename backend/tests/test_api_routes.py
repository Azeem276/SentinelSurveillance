"""HTTP-level tests for the FastAPI routes.

These go through the real app (routers, dependency injection, error handlers)
against the SQLite test database. They complement the service/repository
tests by checking status codes, payload shapes and the cross-layer rules an
API client relies on.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from app.alerts.engine import get_alert_engine
from app.models.enums import (
    FaceReviewStatus, IdentityCategory, RecognitionState, RecordingStatus, SourceType,
)
from app.models.identity import Face
from app.models.recording import RecordingSession
from app.models.source import VideoSource
from app.repositories.identity_repository import generate_identifier
from app.repositories.recording_repository import RecordingRepository


# ------------------------------------------------------------------ helpers
def _source_payload(**overrides) -> dict:
    payload = {
        "uid": "cam_api_01",
        "name": "API camera",
        "type": "FILE",
        "uri": "unit_test_clip.mp4",
        "proximity_a": 10.0,
        "proximity_b": 3.0,
    }
    payload.update(overrides)
    return payload


def _add_source(committed_session, **overrides) -> VideoSource:
    fields = dict(
        uid="cam_db_01", name="DB camera", type=SourceType.FILE, uri="clip.mp4",
        proximity_a=10.0, proximity_b=3.0, alert_policy={}, calibration={},
    )
    fields.update(overrides)
    source = VideoSource(**fields)
    committed_session.add(source)
    committed_session.commit()
    return source


def _add_pending_face(committed_session, source_id: int, *, detected_at=None) -> Face:
    face = Face(
        source_id=source_id,
        detected_at=detected_at or datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc),
        frame_number=42,
        bbox_x1=10, bbox_y1=10, bbox_x2=70, bbox_y2=80,
        detection_confidence=0.9,
        quality_score=0.8,
        quality_ok=True,
        recognition_state=RecognitionState.UNFAMILIAR,
        review_status=FaceReviewStatus.PENDING,
    )
    committed_session.add(face)
    committed_session.commit()
    return face


def _wait_until(predicate, timeout: float = 8.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ------------------------------------------------------------- system state
class TestSystemRoutes:
    def test_health_reports_database_connected(self, api_client):
        body = api_client.get("/api/system/health").json()
        assert body["status"] == "ok"
        assert body["database"]["connected"] is True

    def test_state_starts_with_everything_off(self, api_client):
        body = api_client.get("/api/system/state").json()
        assert body["surveillance_active"] is False
        assert body["intelligence_active"] is False
        assert body["running_sources"] == 0
        assert body["active_alerts"] == 0

    def test_unknown_route_is_404(self, api_client):
        assert api_client.get("/api/does-not-exist").status_code == 404


# --------------------------------------------- surveillance / intelligence
class TestSurveillanceControl:
    def test_intelligence_is_rejected_while_surveillance_is_off(self, api_client):
        response = api_client.post("/api/intelligence/start")
        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "precondition_failed"
        assert "surveillance" in body["message"].lower()
        assert api_client.get("/api/system/state").json()["intelligence_active"] is False

    def test_start_then_intelligence_then_stop(self, api_client):
        started = api_client.post("/api/surveillance/start")
        assert started.status_code == 200
        assert started.json() == {
            "ok": True, "data": {"surveillance": True, "started_sources": []},
        }

        intel = api_client.post("/api/intelligence/start")
        assert intel.status_code == 200
        assert intel.json()["data"]["intelligence"] is True
        state = api_client.get("/api/system/state").json()
        assert state["surveillance_active"] is True
        assert state["intelligence_active"] is True

        stopped = api_client.post("/api/surveillance/stop")
        assert stopped.status_code == 200
        assert stopped.json()["data"] == {"surveillance": False, "stopped_sources": 0}
        state = api_client.get("/api/system/state").json()
        assert state["surveillance_active"] is False
        # Stopping surveillance stops intelligence too; it cannot run alone.
        assert state["intelligence_active"] is False

    def test_intelligence_stop_is_allowed_any_time(self, api_client):
        response = api_client.post("/api/intelligence/stop")
        assert response.status_code == 200
        assert response.json()["data"]["intelligence"] is False

    def test_stop_is_idempotent(self, api_client):
        assert api_client.post("/api/surveillance/stop").status_code == 200
        assert api_client.post("/api/surveillance/stop").status_code == 200

    def test_a_file_source_is_captured_recorded_and_closed_completed(
        self, api_client, sample_video, engine
    ):
        """End to end through the API: real capture thread, real recording file."""
        created = api_client.post(
            "/api/sources",
            json=_source_payload(uri=sample_video.name, intelligence_enabled=False),
        )
        assert created.status_code == 201, created.text
        source_id = created.json()["id"]

        started = api_client.post("/api/surveillance/start").json()["data"]
        assert started["started_sources"] == ["cam_api_01"]

        def _recording():
            runtime = api_client.get(f"/api/sources/{source_id}").json()["runtime"]
            return runtime["running"] and runtime["recording"] and runtime["frames_read"] > 3

        assert _wait_until(_recording), api_client.get(f"/api/sources/{source_id}").json()
        assert api_client.get("/api/system/state").json()["recording_sources"] == 1

        stopped = api_client.post("/api/surveillance/stop")
        assert stopped.status_code == 200
        assert stopped.json()["data"]["stopped_sources"] == 1

        recordings = api_client.get("/api/recordings").json()
        assert len(recordings) == 1
        assert recordings[0]["status"] == RecordingStatus.COMPLETED.value
        assert recordings[0]["frame_count"] > 0
        assert recordings[0]["exists_on_disk"] is True
        assert api_client.get(f"/api/sources/{source_id}").json()["runtime"]["running"] is False


# ------------------------------------------------------------ source CRUD
class TestSourceRoutes:
    def test_create_read_update_delete(self, api_client, sample_video):
        created = api_client.post("/api/sources", json=_source_payload(uri=sample_video.name))
        assert created.status_code == 201, created.text
        body = created.json()
        source_id = body["id"]
        assert body["uid"] == "cam_api_01"
        assert body["proximity_a"] == 10.0 and body["proximity_b"] == 3.0
        assert body["runtime"]["running"] is False

        listed = api_client.get("/api/sources").json()
        assert [s["id"] for s in listed] == [source_id]

        fetched = api_client.get(f"/api/sources/{source_id}")
        assert fetched.status_code == 200
        assert fetched.json()["name"] == "API camera"

        updated = api_client.patch(
            f"/api/sources/{source_id}",
            json={"name": "Front door", "proximity_a": 12.0, "proximity_b": 4.0},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["name"] == "Front door"
        assert updated.json()["proximity_a"] == 12.0
        assert updated.json()["proximity_b"] == 4.0

        deleted = api_client.delete(f"/api/sources/{source_id}")
        assert deleted.status_code == 200
        assert api_client.get(f"/api/sources/{source_id}").status_code == 404
        assert api_client.get("/api/sources").json() == []

    def test_deleting_a_source_cascades_its_recordings(
        self, api_client, committed_session
    ):
        """A source that has recorded is still deletable.

        Regression: VideoSource.recordings had no cascade, so SQLAlchemy tried
        to NULL recording_sessions.source_id -- a NOT NULL column -- and the
        delete blew up with an IntegrityError instead of removing anything.
        The rows must go with the parent, as the FK's ON DELETE CASCADE says.
        """
        source = _add_source(committed_session)
        repo = RecordingRepository(committed_session)
        for n in range(2):
            repo.start(
                source_id=source.id,
                file_path=f"{source.uid}/clip_{n}.mp4",
                started_at=datetime.now(timezone.utc),
                fps=15.0,
                width=960,
                height=540,
            )
        committed_session.commit()
        source_id = source.id

        deleted = api_client.delete(f"/api/sources/{source_id}")
        assert deleted.status_code == 200, deleted.text
        assert api_client.get(f"/api/sources/{source_id}").status_code == 404

        committed_session.expire_all()
        remaining = (
            committed_session.query(RecordingSession)
            .filter_by(source_id=source_id)
            .count()
        )
        assert remaining == 0

    @pytest.mark.parametrize("a,b", [(3.0, 10.0), (5.0, 5.0), (2.9, 3.0)])
    def test_create_rejects_proximity_a_not_greater_than_b(
        self, api_client, sample_video, a, b
    ):
        response = api_client.post(
            "/api/sources",
            json=_source_payload(uri=sample_video.name, proximity_a=a, proximity_b=b),
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "validation_error"
        assert api_client.get("/api/sources").json() == []

    def test_update_rejects_proximity_a_not_greater_than_b(self, api_client, sample_video):
        source_id = api_client.post(
            "/api/sources", json=_source_payload(uri=sample_video.name)
        ).json()["id"]

        # Each field alone would pass the per-field check; the pair must not.
        for patch in ({"proximity_b": 10.0}, {"proximity_a": 3.0}, {"proximity_a": 2.0, "proximity_b": 2.0}):
            response = api_client.patch(f"/api/sources/{source_id}", json=patch)
            assert response.status_code == 422, (patch, response.text)
            assert response.json()["code"] == "validation_error"

        unchanged = api_client.get(f"/api/sources/{source_id}").json()
        assert unchanged["proximity_a"] == 10.0 and unchanged["proximity_b"] == 3.0

    @pytest.mark.parametrize("a,b", [(0, 3.0), (10.0, 0), (-1.0, 3.0), (600.0, 3.0)])
    def test_out_of_range_distances_are_rejected(self, api_client, sample_video, a, b):
        response = api_client.post(
            "/api/sources",
            json=_source_payload(uri=sample_video.name, proximity_a=a, proximity_b=b),
        )
        assert response.status_code == 422

    def test_duplicate_uid_is_rejected(self, api_client, sample_video):
        payload = _source_payload(uri=sample_video.name)
        assert api_client.post("/api/sources", json=payload).status_code == 201
        response = api_client.post("/api/sources", json=payload)
        assert response.status_code == 422
        assert "already exists" in response.json()["message"]

    def test_missing_video_file_is_rejected(self, api_client):
        response = api_client.post("/api/sources", json=_source_payload(uri="nope.mp4"))
        assert response.status_code == 422
        assert "not found" in response.json()["message"]

    @pytest.mark.parametrize(
        "uri", ["../../etc/passwd.mp4", "/etc/shadow.mp4", "C:\\clip.mp4", "clip.exe"]
    )
    def test_unsafe_video_uri_is_rejected(self, api_client, uri):
        response = api_client.post("/api/sources", json=_source_payload(uri=uri))
        assert response.status_code == 400
        assert response.json()["code"] == "storage_error"

    def test_rtsp_source_requires_a_stream_uri(self, api_client):
        response = api_client.post(
            "/api/sources", json=_source_payload(type="RTSP", uri="not-a-stream")
        )
        assert response.status_code == 422
        ok = api_client.post(
            "/api/sources",
            json=_source_payload(type="RTSP", uri="rtsp://192.168.0.10:554/live"),
        )
        assert ok.status_code == 201, ok.text

    def test_unknown_source_is_404(self, api_client):
        assert api_client.get("/api/sources/4242").status_code == 404
        assert api_client.delete("/api/sources/4242").status_code == 404
        assert api_client.get("/api/sources/4242/tracks").status_code == 404


# -------------------------------------------------------------- identities
class TestIdentityRoutes:
    def test_list_and_detail_carry_a_label(self, api_client):
        """Regression: ``label`` was a read-only property the route assigned to,
        so every identity endpoint returned 500 once one identity existed."""
        named = api_client.post(
            "/api/identities", json={"category": "PERMANENT", "display_name": "Azeem"}
        )
        unnamed = api_client.post("/api/identities", json={"category": "TEMPORARY"})
        assert named.status_code == 201, named.text
        assert unnamed.status_code == 201, unnamed.text
        assert named.json()["label"] == "Azeem"
        assert unnamed.json()["label"] == unnamed.json()["generated_identifier"]
        assert unnamed.json()["label"].startswith("Unknown_")

        listed = api_client.get("/api/identities")
        assert listed.status_code == 200, listed.text
        assert sorted(i["label"] for i in listed.json()) == sorted(
            [named.json()["label"], unnamed.json()["label"]]
        )
        detail = api_client.get(f"/api/identities/{named.json()['id']}")
        assert detail.status_code == 200
        assert detail.json()["label"] == "Azeem"
        assert detail.json()["embedding_count"] == 0

    def test_unknown_identity_is_404(self, api_client):
        assert api_client.get("/api/identities/999999").status_code == 404


# ------------------------------------------------------- face classification
class TestFaceClassification:
    def test_named_permanent_classification(self, api_client, committed_session):
        source = _add_source(committed_session)
        face = _add_pending_face(committed_session, source.id)

        assert api_client.get("/api/faces/pending-count").json()["data"]["pending"] == 1
        response = api_client.post(
            f"/api/faces/{face.id}/classify",
            json={"category": "PERMANENT", "display_name": "Azeem"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["created"] is True
        assert body["display_name"] == "Azeem"
        assert body["category"] == "PERMANENT"
        assert body["expires_at"] is None

        identity = api_client.get(f"/api/identities/{body['identity_id']}").json()
        assert identity["display_name"] == "Azeem"
        assert identity["category"] == "PERMANENT"
        assert api_client.get("/api/faces/pending-count").json()["data"]["pending"] == 0
        assert api_client.get("/api/faces/unfamiliar").json() == []

    def test_unnamed_classification_gets_a_timestamp_identifier(
        self, api_client, committed_session
    ):
        source = _add_source(committed_session)
        detected_at = datetime(2026, 9, 14, 2, 13, 15, tzinfo=timezone.utc)
        face = _add_pending_face(committed_session, source.id, detected_at=detected_at)

        response = api_client.post(
            f"/api/faces/{face.id}/classify",
            json={"category": "TEMPORARY", "display_name": "", "retention_preset": "7d"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["display_name"] is None
        assert body["identifier"] == generate_identifier(detected_at)
        assert body["identifier"].startswith("Unknown_")
        assert body["category"] == "TEMPORARY"
        assert body["expires_at"] is not None

    def test_classifying_into_an_existing_identity(self, api_client, committed_session):
        source = _add_source(committed_session)
        face = _add_pending_face(committed_session, source.id)
        existing = api_client.post(
            "/api/identities", json={"category": "PERMANENT", "display_name": "John"}
        )
        assert existing.status_code == 201, existing.text

        response = api_client.post(
            f"/api/faces/{face.id}/classify",
            json={"category": "PERMANENT", "identity_id": existing.json()["id"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["created"] is False
        assert response.json()["identity_id"] == existing.json()["id"]

    def test_unknown_face_is_404(self, api_client):
        response = api_client.post(
            "/api/faces/999999/classify", json={"category": "PERMANENT"}
        )
        assert response.status_code == 404

    def test_invalid_category_is_422(self, api_client, committed_session):
        source = _add_source(committed_session)
        face = _add_pending_face(committed_session, source.id)
        response = api_client.post(
            f"/api/faces/{face.id}/classify", json={"category": "SOMETHING"}
        )
        assert response.status_code == 422

    def test_dismiss_removes_from_review_without_creating_an_identity(
        self, api_client, committed_session
    ):
        source = _add_source(committed_session)
        face = _add_pending_face(committed_session, source.id)
        response = api_client.post(f"/api/faces/{face.id}/dismiss")
        assert response.status_code == 200
        assert api_client.get("/api/faces/pending-count").json()["data"]["pending"] == 0
        assert api_client.get("/api/identities").json() == []


class TestBulkClassification:
    def _three_faces(self, committed_session) -> list[int]:
        source = _add_source(committed_session)
        ids = []
        for minute in (1, 2, 3):
            face = _add_pending_face(
                committed_session, source.id,
                detected_at=datetime(2026, 9, 14, 2, minute, 0, tzinfo=timezone.utc),
            )
            ids.append(face.id)
        return ids

    def test_bulk_does_not_merge_distinct_people_by_default(
        self, api_client, committed_session
    ):
        face_ids = self._three_faces(committed_session)
        response = api_client.post(
            "/api/faces/classify-bulk",
            json={"face_ids": face_ids, "category": "TEMPORARY", "display_name": "Visitor"},
        )
        assert response.status_code == 200, response.text
        results = response.json()
        assert len(results) == 3
        assert all(r["created"] for r in results)
        assert len({r["identity_id"] for r in results}) == 3
        assert len({r["identifier"] for r in results}) == 3
        assert all(r["display_name"] == "Visitor" for r in results)
        assert len(api_client.get("/api/identities").json()) == 3
        assert api_client.get("/api/faces/pending-count").json()["data"]["pending"] == 0

    def test_bulk_merges_only_when_explicitly_asked(self, api_client, committed_session):
        face_ids = self._three_faces(committed_session)
        response = api_client.post(
            "/api/faces/classify-bulk",
            json={
                "face_ids": face_ids,
                "category": "PERMANENT",
                "display_name": "Azeem",
                "merge_into_one_identity": True,
            },
        )
        assert response.status_code == 200, response.text
        results = response.json()
        assert [r["created"] for r in results] == [True, False, False]
        assert len({r["identity_id"] for r in results}) == 1
        assert len(api_client.get("/api/identities").json()) == 1

    def test_bulk_requires_at_least_one_face(self, api_client):
        response = api_client.post(
            "/api/faces/classify-bulk", json={"face_ids": [], "category": "PERMANENT"}
        )
        assert response.status_code == 422


# ------------------------------------------------------------------ alarms
class TestAlertRoutes:
    def _raise_alarm(self, committed_session, source, track_key: int):
        alert = get_alert_engine().start_continuous(
            committed_session, source_id=source.id, source_uid=source.uid,
            reason="unfamiliar_person_in_alarm_zone", track_key=track_key,
        )
        committed_session.commit()
        return alert

    def test_stop_one_alarm(self, api_client, committed_session):
        source = _add_source(committed_session)
        alert = self._raise_alarm(committed_session, source, track_key=7)
        assert [a["id"] for a in api_client.get("/api/alerts/active").json()] == [alert.id]
        assert api_client.get("/api/system/state").json()["active_alerts"] == 1

        response = api_client.post(f"/api/alerts/{alert.id}/stop?stopped_by=operator")
        assert response.status_code == 200, response.text
        assert response.json()["data"] == {"alert_id": alert.id, "state": "STOPPED"}

        assert api_client.get("/api/alerts/active").json() == []
        stored = next(a for a in api_client.get("/api/alerts").json() if a["id"] == alert.id)
        assert stored["state"] == "STOPPED"
        assert stored["stopped_by"] == "operator"
        assert stored["stopped_at"] is not None
        # The engine agrees, so the same track can alarm again later.
        assert get_alert_engine().is_alarming(source.id, 7) is False

    def test_stop_all_alarms(self, api_client, committed_session):
        source = _add_source(committed_session)
        for key in (1, 2, 3):
            self._raise_alarm(committed_session, source, track_key=key)
        assert len(api_client.get("/api/alerts/active").json()) == 3

        response = api_client.post("/api/alerts/stop-all")
        assert response.status_code == 200
        assert response.json()["data"] == {"stopped": 3}
        assert api_client.get("/api/alerts/active").json() == []
        assert api_client.get("/api/system/state").json()["active_alerts"] == 0

    def test_stop_all_can_be_scoped_to_a_source(self, api_client, committed_session):
        first = _add_source(committed_session, uid="cam_a")
        second = _add_source(committed_session, uid="cam_b")
        self._raise_alarm(committed_session, first, track_key=1)
        kept = self._raise_alarm(committed_session, second, track_key=1)

        response = api_client.post(f"/api/alerts/stop-all?source_id={first.id}")
        assert response.json()["data"] == {"stopped": 1}
        assert [a["id"] for a in api_client.get("/api/alerts/active").json()] == [kept.id]

    def test_stopping_an_unknown_alarm_is_404(self, api_client):
        assert api_client.post("/api/alerts/999999/stop").status_code == 404

    def test_stopping_twice_is_harmless(self, api_client, committed_session):
        source = _add_source(committed_session)
        alert = self._raise_alarm(committed_session, source, track_key=7)
        assert api_client.post(f"/api/alerts/{alert.id}/stop").status_code == 200
        second = api_client.post(f"/api/alerts/{alert.id}/stop")
        assert second.status_code == 200
        assert second.json()["data"]["state"] == "STOPPED"


# ------------------------------------------------------- face image serving
class TestFaceImageRoute:
    # HTTP clients (browsers, curl, httpx) collapse a literal ``../`` before the
    # request is sent, so the server-side guard is exercised with the encoded
    # forms that actually reach the route decoded as ``..``.
    @pytest.mark.parametrize(
        "path",
        [
            "..%2F..%2Fetc%2Fpasswd",
            "%2E%2E%2F%2E%2E%2Fetc%2Fpasswd",
            "unfamiliar%2F..%2F..%2F.env",
            "%2Fetc%2Fshadow",
            "C:%5CWindows%5Cwin.ini",
            "..%5C..%5Cwindows%5Csystem32%5Cconfig%5Csam",
        ],
    )
    def test_path_traversal_is_rejected(self, api_client, path):
        response = api_client.get(f"/api/faces/image/{path}")
        assert response.status_code == 400, response.text
        assert response.json()["code"] == "storage_error"

    def test_traversal_never_leaks_a_real_file(self, api_client, _test_environment):
        # A file that exists just outside the face root must still be unreachable.
        outside = _test_environment.storage_path / "secret.txt"
        outside.write_text("top secret")
        response = api_client.get("/api/faces/image/..%2Fsecret.txt")
        assert response.status_code == 400
        assert b"top secret" not in response.content

    def test_missing_image_is_404(self, api_client):
        response = api_client.get("/api/faces/image/unfamiliar/nope.jpg")
        assert response.status_code == 404

    def test_stored_crop_is_served_with_the_right_media_type(
        self, api_client, _test_environment
    ):
        crop = _test_environment.face_storage_path / "unfamiliar" / "cam_track1.jpg"
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        response = api_client.get("/api/faces/image/unfamiliar/cam_track1.jpg")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8\xff\xe0fake-jpeg"


# --------------------------------------------------------------- recordings
class TestArchiveRoutes:
    def test_recording_playback_is_404_when_file_is_gone(
        self, api_client, committed_session
    ):
        source = _add_source(committed_session)
        recording = RecordingRepository(committed_session).start(
            source_id=source.id, file_path=f"{source.uid}/missing.mp4",
            started_at=datetime.now(timezone.utc), fps=15.0, width=960, height=540,
        )
        committed_session.commit()
        assert api_client.get(f"/api/recordings/{recording.id}").json()["exists_on_disk"] is False
        response = api_client.get(f"/api/recordings/{recording.id}/play")
        assert response.status_code == 404

    def test_unknown_recording_is_404(self, api_client):
        assert api_client.get("/api/recordings/424242").status_code == 404
