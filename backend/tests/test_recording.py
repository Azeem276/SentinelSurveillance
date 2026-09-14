"""Recording lifecycle.

The guarantee under test: one uninterrupted capture produces exactly one
file, sessions always close safely, and resuming always opens a NEW file -
an interrupted recording is never appended to.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from app.models.enums import RecordingStatus
from app.recording.engine import RecordingEngine, session_filename, source_directory
from app.repositories.recording_repository import RecordingRepository
from app.storage.paths import recording_root


@pytest.fixture()
def recorder() -> RecordingEngine:
    """Named distinctly so it never shadows the database `engine` fixture."""
    return RecordingEngine(fourcc="mp4v", max_minutes=30)


@pytest.fixture()
def frame() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


class TestFileNaming:
    def test_filename_encodes_the_start_time(self):
        when = datetime(2026, 9, 14, 12, 0, 0)
        assert session_filename(when) == "recording_2026-09-14_12-00-00.mp4"

    def test_each_source_gets_its_own_directory(self):
        assert source_directory("camera_01") == "camera_01"
        assert "/" not in source_directory("../../etc/passwd")

    def test_directory_name_is_sanitised(self):
        assert source_directory("camera 01!@#") == "camera_01"


class TestSessionLifecycle:
    def test_start_creates_a_real_file(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        assert writer.is_open
        assert writer.absolute_path.exists()
        writer.close()

    def test_frames_are_written_and_counted(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        for _ in range(15):
            assert writer.write(frame)
        result = writer.close()
        assert result.frame_count == 15
        assert result.file_size_bytes and result.file_size_bytes > 0
        assert not result.interrupted

    def test_the_file_is_playable_afterwards(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        for _ in range(20):
            writer.write(frame)
        result = writer.close()

        cap = cv2.VideoCapture(str(result.absolute_path))
        try:
            assert cap.isOpened(), "closed recording must be a readable video"
            ok, decoded = cap.read()
            assert ok and decoded is not None
        finally:
            cap.release()

    def test_writes_after_close_are_rejected(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        writer.close()
        assert writer.write(frame) is False

    def test_close_is_idempotent(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        writer.write(frame)
        first = writer.close()
        second = writer.close()
        assert first.frame_count == second.frame_count == 1

    def test_interrupted_close_is_flagged(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        writer.write(frame)
        result = writer.close(interrupted=True, error="source lost")
        assert result.interrupted
        assert result.error == "source lost"
        # The partial file is still readable, not corrupt.
        assert result.absolute_path.exists()

    def test_mismatched_frame_sizes_are_resized_not_dropped(self, recorder):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        odd = np.zeros((90, 200, 3), dtype=np.uint8)
        assert writer.write(odd)
        assert writer.close().frame_count == 1


class TestSessionSeparation:
    def test_restart_creates_a_new_file(self, recorder, frame):
        first = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        first.write(frame)
        first_result = first.close()

        second = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        second.write(frame)
        second_result = second.close()

        assert first_result.absolute_path != second_result.absolute_path
        assert first_result.absolute_path.exists()
        assert second_result.absolute_path.exists()

    def test_two_sessions_in_the_same_second_do_not_collide(self, recorder, frame):
        when = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)
        a = recorder.start_session(source_uid="camera_09", width=160, height=120,
                                 fps=10, when=when)
        a.write(frame)
        a.close()
        b = recorder.start_session(source_uid="camera_09", width=160, height=120,
                                 fps=10, when=when)
        b.write(frame)
        b.close()
        assert a.absolute_path != b.absolute_path

    def test_sources_write_to_separate_directories(self, recorder, frame):
        a = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        b = recorder.start_session(source_uid="camera_02", width=160, height=120, fps=10)
        a.close()
        b.close()
        assert a.absolute_path.parent.name == "camera_01"
        assert b.absolute_path.parent.name == "camera_02"

    def test_recordings_stay_inside_the_storage_root(self, recorder):
        writer = recorder.start_session(source_uid="../escape", width=160, height=120, fps=10)
        writer.close()
        assert recording_root() in writer.absolute_path.parents


class TestRotation:
    def test_no_rotation_before_the_limit(self, recorder, frame):
        writer = recorder.start_session(source_uid="camera_01", width=160, height=120, fps=10)
        assert not recorder.should_rotate(writer)
        writer.close()

    def test_rotation_after_the_limit(self, frame):
        recorder = RecordingEngine(fourcc="mp4v", max_minutes=30)
        writer = recorder.start_session(
            source_uid="camera_01", width=160, height=120, fps=10,
            when=datetime.now(timezone.utc) - timedelta(minutes=31),
        )
        assert recorder.should_rotate(writer)
        writer.close()

    def test_rotation_can_be_disabled(self, frame):
        recorder = RecordingEngine(fourcc="mp4v", max_minutes=0)
        writer = recorder.start_session(
            source_uid="camera_01", width=160, height=120, fps=10,
            when=datetime.now(timezone.utc) - timedelta(days=2),
        )
        assert not recorder.should_rotate(writer)
        writer.close()


class TestRecordingPersistence:
    def test_start_and_finish_round_trip(self, session, make_source):
        source = make_source()
        repo = RecordingRepository(session)
        record = repo.start(source_id=source.id, file_path="camera_01/a.mp4", fps=15.0)
        assert record.status is RecordingStatus.ACTIVE
        assert repo.active_for_source(source.id) is not None

        repo.finish(record.id, status=RecordingStatus.COMPLETED, frame_count=100,
                    file_size_bytes=2048)
        session.refresh(record)
        assert record.status is RecordingStatus.COMPLETED
        assert record.frame_count == 100
        assert record.duration_seconds is not None
        assert repo.active_for_source(source.id) is None

    def test_orphans_from_a_crash_become_interrupted(self, session, make_source):
        """Startup recovery: a session left ACTIVE was not closed cleanly."""
        source = make_source()
        repo = RecordingRepository(session)
        record = repo.start(source_id=source.id, file_path="camera_01/crash.mp4")
        assert repo.close_orphans() == 1
        session.refresh(record)
        assert record.status is RecordingStatus.INTERRUPTED
        assert record.error and "restart" in record.error

    def test_recovery_does_not_touch_finished_sessions(self, session, make_source):
        source = make_source()
        repo = RecordingRepository(session)
        record = repo.start(source_id=source.id, file_path="camera_01/done.mp4")
        repo.finish(record.id, status=RecordingStatus.COMPLETED, frame_count=10,
                    file_size_bytes=10)
        assert repo.close_orphans() == 0
        session.refresh(record)
        assert record.status is RecordingStatus.COMPLETED

    def test_listing_is_newest_first(self, session, make_source):
        source = make_source()
        repo = RecordingRepository(session)
        now = datetime.now(timezone.utc)
        for i in range(3):
            repo.start(source_id=source.id, file_path=f"camera_01/{i}.mp4",
                       started_at=now - timedelta(hours=i))
        rows = repo.list_for_source(source.id)
        assert [r.file_path for r in rows] == [
            "camera_01/0.mp4", "camera_01/1.mp4", "camera_01/2.mp4"
        ]

    def test_stats_aggregate_across_sessions(self, session, make_source):
        source = make_source()
        repo = RecordingRepository(session)
        for i in range(2):
            record = repo.start(source_id=source.id, file_path=f"camera_01/{i}.mp4")
            repo.finish(record.id, status=RecordingStatus.COMPLETED, frame_count=50,
                        file_size_bytes=1000)
        stats = repo.stats(source.id)
        assert stats["count"] == 2
        assert stats["total_bytes"] == 2000
