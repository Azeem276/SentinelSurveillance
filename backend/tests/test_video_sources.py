"""VideoSource abstraction, path safety and the AI adapters.

The RTSP readiness requirement is checked structurally: both adapters satisfy
the same contract, so the intelligence layer cannot depend on which is in use.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import StorageError
from app.intelligence.motion.detector import MOG2MotionDetector
from app.intelligence.tracking.iou_tracker import IoUTracker, PassthroughTracker
from app.intelligence.types import BBox, ObjectDetection
from app.storage.paths import (
    resolve_video, safe_join, slugify, validate_video_upload, video_root,
)
from app.video.base import SourceUnavailableError, VideoSourceAdapter
from app.video.file_source import FileVideoSource
from app.video.rtsp_source import RTSPCameraSource


# ------------------------------------------------------------ abstraction
class TestSourceAbstraction:
    def test_both_adapters_implement_the_same_contract(self):
        for adapter in (FileVideoSource, RTSPCameraSource):
            assert issubclass(adapter, VideoSourceAdapter)
            for method in ("_open", "_read", "_release", "open", "read", "release"):
                assert hasattr(adapter, method)

    def test_adapters_declare_distinct_types(self):
        assert FileVideoSource.source_type == "FILE"
        assert RTSPCameraSource.source_type == "RTSP"

    def test_the_frame_object_is_source_agnostic(self, sample_video):
        """Downstream code sees only Frame, never a file handle or a socket."""
        with FileVideoSource("camera_01", sample_video, realtime=False) as source:
            frame = source.read()
        assert frame is not None
        assert frame.source_uid == "camera_01"
        assert frame.frame_number == 1
        assert isinstance(frame.image, np.ndarray)
        assert frame.width > 0 and frame.height > 0


class TestFileVideoSource:
    def test_open_reports_capabilities(self, sample_video):
        source = FileVideoSource("camera_01", sample_video, realtime=False)
        caps = source.open()
        assert caps.width == 160 and caps.height == 120
        assert caps.fps > 0
        assert caps.frame_count == 30
        assert not caps.is_live and caps.seekable
        source.release()

    def test_reads_every_frame_then_returns_none(self, sample_video):
        with FileVideoSource("camera_01", sample_video, realtime=False) as source:
            count = 0
            while source.read() is not None:
                count += 1
                if count > 100:
                    pytest.fail("source never reported exhaustion")
        assert count == 30

    def test_missing_file_raises_unavailable(self, tmp_path):
        source = FileVideoSource("camera_x", tmp_path / "nope.mp4")
        with pytest.raises(SourceUnavailableError, match="not found"):
            source.open()

    def test_malformed_file_raises_unavailable(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"this is not a video")
        source = FileVideoSource("camera_x", broken)
        with pytest.raises(SourceUnavailableError):
            source.open()

    def test_reading_before_open_raises(self, sample_video):
        source = FileVideoSource("camera_01", sample_video)
        with pytest.raises(SourceUnavailableError, match="not open"):
            source.read()

    def test_looping_keeps_delivering_frames(self, sample_video):
        with FileVideoSource("camera_01", sample_video, loop=True,
                             realtime=False) as source:
            frames = [source.read() for _ in range(45)]
        assert all(f is not None for f in frames), "a looping source never ends"
        assert source.loop_count >= 1

    def test_release_is_idempotent(self, sample_video):
        source = FileVideoSource("camera_01", sample_video, realtime=False)
        source.open()
        source.release()
        source.release()
        assert not source.is_open

    def test_reset_rewinds_a_seekable_source(self, sample_video):
        with FileVideoSource("camera_01", sample_video, realtime=False) as source:
            for _ in range(5):
                source.read()
            assert source.reset()
            assert source.read() is not None


class TestRTSPSourceConfiguration:
    def test_construction_does_not_require_a_network(self):
        source = RTSPCameraSource("cam", "rtsp://192.0.2.1/stream")
        assert source.uri == "rtsp://192.0.2.1/stream"
        assert source.transport == "tcp"
        assert not source.is_open

    def test_live_sources_are_not_seekable(self):
        source = RTSPCameraSource("cam", "rtsp://192.0.2.1/stream")
        assert source.reset() is False


# ------------------------------------------------------------ path safety
class TestPathSafety:
    @pytest.mark.parametrize(
        "evil",
        [
            "../../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "subdir/../../outside.mp4",
        ],
    )
    def test_traversal_attempts_are_rejected(self, evil):
        with pytest.raises(StorageError):
            safe_join(video_root(), evil)

    def test_legitimate_relative_paths_are_allowed(self):
        resolved = safe_join(video_root(), "camera_01/clip.mp4")
        assert video_root() in resolved.parents

    def test_unsupported_video_extensions_are_rejected(self):
        with pytest.raises(StorageError, match="Unsupported video type"):
            resolve_video("payload.exe")

    def test_supported_extensions_are_accepted(self):
        assert resolve_video("clip.mp4").suffix == ".mp4"

    def test_uploads_are_sanitised(self):
        assert validate_video_upload("my movie (1).mp4") == "my_movie_1.mp4"

    def test_upload_of_a_disallowed_type_is_rejected(self):
        with pytest.raises(StorageError):
            validate_video_upload("malware.bat")

    def test_upload_cannot_smuggle_a_path(self):
        assert "/" not in validate_video_upload("../../evil.mp4")

    def test_slugify_produces_a_safe_token(self):
        assert slugify("Front Door #1") == "Front_Door_1"
        assert slugify("") == "item"


# ---------------------------------------------------------------- tracking
def det(x1, y1, x2, y2, cls="person", conf=0.9, track_key=None):
    return ObjectDetection(BBox(x1, y1, x2, y2), cls, conf, track_key=track_key)


class TestIoUTracker:
    def test_a_track_needs_min_hits_to_confirm(self):
        tracker = IoUTracker(min_hits=3, max_age=5)
        assert tracker.update([det(0, 0, 50, 100)]) == []
        assert tracker.update([det(2, 0, 52, 100)]) == []
        assert len(tracker.update([det(4, 0, 54, 100)])) == 1

    def test_identity_of_the_track_key_is_stable(self):
        tracker = IoUTracker(min_hits=2, max_age=5)
        keys = set()
        for i in range(10):
            for obj in tracker.update([det(i * 3, 0, 50 + i * 3, 100)]):
                keys.add(obj.track_key)
        assert keys == {1}, "one moving person is one track"

    def test_two_people_get_two_tracks(self):
        tracker = IoUTracker(min_hits=2, max_age=5)
        for i in range(4):
            tracker.update([det(i, 0, 50 + i, 100), det(300 + i, 0, 350 + i, 100)])
        assert len(tracker.active_tracks) == 2

    def test_a_track_survives_brief_occlusion(self):
        tracker = IoUTracker(min_hits=2, max_age=10)
        for i in range(4):
            tracker.update([det(i * 2, 0, 50 + i * 2, 100)])
        for _ in range(3):
            tracker.update([])          # occluded
        results = tracker.update([det(14, 0, 64, 100)])
        assert results and results[0].track_key == 1

    def test_a_track_ends_after_max_age(self):
        tracker = IoUTracker(min_hits=2, max_age=3)
        for i in range(4):
            tracker.update([det(i, 0, 50 + i, 100)])
        for _ in range(5):
            tracker.update([])
        assert tracker.active_tracks == []
        assert 1 in tracker.removed_track_keys

    def test_low_confidence_detections_do_not_spawn_tracks(self):
        tracker = IoUTracker(min_hits=1, high_confidence=0.55)
        assert tracker.update([det(0, 0, 50, 100, conf=0.3)]) == []

    def test_low_confidence_detections_keep_existing_tracks_alive(self):
        """The ByteTrack idea: weak detections rescue, they do not create."""
        tracker = IoUTracker(min_hits=2, max_age=5, high_confidence=0.55)
        tracker.update([det(0, 0, 50, 100, conf=0.9)])
        tracker.update([det(2, 0, 52, 100, conf=0.9)])
        results = tracker.update([det(4, 0, 54, 100, conf=0.35)])
        assert results and results[0].track_key == 1

    def test_reset_clears_all_state(self):
        tracker = IoUTracker(min_hits=1)
        tracker.update([det(0, 0, 50, 100)])
        tracker.reset()
        assert tracker.active_tracks == []

    def test_object_class_is_carried_on_the_track(self):
        tracker = IoUTracker(min_hits=1)
        results = tracker.update([det(0, 0, 60, 40, cls="dog")])
        assert results[0].object_class == "dog"


class TestPassthroughTracker:
    def test_detector_supplied_ids_are_preserved(self):
        tracker = PassthroughTracker(min_hits=1)
        results = tracker.update([det(0, 0, 50, 100, track_key=42)])
        assert results[0].track_key == 42

    def test_ids_persist_across_frames(self):
        tracker = PassthroughTracker(min_hits=1)
        for i in range(5):
            results = tracker.update([det(i, 0, 50 + i, 100, track_key=42)])
        assert results[0].track_key == 42

    def test_a_disappearing_id_is_retired(self):
        tracker = PassthroughTracker(min_hits=1, max_age=2)
        tracker.update([det(0, 0, 50, 100, track_key=42)])
        for _ in range(4):
            tracker.update([])
        assert 42 in tracker.removed_track_keys

    def test_untracked_detections_still_get_a_track(self):
        """A detection the model failed to id must not vanish."""
        tracker = PassthroughTracker(min_hits=1)
        results = tracker.update([det(0, 0, 50, 100, track_key=None)])
        assert len(results) == 1


# ------------------------------------------------------------------ motion
class TestMotionDetector:
    def _blank(self):
        return np.full((240, 320, 3), 60, dtype=np.uint8)

    def test_a_static_scene_reports_no_motion(self):
        detector = MOG2MotionDetector(start_frames=2, end_frames=3)
        for _ in range(20):
            result = detector.process(self._blank())
        assert not result.is_motion

    def test_a_moving_object_raises_motion(self):
        detector = MOG2MotionDetector(start_frames=2, end_frames=5,
                                      min_area_ratio=0.001)
        for _ in range(25):
            detector.process(self._blank())
        result = None
        for i in range(10):
            frame = self._blank()
            frame[80:180, 20 + i * 20: 120 + i * 20] = 240
            result = detector.process(frame)
        assert result is not None and result.is_motion
        assert result.regions

    def test_motion_requires_sustained_activity(self):
        """Hysteresis: one noisy frame is not an event."""
        detector = MOG2MotionDetector(start_frames=5, end_frames=5,
                                      min_area_ratio=0.001)
        for _ in range(25):
            detector.process(self._blank())
        frame = self._blank()
        frame[80:180, 40:140] = 240
        assert not detector.process(frame).is_motion

    def test_reset_clears_the_background_model(self):
        detector = MOG2MotionDetector()
        for _ in range(10):
            detector.process(self._blank())
        detector.reset()
        assert not detector.is_active

    def test_empty_frames_are_handled(self):
        detector = MOG2MotionDetector()
        assert not detector.process(np.array([])).is_motion
