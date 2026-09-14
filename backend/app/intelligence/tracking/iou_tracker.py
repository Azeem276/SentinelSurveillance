"""IoUTracker - a ByteTrack-style association tracker with no heavy deps.

Two roles:

1. It is the tracker used whenever the detector cannot assign ids itself
   (and it is what the unit tests exercise, since it needs no model weights).
2. It implements the same two-stage association idea as ByteTrack: high
   confidence detections are matched first, then low confidence ones are used
   to keep existing tracks alive through partial occlusion.

Tracks survive ``max_age`` frames without a match, so a person who is briefly
occluded keeps the same track_key - and therefore the same cached identity.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.intelligence.tracking.base import ObjectTracker
from app.intelligence.types import BBox, ObjectDetection, TrackedObject


class _Track:
    __slots__ = (
        "track_key", "bbox", "object_class", "confidence", "age", "hits",
        "time_since_update", "is_confirmed", "velocity",
    )

    def __init__(self, track_key: int, det: ObjectDetection, min_hits: int = 1) -> None:
        self.track_key = track_key
        self.bbox = det.bbox
        self.object_class = det.object_class
        self.confidence = det.confidence
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        # With min_hits=1 the very first observation already confirms.
        self.is_confirmed = self.hits >= min_hits
        self.velocity = (0.0, 0.0)

    def update(self, det: ObjectDetection, min_hits: int) -> None:
        px, py = self.bbox.center
        cx, cy = det.bbox.center
        self.velocity = (cx - px, cy - py)
        self.bbox = det.bbox
        self.confidence = det.confidence
        # A class flip on a confident detection is trusted; otherwise keep the
        # established class so one bad frame does not relabel a person a dog.
        if det.confidence >= self.confidence or det.object_class == self.object_class:
            self.object_class = det.object_class
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= min_hits:
            self.is_confirmed = True

    def predict(self) -> BBox:
        """Constant-velocity guess used for association while unmatched."""
        vx, vy = self.velocity
        b = self.bbox
        return BBox(b.x1 + vx, b.y1 + vy, b.x2 + vx, b.y2 + vy)

    def mark_missed(self) -> None:
        self.age += 1
        self.time_since_update += 1

    def snapshot(self) -> TrackedObject:
        return TrackedObject(
            track_key=self.track_key,
            bbox=self.bbox,
            object_class=self.object_class,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=self.is_confirmed,
        )


class IoUTracker(ObjectTracker):
    name = "iou-bytetrack"

    def __init__(
        self,
        *,
        iou_threshold: float | None = None,
        max_age: int | None = None,
        min_hits: int | None = None,
        high_confidence: float = 0.55,
        low_iou_threshold: float = 0.2,
        start_id: int = 1,
    ) -> None:
        settings = get_settings()
        self.iou_threshold = settings.track_iou_threshold if iou_threshold is None else iou_threshold
        self.max_age = settings.track_max_age if max_age is None else max_age
        self.min_hits = settings.track_min_hits if min_hits is None else min_hits
        self.high_confidence = high_confidence
        self.low_iou_threshold = low_iou_threshold
        self._next_id = start_id
        self._tracks: list[_Track] = []
        self._removed: list[int] = []

    # ------------------------------------------------------------ matching
    def _match(
        self,
        tracks: list[_Track],
        detections: list[ObjectDetection],
        threshold: float,
    ) -> tuple[list[tuple[_Track, ObjectDetection]], list[_Track], list[ObjectDetection]]:
        """Greedy highest-IoU-first association (same class preferred)."""
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            predicted = track.predict() if track.time_since_update > 0 else track.bbox
            for di, det in enumerate(detections):
                iou = max(predicted.iou(det.bbox), track.bbox.iou(det.bbox))
                if iou < threshold:
                    continue
                # Same-class matches win ties.
                score = iou + (0.15 if det.object_class == track.object_class else 0.0)
                pairs.append((score, ti, di))

        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        matched: list[tuple[_Track, ObjectDetection]] = []
        for _score, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matched.append((tracks[ti], detections[di]))

        unmatched_t = [t for i, t in enumerate(tracks) if i not in used_t]
        unmatched_d = [d for i, d in enumerate(detections) if i not in used_d]
        return matched, unmatched_t, unmatched_d

    def update(self, detections: list[ObjectDetection]) -> list[TrackedObject]:
        high = [d for d in detections if d.confidence >= self.high_confidence]
        low = [d for d in detections if d.confidence < self.high_confidence]

        for track in self._tracks:
            track.age += 1

        # Stage 1: confident detections against all tracks.
        matched, unmatched_tracks, unmatched_high = self._match(
            self._tracks, high, self.iou_threshold
        )
        for track, det in matched:
            track.update(det, self.min_hits)

        # Stage 2: rescue remaining tracks with low-confidence detections.
        matched_low, still_unmatched, _unused_low = self._match(
            unmatched_tracks, low, self.low_iou_threshold
        )
        for track, det in matched_low:
            track.update(det, self.min_hits)

        for track in still_unmatched:
            track.mark_missed()

        # Only confident, unmatched detections spawn new tracks.
        for det in unmatched_high:
            self._tracks.append(_Track(self._next_id, det, self.min_hits))
            self._next_id += 1

        alive: list[_Track] = []
        for track in self._tracks:
            if track.time_since_update > self.max_age:
                if track.is_confirmed:
                    self._removed.append(track.track_key)
            else:
                alive.append(track)
        self._tracks = alive

        return [t.snapshot() for t in self._tracks if t.is_confirmed and t.time_since_update == 0]

    def reset(self) -> None:
        self._tracks = []
        self._removed = []

    @property
    def active_tracks(self) -> list[TrackedObject]:
        return [t.snapshot() for t in self._tracks if t.is_confirmed]

    @property
    def removed_track_keys(self) -> list[int]:
        """Track ids that have ended and not yet been drained.

        Accumulates across updates so a caller that does not poll on every
        single frame cannot miss a track ending.
        """
        return list(self._removed)

    def drain_removed(self) -> list[int]:
        keys = list(self._removed)
        self._removed = []
        return keys


class PassthroughTracker(ObjectTracker):
    """Used when the detector already assigns ids (Ultralytics ByteTrack).

    It still owns the lifecycle bookkeeping (age, confirmation, removal) so the
    pipeline sees identical semantics whichever tracker is active.
    """

    name = "passthrough"

    def __init__(self, *, max_age: int | None = None, min_hits: int = 1) -> None:
        settings = get_settings()
        self.max_age = settings.track_max_age if max_age is None else max_age
        self.min_hits = min_hits
        self._tracks: dict[int, TrackedObject] = {}
        self._removed: list[int] = []
        # Same confirmation policy, disjoint id range so keys never collide.
        self._fallback = IoUTracker(min_hits=self.min_hits, max_age=self.max_age,
                                    start_id=1_000_000)

    def update(self, detections: list[ObjectDetection]) -> list[TrackedObject]:
        with_ids = [d for d in detections if d.track_key is not None]
        without_ids = [d for d in detections if d.track_key is None]

        seen: set[int] = set()
        for det in with_ids:
            key = int(det.track_key)  # type: ignore[arg-type]
            seen.add(key)
            existing = self._tracks.get(key)
            if existing is None:
                self._tracks[key] = TrackedObject(
                    track_key=key,
                    bbox=det.bbox,
                    object_class=det.object_class,
                    confidence=det.confidence,
                    is_confirmed=self.min_hits <= 1,
                )
            else:
                existing.bbox = det.bbox
                existing.object_class = det.object_class
                existing.confidence = det.confidence
                existing.hits += 1
                existing.age += 1
                existing.time_since_update = 0
                if existing.hits >= self.min_hits:
                    existing.is_confirmed = True

        for key, track in list(self._tracks.items()):
            if key in seen:
                continue
            track.age += 1
            track.time_since_update += 1
            if track.time_since_update > self.max_age:
                if track.is_confirmed:
                    self._removed.append(key)
                del self._tracks[key]

        results = [
            t for t in self._tracks.values() if t.is_confirmed and t.time_since_update == 0
        ]

        # Detections the detector failed to id still deserve a track.
        if without_ids:
            results.extend(self._fallback.update(without_ids))
        self._removed.extend(self._fallback.drain_removed())
        return results

    def reset(self) -> None:
        self._tracks = {}
        self._removed = []
        self._fallback.reset()

    @property
    def active_tracks(self) -> list[TrackedObject]:
        return [t for t in self._tracks.values() if t.is_confirmed]

    @property
    def removed_track_keys(self) -> list[int]:
        return list(self._removed)

    def drain_removed(self) -> list[int]:
        keys = list(self._removed)
        self._removed = []
        return keys
