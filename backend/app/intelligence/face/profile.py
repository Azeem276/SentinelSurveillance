"""Per-track face profile: many views of one person, one conclusion.

Why this exists
---------------
A single frame is a bad witness. Someone walking through the recognition zone
is seen dozens of times - turning their head, moving in and out of shadow,
getting closer - and the *set* of those observations identifies them far more
reliably than whichever frame happened to be processed last.

``FaceProfile`` gathers up to ``FACE_PROFILE_MAX_SAMPLES`` usable observations
for one track and deliberately spreads them across a grid of pose and lighting
buckets, so that a hundred near-identical frontal frames cannot crowd out the
three profile-view frames that actually carry new information. It then answers
one question - "who is this track?" - by polling every sample against the
recognition index at once and requiring a **majority with a margin**.

Two consequences matter for security:

* a genuine familiar seen at a new angle is still recognised, because only a
  fraction of the profile has to match, not the current frame;
* an unknown is declared unknown on the weight of many observations, so a
  single bad crop can no longer raise an alarm.

The collected samples are also what gets enrolled when an operator classifies
the person, so an identity is born with a multi-angle gallery rather than one
arbitrary crop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.core.config import get_settings
from app.intelligence.types import RecognitionMatch
from app.models.enums import RecognitionState

# Signed-yaw bucket edges: hard-left, left, frontal, right, hard-right.
YAW_EDGES = (-0.55, -0.2, 0.2, 0.55)
# Mean-luminance bucket edges: dim, normal, bright.
LIGHT_EDGES = (85.0, 170.0)

FAMILIAR_STATES = frozenset(
    {RecognitionState.PERMANENT_FAMILIAR, RecognitionState.TEMPORARY_FAMILIAR}
)


def yaw_bucket(yaw: float) -> int:
    for i, edge in enumerate(YAW_EDGES):
        if yaw < edge:
            return i
    return len(YAW_EDGES)


def light_bucket(brightness: float) -> int:
    for i, edge in enumerate(LIGHT_EDGES):
        if brightness < edge:
            return i
    return len(LIGHT_EDGES)


@dataclass(slots=True)
class FaceSample:
    """One usable view of a track's face."""

    embedding: np.ndarray
    quality: float
    yaw: float
    brightness: float
    face_pixels: int
    threshold_penalty: float
    frame_number: int
    timestamp: datetime
    # Full-resolution crop, kept so the best views can be enrolled later.
    crop: np.ndarray | None = None
    # Face box in ORIGINAL frame coordinates, for the stored review record.
    bbox: tuple[float, float, float, float] | None = None

    @property
    def bucket(self) -> tuple[int, int]:
        return yaw_bucket(self.yaw), light_bucket(self.brightness)


@dataclass(slots=True)
class ProfileVerdict:
    """What the whole profile says about who this track is."""

    state: RecognitionState | None          # None = not enough evidence yet
    identity_id: int | None = None
    label: str | None = None
    score: float = 0.0
    runner_up_score: float = 0.0
    support: float = 0.0                    # fraction of samples that agreed
    samples: int = 0
    reason: str = "insufficient_evidence"

    @property
    def conclusive(self) -> bool:
        return self.state is not None

    @property
    def is_familiar(self) -> bool:
        return self.state in FAMILIAR_STATES

    @property
    def is_unfamiliar(self) -> bool:
        return self.state is RecognitionState.UNFAMILIAR

    def to_dict(self) -> dict:
        return {
            "state": self.state.value if self.state else None,
            "identity_id": self.identity_id,
            "label": self.label,
            "score": round(self.score, 4),
            "runner_up_score": round(self.runner_up_score, 4),
            "support": round(self.support, 3),
            "samples": self.samples,
            "reason": self.reason,
        }


@dataclass
class FaceProfile:
    """A bucketed reservoir of face observations for one track."""

    max_samples: int = 0
    per_bucket: int = 0
    _buckets: dict[tuple[int, int], list[FaceSample]] = field(default_factory=dict)
    _total: int = 0
    _added_since_verdict: int = 0
    _last_verdict: ProfileVerdict | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        if self.max_samples <= 0:
            self.max_samples = settings.face_profile_max_samples
        if self.per_bucket <= 0:
            self.per_bucket = settings.face_profile_per_bucket

    # ---------------------------------------------------------- collecting
    @property
    def count(self) -> int:
        return self._total

    @property
    def bucket_coverage(self) -> int:
        """How many distinct pose/lighting conditions have been captured."""
        return len(self._buckets)

    @property
    def is_full(self) -> bool:
        return self._total >= self.max_samples

    @property
    def pending_since_verdict(self) -> int:
        return self._added_since_verdict

    def add(self, sample: FaceSample) -> bool:
        """File one observation. Returns True if it was kept.

        Within a bucket the best-quality samples win, so the profile improves
        as the person gets closer or turns into better light. A full profile
        still accepts a sample from a *new* pose by evicting the weakest
        member of the most over-represented bucket - angle coverage is worth
        more than another copy of the easiest view.
        """
        key = sample.bucket
        bucket = self._buckets.setdefault(key, [])

        if len(bucket) < self.per_bucket and self._total < self.max_samples:
            bucket.append(sample)
            self._total += 1
            self._added_since_verdict += 1
            return True

        if bucket:
            weakest = min(range(len(bucket)), key=lambda i: bucket[i].quality)
            if sample.quality > bucket[weakest].quality:
                bucket[weakest] = sample
                self._added_since_verdict += 1
                return True
            return False

        # New bucket on a full profile: take a slot from the biggest bucket.
        donor_key = max(
            (k for k in self._buckets if k != key),
            key=lambda k: len(self._buckets[k]),
            default=None,
        )
        if donor_key is not None and len(self._buckets[donor_key]) > 1:
            donor = self._buckets[donor_key]
            donor.pop(min(range(len(donor)), key=lambda i: donor[i].quality))
            bucket.append(sample)
            self._added_since_verdict += 1
            return True
        return False

    def samples(self) -> list[FaceSample]:
        return [s for bucket in self._buckets.values() for s in bucket]

    def gallery(self, limit: int | None = None) -> list[FaceSample]:
        """The best samples to enrol, spread across buckets.

        Round-robins the buckets rather than taking a global top-N, so an
        enrolled identity carries the angles the camera actually sees instead
        of N copies of its single easiest view.
        """
        if limit is None:
            limit = get_settings().face_profile_gallery_size
        ordered = [
            sorted(bucket, key=lambda s: s.quality, reverse=True)
            for bucket in self._buckets.values()
        ]
        out: list[FaceSample] = []
        depth = 0
        while len(out) < limit:
            took = False
            for bucket in ordered:
                if depth < len(bucket):
                    out.append(bucket[depth])
                    took = True
                    if len(out) >= limit:
                        break
            if not took:
                break
            depth += 1
        return out

    def best_sample(self) -> FaceSample | None:
        samples = self.samples()
        if not samples:
            return None
        return max(samples, key=lambda s: s.quality)

    def clear(self) -> None:
        self._buckets.clear()
        self._total = 0
        self._added_since_verdict = 0
        self._last_verdict = None

    # ------------------------------------------------------------ deciding
    def ready(self, *, min_samples: int | None = None) -> bool:
        if min_samples is None:
            min_samples = get_settings().face_profile_min_samples
        return self._total >= max(1, min_samples)

    def should_reverdict(self, *, every: int | None = None) -> bool:
        """Avoid re-polling the index on every single frame."""
        if self._last_verdict is None:
            return True
        if every is None:
            every = get_settings().face_profile_reverdict_every
        return self._added_since_verdict >= max(1, every)

    @property
    def last_verdict(self) -> ProfileVerdict | None:
        return self._last_verdict

    def verdict(
        self,
        index,
        *,
        threshold: float,
        margin: float | None = None,
        support_ratio: float | None = None,
        min_samples: int | None = None,
    ) -> ProfileVerdict:
        """Decide who this track is from every sample at once.

        A familiar verdict needs three things, not one lucky frame:
          1. enough observations to be worth judging,
          2. a majority of them agreeing on the same identity,
          3. that identity beating the runner-up by ``margin``.

        An unfamiliar verdict needs the same majority in the other direction,
        which is what stops one bad-angle crop from raising an alarm.
        """
        settings = get_settings()
        if margin is None:
            margin = settings.face_match_margin
        if support_ratio is None:
            support_ratio = settings.face_profile_support_ratio

        samples = self.samples()
        self._added_since_verdict = 0

        if not self.ready(min_samples=min_samples):
            self._last_verdict = ProfileVerdict(
                state=None, samples=len(samples), reason="insufficient_samples"
            )
            return self._last_verdict

        matches: list[RecognitionMatch] = index.recognize_batch(
            [s.embedding for s in samples], threshold=threshold
        )

        votes: dict[int, list[float]] = {}
        labels: dict[int, str] = {}
        states: dict[int, RecognitionState] = {}
        unfamiliar = 0
        unusable = 0

        for sample, match in zip(samples, matches, strict=False):
            # Each sample carries its own bar: an angled crop has to score
            # higher than a frontal one before it counts as a match.
            required = threshold + sample.threshold_penalty
            if match.state is RecognitionState.FACE_UNRECOGNIZABLE:
                unusable += 1
                continue
            if match.identity_id is None or match.score < required:
                unfamiliar += 1
                continue
            votes.setdefault(match.identity_id, []).append(match.score)
            labels[match.identity_id] = match.label or ""
            states[match.identity_id] = match.state

        counted = len(samples) - unusable
        if counted <= 0:
            self._last_verdict = ProfileVerdict(
                state=None, samples=len(samples), reason="no_usable_samples"
            )
            return self._last_verdict

        if votes:
            ranked = sorted(
                votes.items(),
                key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))),
                reverse=True,
            )
            best_id, best_scores = ranked[0]
            best_mean = float(np.mean(best_scores))
            runner_mean = float(np.mean(ranked[1][1])) if len(ranked) > 1 else 0.0
            support = len(best_scores) / counted

            if support >= support_ratio and (best_mean - runner_mean) >= margin:
                self._last_verdict = ProfileVerdict(
                    state=states[best_id],
                    identity_id=best_id,
                    label=labels[best_id] or None,
                    score=round(best_mean, 4),
                    runner_up_score=round(runner_mean, 4),
                    support=support,
                    samples=len(samples),
                    reason="profile_match",
                )
                return self._last_verdict

            # Agreed often enough, but two identities are too close to call.
            # Reporting that honestly is what keeps a look-alike from
            # inheriting a trusted person's label.
            if support >= support_ratio:
                self._last_verdict = ProfileVerdict(
                    state=None,
                    identity_id=best_id,
                    label=labels[best_id] or None,
                    score=round(best_mean, 4),
                    runner_up_score=round(runner_mean, 4),
                    support=support,
                    samples=len(samples),
                    reason="ambiguous_margin",
                )
                return self._last_verdict

        unfamiliar_support = unfamiliar / counted
        if unfamiliar_support >= support_ratio:
            best_seen = max(
                (
                    m.score
                    for m in matches
                    if m.state is not RecognitionState.FACE_UNRECOGNIZABLE
                ),
                default=0.0,
            )
            self._last_verdict = ProfileVerdict(
                state=RecognitionState.UNFAMILIAR,
                score=round(float(best_seen), 4),
                support=unfamiliar_support,
                samples=len(samples),
                reason="profile_unfamiliar",
            )
            return self._last_verdict

        self._last_verdict = ProfileVerdict(
            state=None,
            support=unfamiliar_support,
            samples=len(samples),
            reason="split_evidence",
        )
        return self._last_verdict


def new_sample(
    *,
    embedding: np.ndarray,
    quality,
    frame_number: int,
    timestamp: datetime | None = None,
    crop: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> FaceSample:
    """Build a sample from an embedding plus its :class:`FaceQuality`."""
    return FaceSample(
        embedding=np.asarray(embedding, dtype=np.float32).reshape(-1),
        quality=float(getattr(quality, "score", 0.0)),
        yaw=float(getattr(quality, "yaw", 0.0)),
        brightness=float(getattr(quality, "brightness", 128.0)),
        face_pixels=int(getattr(quality, "face_pixels", 0)),
        threshold_penalty=float(getattr(quality, "threshold_penalty", 0.0)),
        frame_number=frame_number,
        timestamp=timestamp or datetime.now(timezone.utc),
        crop=crop,
        bbox=bbox,
    )
