"""In-memory face recognition index.

Requirements it satisfies:

* One identity owns many embeddings; the best-matching one wins.
* Rebuilding is cheap and thread-safe, so classifying, renaming, deleting or
  expiring an identity takes effect on the very next frame.
* Expired temporary identities stop matching immediately - they are filtered
  both at load time and again at query time against the current clock.

The store is a single normalised matrix, so a query is one matrix-vector
product regardless of how many identities are enrolled.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.intelligence.face.base import FaceRecognizer
from app.intelligence.types import RecognitionMatch
from app.models.enums import IdentityCategory, RecognitionState

log = get_logger(__name__)


@dataclass(slots=True)
class IndexEntry:
    """One enrolled identity as seen by the matcher."""

    identity_id: int
    label: str
    category: IdentityCategory
    expires_at: datetime | None

    def is_active(self, now: datetime | None = None) -> bool:
        if self.category is IdentityCategory.PERMANENT:
            return True
        if self.expires_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires > now


class FaceRecognitionIndex(FaceRecognizer):
    """Cosine-similarity matcher over enrolled identity embeddings."""

    def __init__(self, *, threshold: float | None = None, dim: int = 128) -> None:
        self.dim = dim
        self._threshold = threshold
        self._lock = threading.RLock()
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._owners: list[IndexEntry] = []
        self._entries: dict[int, IndexEntry] = {}
        self._version = 0

    # --------------------------------------------------------------- build
    def rebuild(self, records: list[tuple[IndexEntry, np.ndarray]]) -> None:
        """Replace the whole index. ``records`` is (entry, embedding) pairs."""
        now = datetime.now(timezone.utc)
        vectors: list[np.ndarray] = []
        owners: list[IndexEntry] = []
        entries: dict[int, IndexEntry] = {}

        for entry, vector in records:
            if not entry.is_active(now):
                continue
            vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            if vec.size != self.dim:
                log.warning("embedding_dim_mismatch", identity_id=entry.identity_id,
                            expected=self.dim, got=int(vec.size))
                continue
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                continue
            vectors.append(vec / norm)
            owners.append(entry)
            entries[entry.identity_id] = entry

        with self._lock:
            self._matrix = (
                np.vstack(vectors).astype(np.float32)
                if vectors
                else np.zeros((0, self.dim), dtype=np.float32)
            )
            self._owners = owners
            self._entries = entries
            self._version += 1
        log.info(
            "recognition_index_rebuilt",
            identities=len(entries),
            embeddings=len(owners),
            version=self._version,
        )

    def clear(self) -> None:
        self.rebuild([])

    # --------------------------------------------------------------- query
    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        return get_settings().face_recognition_threshold

    def recognize(
        self, embedding: np.ndarray, *, threshold: float | None = None
    ) -> RecognitionMatch:
        """Match one embedding. Returns UNFAMILIAR when nothing clears the bar.

        Callers must only reach this method with a quality-approved face; an
        UNFAMILIAR result therefore means "compared and matched nobody", which
        is the only condition that may justify an alarm.
        """
        limit = self.threshold if threshold is None else threshold
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return RecognitionMatch(state=RecognitionState.FACE_UNRECOGNIZABLE)
        vec = vec / norm

        with self._lock:
            matrix = self._matrix
            owners = self._owners
            if matrix.shape[0] == 0 or vec.size != matrix.shape[1]:
                return RecognitionMatch(state=RecognitionState.UNFAMILIAR)
            scores = matrix @ vec

            now = datetime.now(timezone.utc)
            # Best score per identity, skipping any that expired since build.
            best_by_identity: dict[int, float] = {}
            for position, score in enumerate(scores):
                entry = owners[position]
                if not entry.is_active(now):
                    continue
                value = float(score)
                if value > best_by_identity.get(entry.identity_id, -2.0):
                    best_by_identity[entry.identity_id] = value

            if not best_by_identity:
                return RecognitionMatch(state=RecognitionState.UNFAMILIAR)

            ranked = sorted(best_by_identity.items(), key=lambda kv: kv[1], reverse=True)
            best_id, best_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            entry = self._entries[best_id]

        if best_score < limit:
            return RecognitionMatch(
                state=RecognitionState.UNFAMILIAR,
                score=round(best_score, 4),
                runner_up_score=round(float(runner_up), 4),
            )

        state = (
            RecognitionState.PERMANENT_FAMILIAR
            if entry.category is IdentityCategory.PERMANENT
            else RecognitionState.TEMPORARY_FAMILIAR
        )
        return RecognitionMatch(
            state=state,
            identity_id=entry.identity_id,
            label=entry.label,
            score=round(best_score, 4),
            runner_up_score=round(float(runner_up), 4),
        )

    def recognize_batch(
        self, embeddings: list[np.ndarray], *, threshold: float | None = None
    ) -> list[RecognitionMatch]:
        """Match many embeddings in one matrix multiply.

        A face profile holds up to a hundred views of one person; polling them
        one at a time would repeat the same gather over the gallery a hundred
        times. Here the whole profile is matched against the whole index in a
        single ``(n, d) @ (d, m)`` product, which is what makes profile-level
        identification affordable at frame rate.
        """
        if not embeddings:
            return []
        limit = self.threshold if threshold is None else threshold

        stacked = np.vstack(
            [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings]
        )
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        usable = (norms > 1e-8).reshape(-1)
        stacked = stacked / np.where(norms < 1e-8, 1.0, norms)

        with self._lock:
            matrix = self._matrix
            owners = self._owners
            entries = dict(self._entries)

        if matrix.shape[0] == 0 or stacked.shape[1] != matrix.shape[1]:
            return [
                RecognitionMatch(
                    state=RecognitionState.UNFAMILIAR
                    if usable[i]
                    else RecognitionState.FACE_UNRECOGNIZABLE
                )
                for i in range(stacked.shape[0])
            ]

        scores = stacked @ matrix.T          # (samples, gallery vectors)
        now = datetime.now(timezone.utc)
        active = np.array([owner.is_active(now) for owner in owners], dtype=bool)
        if not active.any():
            return [RecognitionMatch(state=RecognitionState.UNFAMILIAR)] * scores.shape[0]

        # Collapse gallery columns down to one best score per identity.
        identity_ids = sorted({owners[i].identity_id for i in np.flatnonzero(active)})
        columns = {
            ident: np.flatnonzero(
                active & np.array([o.identity_id == ident for o in owners], dtype=bool)
            )
            for ident in identity_ids
        }
        per_identity = np.stack(
            [scores[:, cols].max(axis=1) for ident, cols in columns.items()], axis=1
        )

        out: list[RecognitionMatch] = []
        for row in range(per_identity.shape[0]):
            if not usable[row]:
                out.append(RecognitionMatch(state=RecognitionState.FACE_UNRECOGNIZABLE))
                continue
            values = per_identity[row]
            order = np.argsort(values)[::-1]
            best_idx = int(order[0])
            best_score = float(values[best_idx])
            runner_up = float(values[int(order[1])]) if values.size > 1 else 0.0
            if best_score < limit:
                out.append(
                    RecognitionMatch(
                        state=RecognitionState.UNFAMILIAR,
                        score=round(best_score, 4),
                        runner_up_score=round(runner_up, 4),
                    )
                )
                continue
            entry = entries[identity_ids[best_idx]]
            out.append(
                RecognitionMatch(
                    state=(
                        RecognitionState.PERMANENT_FAMILIAR
                        if entry.category is IdentityCategory.PERMANENT
                        else RecognitionState.TEMPORARY_FAMILIAR
                    ),
                    identity_id=entry.identity_id,
                    label=entry.label,
                    score=round(best_score, 4),
                    runner_up_score=round(runner_up, 4),
                )
            )
        return out

    # ---------------------------------------------------------- inspection
    @property
    def size(self) -> int:
        with self._lock:
            return int(self._matrix.shape[0])

    @property
    def identity_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def has_identity(self, identity_id: int) -> bool:
        with self._lock:
            entry = self._entries.get(identity_id)
        return entry is not None and entry.is_active()

    def stats(self) -> dict:
        with self._lock:
            permanent = sum(
                1 for e in self._entries.values() if e.category is IdentityCategory.PERMANENT
            )
            return {
                "identities": len(self._entries),
                "permanent": permanent,
                "temporary": len(self._entries) - permanent,
                "embeddings": int(self._matrix.shape[0]),
                "threshold": self.threshold,
                "version": self._version,
            }


# Process-wide index shared by every source pipeline.
_index: FaceRecognitionIndex | None = None
_index_lock = threading.Lock()


def get_index() -> FaceRecognitionIndex:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = FaceRecognitionIndex()
    return _index


def reset_index() -> None:
    """Used by tests to get a clean matcher."""
    global _index
    with _index_lock:
        _index = None
