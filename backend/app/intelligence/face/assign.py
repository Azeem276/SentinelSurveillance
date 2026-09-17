"""Deciding which face belongs to which person.

The naive rule - "take the biggest face inside this person's box" - is wrong
in exactly the situation that matters most. In a crowd, boxes overlap, and the
biggest face inside person A's box frequently belongs to person B standing
closer to the camera. That produces a *confident wrong identity*, which is far
worse than no identity at all: a stranger can inherit a trusted person's label
and then never trigger an alarm.

This module replaces that heuristic with a global, exclusive assignment:

* a face must genuinely sit inside the person box (``containment``),
* it must sit where a head sits - near the top of the box, not the waist,
* it must be plausibly sized relative to the body,
* and each face may be claimed by **at most one** person, each person by at
  most one face.

Assignment is greedy over the combined score, highest first. For the handful
of people visible in one frame this is equivalent to the optimal matching in
practice, and it is O(f*p log(f*p)) with no dependency on scipy.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.intelligence.types import BBox, DetectedFace

# Where a head is expected to sit inside a person box, as a fraction of box
# height measured from the top. Full-body boxes put it in the first ~20%;
# head-and-shoulders boxes put it lower, so the window is generous.
HEAD_ZONE_END = 0.45
# Plausible face-height to person-height ratios. Outside this the face is
# probably a bystander's (too big) or a false positive (too small).
MIN_FACE_RATIO = 0.04
MAX_FACE_RATIO = 0.85


@dataclass(slots=True)
class FaceCandidate:
    """One person box competing for faces."""

    track_key: int
    bbox: BBox


@dataclass(slots=True)
class Assignment:
    track_key: int
    face: DetectedFace
    score: float
    containment: float


def containment(face: BBox, person: BBox) -> float:
    """Fraction of the face box that lies inside the person box."""
    area = face.area
    if area <= 0:
        return 0.0
    ix1, iy1 = max(face.x1, person.x1), max(face.y1, person.y1)
    ix2, iy2 = min(face.x2, person.x2), min(face.y2, person.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return (iw * ih) / area


def _vertical_prior(face: BBox, person: BBox) -> float:
    """1.0 when the face sits where a head should, decaying downwards."""
    height = person.height
    if height <= 0:
        return 0.0
    relative = (face.center[1] - person.y1) / height
    if relative < 0.0:
        # Above the box entirely - a detector artefact, not a head.
        return max(0.0, 1.0 + relative * 4.0)
    if relative <= HEAD_ZONE_END:
        return 1.0
    return max(0.0, 1.0 - (relative - HEAD_ZONE_END) / (1.0 - HEAD_ZONE_END))


def _size_prior(face: BBox, person: BBox) -> float:
    """Penalise faces that are implausibly large or small for this body."""
    height = person.height
    if height <= 0:
        return 0.0
    ratio = face.height / height
    if MIN_FACE_RATIO <= ratio <= MAX_FACE_RATIO:
        return 1.0
    if ratio > MAX_FACE_RATIO:
        return max(0.0, 1.0 - (ratio - MAX_FACE_RATIO) * 2.0)
    return max(0.0, ratio / MIN_FACE_RATIO)


def score_pair(face: DetectedFace, person: BBox) -> tuple[float, float]:
    """Return ``(score, containment)`` for one face/person pairing."""
    inside = containment(face.bbox, person)
    if inside <= 0.0:
        return 0.0, 0.0
    vertical = _vertical_prior(face.bbox, person)
    size = _size_prior(face.bbox, person)
    confidence = max(0.0, min(1.0, face.confidence))
    score = inside * (0.15 + 0.85 * vertical) * (0.25 + 0.75 * size) * (
        0.5 + 0.5 * confidence
    )
    return score, inside


def assign_faces(
    faces: list[DetectedFace],
    candidates: list[FaceCandidate],
    *,
    min_containment: float | None = None,
) -> dict[int, Assignment]:
    """Match detected faces to tracked people, one-to-one.

    Returns a mapping of ``track_key -> Assignment``. Tracks that got no
    defensible face are simply absent, which the caller must treat as
    "no face this frame" rather than guessing.
    """
    if not faces or not candidates:
        return {}
    if min_containment is None:
        min_containment = get_settings().face_min_containment

    pairs: list[tuple[float, float, int, int]] = []
    for ci, candidate in enumerate(candidates):
        for fi, face in enumerate(faces):
            score, inside = score_pair(face, candidate.bbox)
            if inside < min_containment or score <= 0.0:
                continue
            pairs.append((score, inside, ci, fi))

    pairs.sort(key=lambda p: p[0], reverse=True)

    taken_tracks: set[int] = set()
    taken_faces: set[int] = set()
    out: dict[int, Assignment] = {}
    for score, inside, ci, fi in pairs:
        if ci in taken_tracks or fi in taken_faces:
            continue
        taken_tracks.add(ci)
        taken_faces.add(fi)
        candidate = candidates[ci]
        out[candidate.track_key] = Assignment(
            track_key=candidate.track_key,
            face=faces[fi],
            score=round(score, 4),
            containment=round(inside, 4),
        )
    return out
