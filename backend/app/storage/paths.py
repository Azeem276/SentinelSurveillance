"""Filesystem path resolution, confined to the configured storage roots.

Every path that reaches the filesystem from an API client, a database row or a
generated filename goes through here. The rule is simple and absolute: a
caller supplies a path *relative to a root*, and this module is the only thing
that turns it into an absolute path - after checking that it still lands
inside that root.

That check is the point. Face crops, recordings and source videos are all
addressed by strings that came from somewhere else, and ``../`` in any of them
would otherwise let a request read or write arbitrary files.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import StorageError

# Containers OpenCV can actually open, and that we are willing to store.
ALLOWED_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm", ".wmv"}
)
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

_UNSAFE = re.compile(r"[^A-Za-z0-9]+")


# ------------------------------------------------------------------- roots
def video_root() -> Path:
    return Path(get_settings().video_storage_path)


def face_root() -> Path:
    return Path(get_settings().face_storage_path)


def recording_root() -> Path:
    return Path(get_settings().recording_path)


def snapshot_root() -> Path:
    return Path(get_settings().snapshot_path)


def event_asset_root() -> Path:
    return Path(get_settings().event_asset_path)


# ------------------------------------------------------------------ helpers
def slugify(text: str, *, fallback: str = "item") -> str:
    """Reduce arbitrary text to a filesystem-safe token.

    Used for filenames built from operator-supplied names (source uids,
    identity labels), so it must never emit a separator or a dot.
    """
    cleaned = _UNSAFE.sub("_", str(text or "")).strip("_")
    return cleaned or fallback


def ensure_parent(path: Path) -> Path:
    """Create the directory a file is about to be written into."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_join(root: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` inside ``root``, refusing to escape it.

    Rejects absolute paths, drive letters and any ``..`` that would climb out,
    by comparing the *resolved* result against the resolved root rather than
    by pattern-matching the input - which is the only way to catch traversal
    hidden behind symlinks or mixed separators.
    """
    root = Path(root).resolve()
    candidate = Path(str(relative).replace("\\", "/"))

    if candidate.is_absolute() or candidate.drive:
        raise StorageError(f"absolute paths are not accepted: {relative!r}")

    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise StorageError(f"path escapes its storage root: {relative!r}")
    return resolved


def relative_to_root(path: Path, root: Path) -> str:
    """Express an absolute path as the root-relative string we persist.

    Databases store relative paths so that moving or remounting the storage
    directory does not invalidate every row.
    """
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise StorageError(f"{path} is not inside {root}") from exc


def _resolve_checked(
    root: Path, relative: str | Path, *, allowed: frozenset[str], kind: str
) -> Path:
    resolved = safe_join(root, relative)
    if resolved.suffix.lower() not in allowed:
        raise StorageError(f"Unsupported {kind} type: {resolved.suffix or 'none'}")
    return resolved


# --------------------------------------------------------------- resolvers
def resolve_video(uri: str | Path) -> Path:
    """Absolute path of a source clip inside VIDEO_STORAGE_PATH."""
    return _resolve_checked(
        video_root(), uri, allowed=ALLOWED_VIDEO_SUFFIXES, kind="video"
    )


def resolve_face(path: str | Path) -> Path:
    """Absolute path of a stored face crop inside FACE_STORAGE_PATH."""
    return _resolve_checked(
        face_root(), path, allowed=ALLOWED_IMAGE_SUFFIXES, kind="image"
    )


def resolve_recording(path: str | Path) -> Path:
    """Absolute path of a recorded clip inside RECORDING_PATH."""
    return _resolve_checked(
        recording_root(), path, allowed=ALLOWED_VIDEO_SUFFIXES, kind="video"
    )


def validate_video_upload(filename: str) -> str:
    """Sanitise an uploaded filename down to a bare, allow-listed name.

    Any directory component is discarded rather than rejected, so an upload
    cannot smuggle a path, and the stem is slugified so the stored name
    contains nothing that could be reinterpreted as one.
    """
    name = Path(str(filename or "").replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise StorageError(f"Unsupported video type: {suffix or 'none'}")
    stem = slugify(Path(name).stem, fallback="upload")
    return f"{stem}{suffix}"
