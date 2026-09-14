"""Shared test fixtures.

The suite runs against a file-backed SQLite database so it needs no server,
and against small synthetic media so it needs no large fixtures. The tests
that require the real model weights are marked ``requires_models`` and skip
cleanly when they are not installed.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _test_environment(tmp_path_factory: pytest.TempPathFactory):
    """Point every configured path at a throwaway directory before imports."""
    root = tmp_path_factory.mktemp("sentinel")
    os.environ.update(
        {
            "ENVIRONMENT": "test",
            "DATABASE_URL": f"sqlite:///{(root / 'test.db').as_posix()}",
            "STORAGE_PATH": str(root / "storage"),
            "RECORDING_PATH": str(root / "storage" / "recordings"),
            "FACE_STORAGE_PATH": str(root / "storage" / "faces"),
            "SNAPSHOT_PATH": str(root / "storage" / "snapshots"),
            "EVENT_ASSET_PATH": str(root / "storage" / "events"),
            "VIDEO_STORAGE_PATH": str(root / "videos"),
            "FACE_DATASET_PATH": str(root / "faces_dataset"),
            "AUTO_DOWNLOAD_MODELS": "false",
            "AI_DEVICE": "cpu",
            "LOG_LEVEL": "WARNING",
            "RECORDING_MAX_MINUTES": "30",
        }
    )

    from app.core.config import reload_settings

    settings = reload_settings()
    settings.ensure_directories()
    yield settings


@pytest.fixture(scope="session")
def engine(_test_environment):
    from app.db.session import configure_engine, dispose_engine, get_engine
    from app.models import Base

    configure_engine(_test_environment.database_url)
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield engine
    dispose_engine()


@pytest.fixture()
def session(engine):
    """A session wrapped in a transaction that is rolled back after each test."""
    from sqlalchemy.orm import sessionmaker

    connection = engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset process-wide singletons so tests cannot leak state into each other."""
    from app.alerts.engine import reset_alert_engine
    from app.events.bus import get_bus
    from app.intelligence.face.index import reset_index

    reset_index()
    reset_alert_engine()
    get_bus().clear()
    yield
    reset_index()
    reset_alert_engine()
    get_bus().clear()


# ------------------------------------------------------------------ helpers
@pytest.fixture()
def make_source(session):
    from app.models.enums import SourceType
    from app.models.source import VideoSource

    counter = {"n": 0}

    def _make(**overrides):
        counter["n"] += 1
        defaults = dict(
            uid=f"camera_{counter['n']:02d}",
            name=f"Camera {counter['n']:02d}",
            type=SourceType.FILE,
            uri=f"camera_{counter['n']:02d}.mp4",
            proximity_a=10.0,
            proximity_b=3.0,
            alert_policy={},
            calibration={},
        )
        defaults.update(overrides)
        source = VideoSource(**defaults)
        session.add(source)
        session.flush()
        return source

    return _make


@pytest.fixture()
def make_identity(session):
    from app.models.enums import IdentityCategory
    from app.repositories.identity_repository import IdentityRepository

    def _make(name=None, category=IdentityCategory.PERMANENT, retention_days=None,
              first_detected_at=None):
        return IdentityRepository(session).create(
            category=category,
            first_detected_at=first_detected_at or datetime.now(timezone.utc),
            display_name=name,
            retention_days=retention_days,
        )

    return _make


@pytest.fixture()
def unit_vector():
    def _make(seed: int, dim: int = 128) -> np.ndarray:
        vec = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
        return vec / float(np.linalg.norm(vec))

    return _make


@pytest.fixture()
def sample_video(_test_environment):
    """A small real MP4 the video/recording layers can actually open."""
    import cv2

    path = Path(_test_environment.video_storage_path) / "unit_test_clip.mp4"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    assert writer.isOpened(), "OpenCV cannot write MP4 in this environment"
    rng = np.random.default_rng(3)
    for i in range(30):
        frame = np.full((120, 160, 3), 40, dtype=np.uint8)
        x = 10 + i * 4
        frame[40:90, x: x + 24] = (200, 180, 160)
        frame = np.clip(
            frame.astype(np.int16) + rng.normal(0, 2, frame.shape), 0, 255
        ).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture()
def models_available() -> bool:
    from app.intelligence.model_registry import is_installed

    return all(
        is_installed(k) for k in ("object_detector", "face_detector", "face_embedder")
    )


@pytest.fixture()
def past() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=1)
