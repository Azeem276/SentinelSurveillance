"""SQLAlchemy ORM models. Importing this package registers every table."""
from app.models.base import Base, TimestampMixin, utcnow
from app.models.detection import Detection, MotionEvent, Track
from app.models.enums import (
    AlertState, AlertType, EventSeverity, EventType, FaceReviewStatus, IdentityCategory,
    IdentityStatus, ProximityZone, RecognitionState, RecordingStatus, SourceStatus,
    SourceType, TrackStatus,
)
from app.models.event import Alert, SecurityEvent, SystemSetting
from app.models.identity import (
    Face, FaceEmbedding, FaceProfileSample, Identity, TemporaryIdentityExpiration,
)
from app.models.recording import RecordingSession
from app.models.source import VideoSource

__all__ = [
    "Base", "TimestampMixin", "utcnow",
    "VideoSource", "RecordingSession", "Track", "Detection", "MotionEvent",
    "Identity", "Face", "FaceEmbedding", "FaceProfileSample",
    "TemporaryIdentityExpiration",
    "SecurityEvent", "Alert", "SystemSetting",
    "SourceType", "SourceStatus", "RecordingStatus", "TrackStatus",
    "IdentityCategory", "IdentityStatus", "RecognitionState", "ProximityZone",
    "EventType", "EventSeverity", "AlertType", "AlertState", "FaceReviewStatus",
]
