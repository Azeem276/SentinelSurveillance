"""Application-level exceptions mapped to HTTP responses in app.api.errors."""
from __future__ import annotations


class SentinelError(Exception):
    """Base class for all domain errors."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(SentinelError):
    status_code = 404
    code = "not_found"


class ValidationError(SentinelError):
    status_code = 422
    code = "validation_error"


class ConflictError(SentinelError):
    status_code = 409
    code = "conflict"


class PreconditionError(SentinelError):
    """Operation not allowed in the current system state (e.g. surveillance off)."""

    status_code = 409
    code = "precondition_failed"


class StorageError(SentinelError):
    status_code = 400
    code = "storage_error"


class ModelNotAvailableError(SentinelError):
    status_code = 503
    code = "model_unavailable"
