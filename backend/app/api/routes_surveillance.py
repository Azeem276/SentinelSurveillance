"""Global surveillance and intelligence control."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import DbSession, Manager
from app.core.hardware import get_hardware
from app.db.session import check_connection
from app.repositories.event_repository import AlertRepository
from app.repositories.identity_repository import FaceRepository
from app.schemas.common import OperationResult
from app.schemas.intelligence import SystemState

router = APIRouter(prefix="/api", tags=["surveillance"])


def _system_state(session, manager) -> SystemState:
    state = manager.system_state()
    connected, _ = check_connection()
    return SystemState(
        **state,
        active_alerts=len(AlertRepository(session).active_alerts()),
        pending_review=FaceRepository(session).pending_review_count(),
        database_connected=connected,
        device=get_hardware().device,
    )


@router.get("/system/state", response_model=SystemState)
def system_state(session: DbSession, manager: Manager):
    return _system_state(session, manager)


@router.post("/surveillance/start", response_model=OperationResult)
def start_surveillance(manager: Manager):
    return OperationResult(data=manager.start_surveillance())


@router.post("/surveillance/stop", response_model=OperationResult)
def stop_surveillance(manager: Manager):
    """Stopping surveillance stops recording and intelligence for every source."""
    return OperationResult(data=manager.stop_surveillance())


@router.post("/intelligence/start", response_model=OperationResult)
def start_intelligence(manager: Manager, apply_to_sources: bool = True):
    """Enable intelligence globally.

    Requires surveillance to be active. By default this also switches every
    per-source toggle on; pass ``apply_to_sources=false`` to leave individual
    source settings untouched.
    """
    result = manager.start_intelligence()
    if apply_to_sources:
        result["sources_updated"] = manager.set_all_source_intelligence(True)
    return OperationResult(data=result)


@router.post("/intelligence/stop", response_model=OperationResult)
def stop_intelligence(manager: Manager, apply_to_sources: bool = False):
    result = manager.stop_intelligence()
    if apply_to_sources:
        result["sources_updated"] = manager.set_all_source_intelligence(False)
    return OperationResult(data=result)
