"""FastAPI dependencies and shared route helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.surveillance import SurveillanceManager, get_manager

DbSession = Annotated[Session, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]
Manager = Annotated[SurveillanceManager, Depends(get_manager)]


class Pagination:
    def __init__(
        self,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset


PaginationDep = Annotated[Pagination, Depends(Pagination)]


def window_since(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)
