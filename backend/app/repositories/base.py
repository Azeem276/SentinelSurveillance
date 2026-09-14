"""Generic repository base.

Repositories own all query construction. Services never build SQL, and
nothing outside this package imports ``sqlalchemy.select``.
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, pk: Any) -> ModelT | None:
        return self.session.get(self.model, pk)

    def add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        self.session.flush()
        return entity

    def delete(self, entity: ModelT) -> None:
        self.session.delete(entity)
        self.session.flush()

    def delete_by_id(self, pk: Any) -> int:
        result = self.session.execute(
            delete(self.model).where(self.model.__table__.c.id == pk)
        )
        return int(result.rowcount or 0)

    def list_all(self, *, limit: int | None = None, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        return int(
            self.session.execute(select(func.count()).select_from(self.model)).scalar() or 0
        )

    def flush(self) -> None:
        self.session.flush()

    def commit(self) -> None:
        self.session.commit()
