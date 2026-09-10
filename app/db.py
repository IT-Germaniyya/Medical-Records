from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str | None = None):
    url = database_url or settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=make_engine(database_url), expire_on_commit=False, class_=Session)


def create_schema(database_url: str | None = None) -> None:
    # Convenience for local MVP usage. Production uses Alembic migrations.
    from app import models  # noqa: F401
    Base.metadata.create_all(make_engine(database_url))

