"""Database engine and session management.

Works with SQLite out of the box and is PostgreSQL-ready (see config.py,
which rewrites ``postgres://`` to ``postgresql+psycopg://``).
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

engine = create_engine(
    settings.sqlalchemy_database_uri,
    echo=settings.DB_ECHO,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False}
    if settings.sqlalchemy_database_uri.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
)


def init_db() -> None:
    """Create all tables and persist metadata/model files to disk if missing."""
    from app.database import models  # noqa: F401  (register models on Base)
    from app.database.base import Base

    Base.metadata.create_all(bind=engine)
    logger.info("Database initialised at %s", settings.sqlalchemy_database_uri)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
