"""PyTest fixtures.

Configures an isolated SQLite database and a FastAPI TestClient. Heavy ML
dependencies (TensorFlow/dlib) are not required to run the test suite —
services that would touch them are monkeypatched in the relevant test modules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Point the app at an isolated test DB + test API key BEFORE any app import.
TEST_DB = Path(__file__).resolve().parent / "test_retail.db"
TEST_MODELS = Path(__file__).resolve().parent / "tmp_models"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{TEST_DB}")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ENV", "test")
# Isolate model artifacts so tests always exercise deterministic fallbacks.
os.environ.setdefault("MODEL_DIR", str(TEST_MODELS))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_db():
    if TEST_DB.exists():
        TEST_DB.unlink(missing_ok=True)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    TEST_DB.unlink(missing_ok=True)
    if TEST_MODELS.exists():
        import shutil

        shutil.rmtree(TEST_MODELS, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_tables(db):
    """Clear all table rows before every test for full isolation."""
    yield
    for table in reversed(Base.metadata.sorted_tables):
        db.execute(delete(table))
    db.commit()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def api_key() -> str:
    return settings.API_KEY


@pytest.fixture()
def auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key}
