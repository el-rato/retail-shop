"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent


class Settings(BaseSettings):
    """Runtime settings for the platform. Overridable via env vars / .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Smart Retail Customer Intelligence API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENV: Literal["development", "test", "production"] = "development"

    # Security
    API_KEY: str = "change-me-in-production"
    API_KEY_NAME: str = "X-API-Key"
    CORS_ORIGINS: str = "*"

    # Database (SQLite default, PostgreSQL-ready via env override)
    DATABASE_URL: str = "sqlite:///./data/retail.db"
    DB_ECHO: bool = False

    # Model artifacts
    MODEL_DIR: Path = PROJECT_ROOT / "models" / "artifacts"
    FACE_ENCODING_MODEL: str = "cnn"
    FACE_TOLERANCE: float = 0.5

    # Data dirs
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOG_DIR: Path = PROJECT_ROOT / "data" / "logs"

    # Uploaded image size limit (bytes)
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

    # Streamlit / dashboard API base
    DASHBOARD_API_BASE: str = "http://localhost:8000"

    @property
    def sqlalchemy_database_uri(self) -> str:
        """Return a SQLAlchemy-compatible URI (postgres needs psycopg)."""
        uri = self.DATABASE_URL
        if uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql+psycopg://", 1)
        elif uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            # resolve relative sqlite paths against project root
            rel = uri[len("sqlite:///"):]
            if not os.path.isabs(rel):
                uri = f"sqlite:///{PROJECT_ROOT / rel}"
        return uri

    @property
    def model_paths(self) -> dict[str, Path]:
        """Standard locations for trained artifacts."""
        return {
            "product": self.MODEL_DIR / "product_mobilenetv2.keras",
            "product_labels": self.MODEL_DIR / "product_labels.json",
            "sentiment_vectorizer": self.MODEL_DIR / "sentiment_tfidf.joblib",
            "sentiment_model": self.MODEL_DIR / "sentiment_lr.joblib",
        }

    @property
    def intents_path(self) -> Path:
        return self.DATA_DIR / "intents.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
