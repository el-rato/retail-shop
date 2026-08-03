"""Centralised logging configuration.

- Writes to console and a rotating file under data/logs.
- Provides a ``get_logger`` helper used across the codebase.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import settings


def _configure() -> None:
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = settings.LOG_DIR / "retail.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Tame noisy third-party loggers
    for noisy in ("httpx", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    return logging.getLogger(name)
