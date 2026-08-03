"""Application exceptions and global FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """Base application error with an associated HTTP status code."""

    status_code = 500
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, status_code: int | None = None):
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)


class InvalidImageError(AppError):
    status_code = 400
    message = "Invalid or unreadable image payload."


class ModelLoadError(AppError):
    status_code = 503
    message = "Required ML model is not available."


class DatabaseError(AppError):
    status_code = 500
    message = "A database error occurred."


class RateLimitError(AppError):
    status_code = 429
    message = "Too many requests."


def register_exception_handlers(app: FastAPI) -> None:
    """Attach JSON handlers for AppError and generic 500s."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.error("AppError on %s: %s", request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "type": exc.__class__.__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "type": "InternalError"},
        )
