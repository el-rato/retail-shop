"""Security utilities: API-key authentication and password-free hashing helpers."""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)


def verify_api_key(provided_key: str) -> bool:
    """Constant-time comparison of the provided key against the configured one."""
    expected = settings.API_KEY.encode("utf-8")
    actual = (provided_key or "").encode("utf-8")
    return hmac.compare_digest(actual, expected)


async def require_api_key(api_key: str | None = Depends(api_key_header)) -> str:
    """FastAPI dependency enforcing API-key authentication."""
    if api_key is None or not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
