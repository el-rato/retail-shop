"""GET /dashboard/stats endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.database.session import get_db
from app.schemas.api import DashboardStats
from app.services import dashboard_service

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
def dashboard_stats(
    db: Session = Depends(get_db),
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Return aggregated business-intelligence statistics."""
    return dashboard_service.build_dashboard_stats(db)
