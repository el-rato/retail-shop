"""POST /analyze-sentiment endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.database.session import get_db
from app.schemas.api import SentimentRequest, SentimentResponse
from app.services import sentiment_service

router = APIRouter(prefix="/analyze-sentiment", tags=["Sentiment Analysis"])


@router.post("", response_model=SentimentResponse)
def analyze_sentiment(
    payload: SentimentRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Predict whether a text is positive, neutral, or negative."""
    return sentiment_service.analyze_sentiment(db, payload.text, payload.customer_id)
