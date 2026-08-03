"""Sentiment analysis orchestration with result logging."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.models import SentimentLog
from app.services.pipeline import get_pipeline

logger = get_logger(__name__)


def analyze_sentiment(db: Session, text: str, customer_id: int | None = None) -> dict:
    """Analyse sentiment, persist the log, and return results."""
    pipeline = get_pipeline()
    sentiment, confidence, probabilities = pipeline.sentiment.analyze(text)

    record = SentimentLog(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        customer_id=customer_id,
    )
    db.add(record)
    db.commit()

    logger.info("Sentiment='%s' conf=%.3f for '%s'", sentiment, confidence, text[:40])
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
    }
