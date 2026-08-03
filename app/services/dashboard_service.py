"""Dashboard statistics aggregation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.models import ChatLog, Customer, ProductPrediction, SentimentLog, Visit

logger = get_logger(__name__)


def _recent_visits(db: Session, limit: int = 10) -> list[dict]:
    rows = (
        db.execute(
            select(Visit, Customer.name)
            .join(Customer, Visit.customer_id == Customer.id)
            .order_by(Visit.entered_at.desc())
            .limit(limit)
        )
        .all()
    )
    return [
        {
            "customer_name": name,
            "entered_at": visit.entered_at.isoformat(),
            "confidence": round(visit.confidence, 3),
            "location": visit.location,
        }
        for visit, name in rows
    ]


def _sentiment_trend(db: Session, days: int = 7) -> list[dict]:
    since = datetime.now(UTC) - timedelta(days=days)
    rows = (
        db.execute(
            select(SentimentLog.created_at, SentimentLog.sentiment)
            .where(SentimentLog.created_at >= since)
            .order_by(SentimentLog.created_at.asc())
        )
        .all()
    )
    buckets: dict[str, dict[str, int]] = {}
    for created_at, sentiment in rows:
        day = created_at.date().isoformat() if created_at else "unknown"
        bucket = buckets.setdefault(day, {"date": day, "positive": 0, "neutral": 0, "negative": 0})
        bucket[sentiment] = bucket.get(sentiment, 0) + 1
    return list(buckets.values())


def build_dashboard_stats(db: Session) -> dict:
    """Aggregate all dashboard statistics into a single JSON-friendly dict."""
    total_customers = db.execute(select(func.count(Customer.id))).scalar_one()
    total_visits = db.execute(select(func.count(Visit.id))).scalar_one()
    unique_visitors = db.execute(select(func.count(func.distinct(Visit.customer_id)))).scalar_one()

    top_row = db.execute(
        select(Customer.name, func.count(Visit.id).label("visit_count"))
        .join(Visit, Visit.customer_id == Customer.id)
        .group_by(Customer.id)
        .order_by(func.count(Visit.id).desc())
        .limit(1)
    ).first()
    top_customer = {"name": top_row[0], "visits": top_row[1]} if top_row else None

    sentiment_counts = {
        "positive": db.execute(
            select(func.count(SentimentLog.id)).where(SentimentLog.sentiment == "positive")
        ).scalar_one(),
        "neutral": db.execute(
            select(func.count(SentimentLog.id)).where(SentimentLog.sentiment == "neutral")
        ).scalar_one(),
        "negative": db.execute(
            select(func.count(SentimentLog.id)).where(SentimentLog.sentiment == "negative")
        ).scalar_one(),
    }

    chat_rows = db.execute(
        select(ChatLog.intent, func.count(ChatLog.id))
        .group_by(ChatLog.intent)
        .order_by(func.count(ChatLog.id).desc())
    ).all()
    chat_counts = {intent: count for intent, count in chat_rows}

    product_rows = db.execute(
        select(ProductPrediction.category, func.count(ProductPrediction.id))
        .group_by(ProductPrediction.category)
        .order_by(func.count(ProductPrediction.id).desc())
    ).all()
    product_counts = {category: count for category, count in product_rows}

    recent_chats = db.execute(
        select(ChatLog).order_by(ChatLog.created_at.desc()).limit(10)
    ).scalars().all()

    logger.info("Dashboard stats computed: %d customers, %d visits.", total_customers, total_visits)
    return {
        "total_customers": total_customers,
        "total_visits": total_visits,
        "unique_visitors": unique_visitors,
        "top_customer": top_customer,
        "sentiment_counts": sentiment_counts,
        "sentiment_trend": _sentiment_trend(db),
        "chat_counts": chat_counts,
        "product_counts": product_counts,
        "recent_visits": _recent_visits(db),
        "recent_chats": [
            {
                "user_message": c.user_message,
                "bot_reply": c.bot_reply,
                "intent": c.intent,
                "created_at": c.created_at.isoformat(),
            }
            for c in recent_chats
        ],
        "generated_at": datetime.now(UTC).isoformat(),
    }
