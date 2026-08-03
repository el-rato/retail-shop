"""Chatbot orchestration with interaction logging."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.models import ChatLog
from app.services.pipeline import get_pipeline

logger = get_logger(__name__)


def chat(db: Session, message: str) -> dict:
    """Produce a chatbot reply and persist the interaction."""
    pipeline = get_pipeline()
    reply, intent, confidence, matched_pattern = pipeline.chatbot.respond(message)

    record = ChatLog(
        user_message=message,
        bot_reply=reply,
        intent=intent,
        confidence=confidence,
    )
    db.add(record)
    db.commit()

    logger.info("Chatbot intent='%s' conf=%.3f for '%s'", intent, confidence, message[:40])
    return {
        "reply": reply,
        "intent": intent,
        "confidence": confidence,
        "matched_pattern": matched_pattern,
    }
