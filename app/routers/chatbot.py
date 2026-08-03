"""POST /chatbot endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.database.session import get_db
from app.schemas.api import ChatRequest, ChatResponse
from app.services import chatbot_service

router = APIRouter(prefix="/chatbot", tags=["FAQ Chatbot"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Answer a customer FAQ using rule-based + ML intent classification."""
    return chatbot_service.chat(db, payload.message)
