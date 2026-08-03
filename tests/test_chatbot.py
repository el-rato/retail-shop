"""Tests for the FAQ chatbot (rule-based + ML intent classification)."""

from __future__ import annotations

from app.models.chatbot_model import FAQChatbot
from app.services.chatbot_service import chat


def test_chatbot_recognises_greeting():
    bot = FAQChatbot()
    reply, intent, confidence, _ = bot.respond("hello")
    assert intent == "greeting"
    assert isinstance(reply, str) and reply


def test_chatbot_recognises_hours():
    bot = FAQChatbot()
    reply, intent, confidence, _ = bot.respond("What are your opening hours?")
    assert intent == "hours"
    assert isinstance(reply, str)


def test_chatbot_unknown_fallback():
    bot = FAQChatbot()
    _, intent, _, _ = bot.respond("zxqkxjdhfjfjfjfjfjfjfjj")
    assert intent == "unknown"


def test_chatbot_service_logs(db):
    result = chat(db, "How do I return an item?")
    assert result["intent"] in {"returns", "unknown"}
    assert "reply" in result


def test_chatbot_endpoint(client, auth_headers):
    resp = client.post("/chatbot", json={"message": "hi there"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "reply" in body
    assert "intent" in body
