"""Tests for sentiment analysis service + endpoint (lexicon fallback path)."""

from __future__ import annotations

from app.models.sentiment_model import SentimentAnalyzer
from app.services.sentiment_service import analyze_sentiment


def test_lexicon_positive():
    analyzer = SentimentAnalyzer()
    label, confidence, probs = analyzer.analyze("I love this dress, it is great!")
    assert label == "positive"
    assert 0 <= confidence <= 1


def test_lexicon_negative():
    analyzer = SentimentAnalyzer()
    label, confidence, _ = analyzer.analyze("Terrible product, I hate it.")
    assert label == "negative"


def test_lexicon_empty_neutral():
    analyzer = SentimentAnalyzer()
    label, _, _ = analyzer.analyze("")
    assert label == "neutral"


def test_analyze_sentiment_service_logs(db):
    result = analyze_sentiment(db, "Excellent quality and fast shipping")
    assert result["sentiment"] == "positive"
    assert "probabilities" in result
    assert result["confidence"] > 0


def test_sentiment_endpoint(client, auth_headers):
    resp = client.post(
        "/analyze-sentiment",
        json={"text": "I love this store"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"] in {"positive", "neutral", "negative"}
    assert "probabilities" in body


def test_sentiment_endpoint_validation(client, auth_headers):
    resp = client.post(
        "/analyze-sentiment",
        json={"text": ""},
        headers=auth_headers,
    )
    assert resp.status_code == 422
