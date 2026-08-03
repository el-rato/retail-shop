"""FAQ chatbot: rule-based + ML intent classification.

Intents are defined in ``data/intents.json``. At startup we build a small
TF-IDF + Logistic Regression classifier over the pattern corpus (trained
in-memory in ~seconds — no disk artifacts needed). Matching uses either the
ML prediction or, as a fallback, token-overlap similarity. Responses are
selected with light randomness from the intent's response pool.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChatbotError(RuntimeError):
    pass


class FAQChatbot:
    """Rule-based + ML intent classifier backed by ``intents.json``."""

    def __init__(self, intents_path: Path | None = None) -> None:
        self.intents_path = intents_path or settings.intents_path
        self.intents: list[dict] = []
        self._patterns: list[str] = []
        self._tags: list[str] = []
        self._vectorizer = None
        self._model = None
        self._fallback_response = "I'm sorry, I didn't understand that. Try asking about store hours, returns, or products."
        self.load()

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not self.intents_path.exists():
            raise ChatbotError(f"intents.json not found at {self.intents_path}")
        try:
            data = json.loads(self.intents_path.read_text(encoding="utf-8"))
            self.intents = data.get("intents", [])
        except (json.JSONDecodeError, OSError) as exc:
            raise ChatbotError(f"Failed to parse intents.json: {exc}") from exc

        if not self.intents:
            raise ChatbotError("intents.json contains no intents.")

        for intent in self.intents:
            for pattern in intent.get("patterns", []):
                self._patterns.append(pattern.lower())
                self._tags.append(intent["tag"])
        self._train_ml()

    def _train_ml(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

            from app.utils.text_utils import clean_text  # noqa: PLC0415

            cleaned = [clean_text(p) for p in self._patterns]
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            x = self._vectorizer.fit_transform(cleaned)
            self._model = LogisticRegression(max_iter=2000, C=1.0)
            self._model.fit(x, self._tags)
            logger.info("Chatbot ML intent classifier trained on %d patterns.", len(cleaned))
        except Exception as exc:
            self._vectorizer = self._model = None
            logger.warning("Chatbot ML classifier unavailable (%s); using rule matching.", exc)

    # ------------------------------------------------------------------ #
    def _predict_intent(self, message: str) -> tuple[str, float]:
        from app.utils.text_utils import clean_text  # noqa: PLC0415

        cleaned = clean_text(message)
        if self._model is not None and cleaned:
            try:
                vec = self._vectorizer.transform([cleaned])
                proba = self._model.predict_proba(vec)[0]
                idx = int(proba.argmax())
                confidence = float(proba[idx])
                if confidence >= 0.35:
                    return str(self._model.classes_[idx]), confidence
            except Exception:
                pass

        # Rule-based fallback: token overlap similarity
        best_tag, best_score = self._fallback_response, 0.0
        msg_tokens = set(cleaned.split()) if cleaned else set()
        for intent in self.intents:
            for pattern in intent.get("patterns", []):
                pattern_tokens = set(clean_text(pattern).split())
                if not pattern_tokens:
                    continue
                overlap = len(msg_tokens & pattern_tokens)
                similarity = overlap / max(len(pattern_tokens), 1)
                if similarity > best_score:
                    best_score = similarity
                    best_tag = intent["tag"]
        if best_score >= 0.5:
            return best_tag, min(0.9, best_score)
        return "unknown", 0.0

    # ------------------------------------------------------------------ #
    def respond(self, message: str) -> tuple[str, str, float, str | None]:
        """Return (reply, intent_tag, confidence, matched_pattern)."""
        tag, confidence = self._predict_intent(message)
        for intent in self.intents:
            if intent["tag"] == tag:
                responses = intent.get("responses", [])
                reply = random.choice(responses) if responses else self._fallback_response
                return reply, tag, confidence, None
        return self._fallback_response, "unknown", confidence, None
