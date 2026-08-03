"""Sentiment analysis with TF-IDF + Logistic Regression.

Loads a trained pipeline (vectorizer + classifier) from disk. If missing,
uses a compact rule-based lexicon fallback so the endpoint never 500s on a
fresh install; run ``training/train_sentiment.py`` to get real ML quality.
"""

from __future__ import annotations

import joblib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False

_POSITIVE_WORDS = {
    "love", "great", "excellent", "good", "best", "awesome", "amazing", "happy",
    "perfect", "recommend", "like", "nice", "fantastic", "wonderful", "clean",
    "comfortable", "quality", "fast", "helpful", "beautiful", "favorite",
    "satisfied", "worth", "thank", "pleased", "enjoy", "superb", "top",
}
_NEGATIVE_WORDS = {
    "bad", "worst", "terrible", "awful", "hate", "poor", "disappointed",
    "disappointing", "broken", "defective", "cheap", "dirty", "slow", "rude",
    "waste", "horrible", "unhappy", "returned", "refund", "complaint", "ugly",
    "uncomfortable", "useless", "frustrating", "annoying", "disappointed",
    "never", "not", "no",
}

_NEUTRAL_WORDS = {
    "product", "shipping", "size", "color", "material", "fit", "order",
    "delivery", "packaging", "price", "store", "website", "bought", "ordered",
}


class SentimentAnalyzer:
    """TF-IDF + Logistic Regression sentiment classifier."""

    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None
        self.labels = ["negative", "neutral", "positive"]
        self._load()

    def _load(self) -> None:
        vec_path = settings.model_paths["sentiment_vectorizer"]
        mdl_path = settings.model_paths["sentiment_model"]
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn unavailable; using lexicon fallback.")
            return
        if vec_path.exists() and mdl_path.exists():
            try:
                self.vectorizer = joblib.load(vec_path)
                self.model = joblib.load(mdl_path)
                logger.info("Loaded trained sentiment pipeline.")
                return
            except Exception as exc:
                logger.error("Failed to load sentiment artifacts (%s); using fallback.", exc)
        logger.info("No trained sentiment artifacts found; using lexicon fallback.")

    @property
    def is_ml(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Return (label, confidence, probability-dict)."""
        if self.is_ml:
            return self._analyze_ml(text)
        return self._analyze_lexicon(text)

    def _analyze_ml(self, text: str) -> tuple[str, float, dict[str, float]]:
        from app.utils.text_utils import clean_text  # noqa: PLC0415

        cleaned = clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(vec)[0]
        label = str(self.model.predict(vec)[0])
        probabilities = {
            str(cls): round(float(p), 4) for cls, p in zip(self.model.classes_, proba)
        }
        return label, round(float(proba.max()), 4), probabilities

    def _analyze_lexicon(self, text: str) -> tuple[str, float, dict[str, float]]:
        from app.utils.text_utils import tokenize  # noqa: PLC0415

        tokens = tokenize(text)
        if not tokens:
            return "neutral", 0.5, {"negative": 0.1, "neutral": 0.8, "positive": 0.1}

        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        neu = sum(1 for t in tokens if t in _NEUTRAL_WORDS)
        total = len(tokens) + 1.0

        pos_score, neg_score, neu_score = pos / total, neg / total, neu / total
        if pos_score > neg_score and pos_score > neu_score:
            label = "positive"
        elif neg_score > pos_score and neg_score > neu_score:
            label = "negative"
        else:
            label = "neutral"

        confidence = max(pos_score, neg_score, neu_score)
        if confidence < 0.25:
            label, confidence = "neutral", 0.6
        probabilities = {
            "negative": round(neg_score, 4),
            "neutral": round(neu_score, 4),
            "positive": round(pos_score, 4),
        }
        return label, round(confidence, 4), probabilities
