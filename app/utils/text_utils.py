"""Text helpers shared by the sentiment model and chatbot."""

from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords

_STOPWORDS: set[str] | None = None


def _ensure_nltk_resources() -> None:
    """Download stopwords once, silently ignoring failures (offline mode)."""
    global _STOPWORDS
    if _STOPWORDS is not None:
        return
    try:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS = set()


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/mentions/extra whitespace; keep letters/digits."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter English stopwords from a token list."""
    _ensure_nltk_resources()
    return [t for t in tokens if t not in (_STOPWORDS or set())]


def tokenize(text: str) -> list[str]:
    """Tokenise cleaned text into lowercase word tokens."""
    return clean_text(text).split()
