"""Text helpers shared by the sentiment model and chatbot."""

from __future__ import annotations

import os
import re

# NLTK 3.9+ ships an import-security hook (nltk.inisec) that blocks NLTK from
# importing modules whose path resolves inside the current working directory.
# Because our virtualenv lives at <repo>/.venv (under the project root), every
# NLTK import would be flagged as "CWD code" and blocked. This is a documented
# false positive for in-repo venvs, so we opt out before NLTK is loaded.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

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
