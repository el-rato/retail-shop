"""Unified ML pipeline: lazy singleton loading of every prediction service.

All model wrappers are instantiated exactly once per process and shared via
FastAPI dependency injection through ``get_pipeline()``.
"""

from __future__ import annotations

from threading import Lock

from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = Lock()
_pipeline: MLPipeline | None = None


class MLPipeline:
    """Container for the four prediction services."""

    def __init__(self) -> None:
        from app.models.chatbot_model import FAQChatbot
        from app.models.face_model import FaceDetector
        from app.models.product_model import ProductClassifier
        from app.models.sentiment_model import SentimentAnalyzer

        self.face = FaceDetector()
        self.product = ProductClassifier()
        self.sentiment = SentimentAnalyzer()
        self.chatbot = FAQChatbot()
        logger.info("ML pipeline initialised: face=%s, product=%s, sentiment=ml:%s, chatbot=ml:%s",
                    "dlib" if self.face.using_dlib else "opencv",
                    "fine-tuned" if self.product.model is not None else "imagenet",
                    self.sentiment.is_ml,
                    self.chatbot._model is not None)


def get_pipeline() -> MLPipeline:
    """Return the process-wide MLPipeline, constructing it once (thread-safe)."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                try:
                    _pipeline = MLPipeline()
                except Exception as exc:
                    logger.exception("Failed to initialise ML pipeline")
                    raise ModelLoadError(f"ML pipeline initialisation failed: {exc}") from exc
    return _pipeline


def reset_pipeline_for_tests() -> None:
    """Clear the cached pipeline (used by the test suite)."""
    global _pipeline
    with _lock:
        _pipeline = None
