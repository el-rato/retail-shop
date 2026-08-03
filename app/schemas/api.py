"""Pydantic request/response models used by the FastAPI routers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Face recognition
# --------------------------------------------------------------------------- #
class FaceRecognitionRequest(BaseModel):
    """Body for POST /recognize-face.

    ``image_base64`` is a standard base64-encoded image (JPEG/PNG/WebP).
    ``register`` triggers face-encoding enrolment when the face is unknown.
    """

    image_base64: str = Field(..., description="Base64-encoded image string")
    enroll: bool = Field(
        False, description="Enrol unknown face as a new customer when True"
    )
    customer_name: str | None = Field(None, max_length=120, description="Name for new customer")
    location: str | None = Field(None, max_length=120)

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("image_base64 cannot be empty")
        # Allow both raw base64 and data-URI prefixed forms.
        if v.startswith("data:"):
            v = v.split(",", 1)[1]
        return v


class FaceRecognitionResponse(BaseModel):
    recognized: bool
    customer_id: int | None = None
    customer_name: str | None = None
    confidence: float | None = None
    visits: int | None = None
    faces_detected: int
    message: str


# --------------------------------------------------------------------------- #
# Product classification
# --------------------------------------------------------------------------- #
class ClassifyProductRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded product image")

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("image_base64 cannot be empty")
        if v.startswith("data:"):
            v = v.split(",", 1)[1]
        return v


class ProductPrediction(BaseModel):
    label: str
    category: str
    confidence: float


class ClassifyProductResponse(BaseModel):
    top_prediction: ProductPrediction
    all_predictions: list[ProductPrediction]


# --------------------------------------------------------------------------- #
# Sentiment analysis
# --------------------------------------------------------------------------- #
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    customer_id: int | None = None


class SentimentResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float
    probabilities: dict[str, float]


# --------------------------------------------------------------------------- #
# Chatbot
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    matched_pattern: str | None = None


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class DashboardStats(BaseModel):
    total_customers: int
    total_visits: int
    unique_visitors: int
    top_customer: dict[str, Any] | None = None
    sentiment_counts: dict[str, int]
    sentiment_trend: list[dict[str, Any]]
    chat_counts: dict[str, int]
    product_counts: dict[str, int]
    recent_visits: list[dict[str, Any]]
    recent_chats: list[dict[str, Any]]
    generated_at: datetime
