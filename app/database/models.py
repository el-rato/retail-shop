"""SQLAlchemy ORM models: customers, visits, and analytics logs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, utc_now


class Customer(Base):
    """A returning retail customer identified by face encoding."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, default="Customer")
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON list of 128-d face encodings for multiple reference images
    face_encodings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    visits: Mapped[list[Visit]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )


class Visit(Base):
    """A single store visit logged when a face is recognised."""

    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    location: Mapped[str | None] = mapped_column(String(120), nullable=True)
    entered_at: Mapped[datetime] = mapped_column(
        default=utc_now, nullable=False
    )

    customer: Mapped[Customer] = relationship(back_populates="visits")


class SentimentLog(Base):
    """Logged customer-review / feedback sentiment analysis."""

    __tablename__ = "sentiment_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    customer_id: Mapped[int | None] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"), nullable=True
    )


class ChatLog(Base):
    """Log of every FAQ chatbot interaction."""

    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    bot_reply: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(60), nullable=False, default="unknown")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class ProductPrediction(Base):
    """Log of product-image classification predictions."""

    __tablename__ = "product_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_label: Mapped[str] = mapped_column(String(120), nullable=False)
    category: Mapped[str] = mapped_column(String(120), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    predicted_at: Mapped[datetime] = mapped_column(
        default=utc_now, nullable=False
    )
