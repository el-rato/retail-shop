"""Seed the database with realistic demo data for the dashboard.

Creates a handful of customers with visits, sentiment logs, chat logs, and
product predictions so the Streamlit dashboard is populated on first run.

Usage:
    python -m scripts.seed_demo_data  [--n 20]
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger  # noqa: E402
from app.database.models import (  # noqa: E402
    ChatLog,
    Customer,
    ProductPrediction,
    SentimentLog,
    Visit,
)
from app.database.session import SessionLocal, init_db  # noqa: E402

logger = get_logger(__name__)

NAMES = ["Alice Chen", "Bob Martin", "Carla Diaz", "David Kim", "Elena Rossi", "Frank Wells"]
LOCATIONS = ["Main Street", "Mall Branch", "Airport", "Downtown", "Outlet"]

SENTIMENTS = [
    ("Great service and friendly staff!", "positive"),
    ("The new collection is beautiful.", "positive"),
    ("Fast checkout, love this store.", "positive"),
    ("Prices are a bit high but okay.", "neutral"),
    ("Average experience overall.", "neutral"),
    ("Long queue at the counter.", "negative"),
    ("The fitting room was dirty.", "negative"),
    ("Item arrived damaged.", "negative"),
]

CHATS = [
    ("What are your store hours?", "hours"),
    ("How can I return an item?", "returns"),
    ("Do you offer gift cards?", "gift_cards"),
    ("What is your shipping policy?", "shipping"),
    ("Are there any discounts today?", "promotions"),
]

PRODUCTS = [
    ("Ankle Boots", "footwear"), ("Running Shoes", "footwear"),
    ("Leather Handbag", "bags"), ("Backpack", "bags"),
    ("Smart Watch", "electronics"), ("Bluetooth Earbuds", "electronics"),
    ("Denim Jacket", "apparel"), ("Silk Scarf", "apparel"),
]


def seed(n: int) -> None:
    init_db()
    db = SessionLocal()
    if db.query(Customer).count() > 0:
        logger.info("DB already seeded; skipping.")
        db.close()
        return

    customers: list[Customer] = []
    for name in NAMES:
        customer = Customer(name=name, face_encodings=[])
        db.add(customer)
        customers.append(customer)
    db.commit()

    for customer in customers:
        num_visits = random.randint(2, 8)
        for _ in range(num_visits):
            db.add(
                Visit(
                    customer_id=customer.id,
                    confidence=round(random.uniform(0.75, 0.99), 3),
                    location=random.choice(LOCATIONS),
                    entered_at=datetime.now(UTC) - timedelta(days=random.randint(0, 14), hours=random.randint(0, 12)),
                )
            )
    for _ in range(n):
        text, label = random.choice(SENTIMENTS)
        db.add(
            SentimentLog(
                text=text,
                sentiment=label,
                confidence=round(random.uniform(0.6, 0.98), 3),
                customer_id=random.choice(customers).id,
            )
        )
    for message, intent in CHATS:
        db.add(
            ChatLog(
                user_message=message,
                bot_reply=f"Thanks for asking about {intent}.",
                intent=intent,
                confidence=round(random.uniform(0.6, 0.98), 3),
            )
        )
    for label, category in PRODUCTS:
        db.add(
            ProductPrediction(
                product_label=label,
                category=category,
                confidence=round(random.uniform(0.7, 0.99), 3),
            )
        )
    db.commit()
    db.close()
    logger.info("Seeded %d demo records.", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    seed(args.n)
