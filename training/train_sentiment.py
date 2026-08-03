"""Train the TF-IDF + Logistic Regression sentiment model.

Data sources (in priority order):
1. ``data/ecommerce_reviews.csv``  (Women's E-Commerce Clothing Reviews)
2. Built-in labeled example corpus (always available)

Usage:
    python -m training.train_sentiment  [--csv path] [--save-to path]

Outputs ``sentiment_tfidf.joblib`` and ``sentiment_lr.joblib`` into
``models/artifacts/`` (overridable via --save-to).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# Small built-in corpus so training works without any external download.
DEFAULT_CORPUS: list[tuple[str, str]] = [
    ("I love this dress, it fits perfectly and looks amazing", "positive"),
    ("Excellent quality and fast shipping, very happy", "positive"),
    ("The material feels great and the color is beautiful", "positive"),
    ("Great value for money, would definitely recommend", "positive"),
    ("The jeans are comfortable and stylish", "positive"),
    ("This product is okay but nothing special", "neutral"),
    ("Average quality, does the job I guess", "neutral"),
    ("The shirt arrived on time, sizing was fine", "neutral"),
    ("Not great but not terrible either", "neutral"),
    ("Pretty standard product, no complaints", "neutral"),
    ("Terrible quality, fell apart after one wash", "negative"),
    ("I hate this, the stitching is awful", "negative"),
    ("Poor fit and the color faded quickly", "negative"),
    ("Broken on arrival, very disappointing", "negative"),
    ("Waste of money, do not buy this", "negative"),
    ("The delivery was slow and the package was damaged", "negative"),
    ("Customer service was rude and unhelpful", "negative"),
    ("The size runs too small and material is cheap", "negative"),
]


def load_data(csv_path: Path | None) -> tuple[list[str], list[str]]:
    """Load labeled (text, label) pairs from CSV or the built-in corpus."""
    if csv_path and csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        # E-commerce reviews: rating>=4 positive, rating==3 neutral, <=2 negative
        if "Rating" in df.columns and "Review Text" in df.columns:
            df = df.dropna(subset=["Review Text"])
            texts = df["Review Text"].astype(str).tolist()
            labels = df["Rating"].map(lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative")).tolist()
            logger.info("Loaded %d labelled reviews from %s", len(texts), csv_path)
            return texts, labels
        if "label" in df.columns:
            texts = df["text"].astype(str).tolist()
            labels = df["label"].astype(str).tolist()
            return texts, labels
        logger.warning("CSV columns not recognised; using built-in corpus.")
    texts, labels = zip(*DEFAULT_CORPUS)
    logger.info("Using built-in corpus (%d samples).", len(texts))
    return list(texts), list(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model.")
    parser.add_argument("--csv", type=Path, default=None, help="Path to labelled CSV")
    parser.add_argument("--save-to", type=Path, default=settings.MODEL_DIR)
    args = parser.parse_args()

    args.save_to.mkdir(parents=True, exist_ok=True)
    texts, labels = load_data(args.csv)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=1)),
            ("clf", LogisticRegression(max_iter=3000, C=1.5, solver="lbfgs")),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, labels=["negative", "neutral", "positive"], zero_division=0)
    logger.info("Validation report:\n%s", report)

    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["clf"]

    vec_path = args.save_to / "sentiment_tfidf.joblib"
    mdl_path = args.save_to / "sentiment_lr.joblib"
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, mdl_path)
    logger.info("Saved %s and %s", vec_path, mdl_path)


if __name__ == "__main__":
    main()
