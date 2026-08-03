"""Product classification orchestration with visit/prediction logging."""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.models import ProductPrediction
from app.services.pipeline import get_pipeline

logger = get_logger(__name__)


def classify_product(db: Session, image_bgr: np.ndarray, top_k: int = 3) -> dict:
    """Run MobileNetV2 classification and log the top prediction."""
    pipeline = get_pipeline()
    predictions = pipeline.product.classify(image_bgr, top_k=top_k)

    if predictions:
        top = predictions[0]
        record = ProductPrediction(
            product_label=top["label"],
            category=top["category"],
            confidence=top["confidence"],
        )
        db.add(record)
        db.commit()

    logger.info("Classified product: %s", predictions[:1])
    return {
        "top_prediction": predictions[0] if predictions else None,
        "all_predictions": predictions,
    }
