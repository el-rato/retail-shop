"""Product image classification with MobileNetV2.

Strategy:
1. If a fine-tuned checkpoint exists (training/train_product.py), load it and
   use the trained retail labels.
2. Otherwise use ImageNet-pretrained MobileNetV2 to predict, decoding the
   synset labels via ``decode_predictions`` and mapping them to friendly retail
   categories through ``IMAGENET_RETAIL_MAP``.
This keeps the endpoint functional out-of-the-box and improved after training.
"""

from __future__ import annotations

import json

import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelLoadError
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from keras.applications import MobileNetV2
    from keras.models import load_model

    KERAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    KERAS_AVAILABLE = False

# Retail-friendly labels used by the transfer-learning script (Fashion-MNIST).
DEFAULT_RETAIL_LABELS = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle-boot",
]

# Maps a subset of ImageNet synsets to friendly retail categories.
IMAGENET_RETAIL_MAP: dict[str, str] = {
    "backpack": "bags",
    "purse": "bags",
    "wallet": "accessories",
    "sunscreen": "beauty",
    "lipstick": "beauty",
    "sunglasses": "accessories",
    "watch": "accessories",
    "hand-blower": "electronics",
    "cellular_telephone": "electronics",
    "notebook": "electronics",
    "desktop_computer": "electronics",
    "laptop": "electronics",
    "digital_watch": "accessories",
    "jigsaw_puzzle": "toys",
    "teddy": "toys",
    "basketball": "sports",
    "football_helmet": "sports",
    "tennis_ball": "sports",
    "sandal": "footwear",
    "sneaker": "footwear",
    "running_shoe": "footwear",
    "bathing_trunks": "apparel",
    "jean": "apparel",
    "jersey": "apparel",
    "sweatshirt": "apparel",
    "shirt": "apparel",
    "suit": "apparel",
    "maillot": "apparel",
    "hoodie": "apparel",
    "mask": "apparel",
    "monitor": "electronics",
    "printer": "electronics",
    "hard_disc": "electronics",
}


class ProductClassifier:
    """MobileNetV2-based product image classifier."""

    def __init__(self) -> None:
        self.model = None
        self._imagenet_model = None
        self.labels: list[str] = []
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not KERAS_AVAILABLE:
            raise ModelLoadError("TensorFlow/Keras is not installed in this environment.")

        artifact = settings.model_paths["product"]
        labels_path = settings.model_paths["product_labels"]

        if artifact.exists() and labels_path.exists():
            try:
                self.model = load_model(artifact)
                self.labels = list(json.loads(labels_path.read_text(encoding="utf-8")))
                logger.info("Loaded fine-tuned product model (%d classes).", len(self.labels))
                return
            except Exception as exc:
                logger.error(
                    "Failed to load fine-tuned model (%s); falling back to ImageNet.", exc
                )

        try:
            self._imagenet_model = MobileNetV2(
                weights="imagenet", include_top=True, input_shape=(224, 224, 3)
            )
            self.labels = []
            logger.info("Using ImageNet-pretrained MobileNetV2 (fallback).")
        except Exception as exc:
            raise ModelLoadError(
                f"Could not initialise MobileNetV2 (weights download may be blocked): {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    def classify(self, image_bgr: np.ndarray, top_k: int = 3) -> list[dict]:
        """Return top-k predictions as ``[{label, category, confidence}, ...]``."""
        from app.utils.image_utils import preprocess_product_image  # noqa: PLC0415

        x = preprocess_product_image(image_bgr)
        batch = np.expand_dims(x, axis=0)

        if self.model is not None:
            proba = np.asarray(self.model.predict(batch, verbose=0))[0]
            return self._format_topk(proba, self.labels, top_k)

        proba = np.asarray(self._imagenet_model.predict(batch, verbose=0))[0]
        return self._format_imagenet(proba, top_k)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_topk(proba: np.ndarray, labels: list[str], top_k: int) -> list[dict]:
        order = np.argsort(proba)[::-1][:top_k]
        results: list[dict] = []
        for idx in order:
            label = str(labels[int(idx)])
            results.append(
                {
                    "label": label,
                    "category": label,
                    "confidence": round(float(proba[int(idx)]), 4),
                }
            )
        return results

    @staticmethod
    def _format_imagenet(proba: np.ndarray, top_k: int) -> list[dict]:
        from keras.applications.mobilenet_v2 import decode_predictions  # noqa: PLC0415

        decoded = decode_predictions(np.expand_dims(proba, 0), top=top_k)[0]
        results: list[dict] = []
        for (_, synset_label, score) in decoded:
            human = synset_label.replace("_", " ").title()
            category = IMAGENET_RETAIL_MAP.get(synset_label, "general")
            results.append(
                {
                    "label": human,
                    "category": category,
                    "confidence": round(float(score), 4),
                }
            )
        return results
