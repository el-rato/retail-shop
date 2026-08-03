"""Train MobileNetV2 (transfer learning) on Fashion-MNIST.

Fashion-MNIST (28x28 grayscale) is upscaled to 224x224 RGB and used to
fine-tune a MobileNetV2 backbone topped with a 10-class dense head. The
resulting ``.keras`` checkpoint + ``labels.json`` are dropped into
``models/artifacts/`` and picked up automatically by the API.

Training uses a ``Sequence`` generator so the full 60k image set is never
held in memory as 224x224 arrays.

Usage:
    python -m training.train_product [--epochs 6] [--batch-size 32]
                                     [--limit 5000] [--save-to models/artifacts]

Requires TensorFlow. Uses keras.datasets if the local gz binaries are missing
(they can also be fetched with ``python -m training.download_datasets --fashion``).
"""

from __future__ import annotations

import argparse
import gzip
import sys
from math import ceil
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.models.product_model import DEFAULT_RETAIL_LABELS  # noqa: E402

logger = get_logger(__name__)

FASHION_DIR = settings.DATA_DIR / "fashion-mnist"
IMG_SIZE = 224
NUM_CLASSES = len(DEFAULT_RETAIL_LABELS)

try:
    from keras.utils import Sequence as _KerasSequence

    _SEQUENCE_BASE = _KerasSequence
except Exception:  # pragma: no cover - keras is required to train anyway
    _SEQUENCE_BASE = object


def _read_idx(path: Path, dtype: type) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=dtype)
    return data.reshape(dims)


def load_fashion_mnist() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load Fashion-MNIST from local gz files, else keras built-in download."""
    needed = [
        FASHION_DIR / "train-images-idx3-ubyte.gz",
        FASHION_DIR / "train-labels-idx1-ubyte.gz",
        FASHION_DIR / "t10k-images-idx3-ubyte.gz",
        FASHION_DIR / "t10k-labels-idx1-ubyte.gz",
    ]
    if all(p.exists() for p in needed):
        x_train = _read_idx(needed[0], np.uint8)
        y_train = _read_idx(needed[1], np.uint8)
        x_test = _read_idx(needed[2], np.uint8)
        y_test = _read_idx(needed[3], np.uint8)
        logger.info("Loaded Fashion-MNIST from %s", FASHION_DIR)
        return (x_train, y_train), (x_test, y_test)

    logger.info("Local Fashion-MNIST missing; downloading via keras (first run).")
    from keras.datasets import fashion_mnist  # noqa: PLC0415

    return fashion_mnist.load_data()


def _preprocess_one(image28: np.ndarray) -> np.ndarray:
    """Upscale one 28x28 grayscale image to 224x224 RGB float in [0,1]."""
    from PIL import Image  # noqa: PLC0415

    rgb = np.repeat(image28[..., np.newaxis], 3, axis=-1)
    resized = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


class FashionSequence(_SEQUENCE_BASE):
    """Keras-compatible generator upscaling 28x28 batches on demand."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int) -> None:
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self) -> int:
        return ceil(len(self.images) / self.batch_size)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.images))
        batch = self.images[start:end]

        x = np.empty((len(batch), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for i, img in enumerate(batch):
            x[i] = _preprocess_one(img)
        y = self.labels[start:end]
        return x, y


def build_model() -> object:
    import tensorflow as tf  # noqa: PLC0415
    from keras.applications import MobileNetV2  # noqa: PLC0415
    from keras.layers import Dense, Dropout, GlobalAveragePooling2D  # noqa: PLC0415
    from keras.models import Sequential  # noqa: PLC0415

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    base.trainable = False  # fine-tune only the head first

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 product classifier.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per split (speeds up quick runs)")
    parser.add_argument("--save-to", type=Path, default=settings.MODEL_DIR)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
        x_test, y_test = x_test[: max(200, args.limit // 10)], y_test[: max(200, args.limit // 10)]
    logger.info("Train=%s Test=%s", x_train.shape, x_test.shape)

    train_seq = FashionSequence(x_train, y_train, args.batch_size)
    test_seq = FashionSequence(x_test, y_test, args.batch_size)

    model = build_model()
    model.fit(
        train_seq,
        validation_data=test_seq,
        epochs=args.epochs,
        verbose=1,
    )

    loss, acc = model.evaluate(test_seq, verbose=0)
    logger.info("Test accuracy: %.4f (loss %.4f)", acc, loss)

    args.save_to.mkdir(parents=True, exist_ok=True)
    artifact = args.save_to / "product_mobilenetv2.keras"
    labels_path = args.save_to / "product_labels.json"

    model.save(artifact)
    labels_path.write_text(
        __import__("json").dumps(DEFAULT_RETAIL_LABELS), encoding="utf-8"
    )
    logger.info("Saved %s and %s", artifact, labels_path)


if __name__ == "__main__":
    main()
