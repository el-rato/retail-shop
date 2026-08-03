"""Image helpers: base64 decoding, validation, and NumPy/OpenCV conversion."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.exceptions import InvalidImageError


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64 image string into a BGR NumPy array (OpenCV format)."""
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise InvalidImageError("Image is not valid base64.") from exc

    if not raw:
        raise InvalidImageError("Image payload is empty.")
    if len(raw) > settings.MAX_IMAGE_BYTES:
        raise InvalidImageError(f"Image exceeds {settings.MAX_IMAGE_BYTES} bytes limit.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise InvalidImageError("Could not decode image bytes.") from exc

    if image.mode not in ("RGB", "L", "RGBA", "P"):
        raise InvalidImageError(f"Unsupported image mode: {image.mode}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    array = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load raw image bytes into a BGR NumPy array."""
    try:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise InvalidImageError("Could not read image bytes.") from exc
    if image is None:
        raise InvalidImageError("Image bytes could not be decoded by OpenCV.")
    return image


def to_base64(image_bgr: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR image back to a base64 string (useful for debugging)."""
    ok, buf = cv2.imencode(fmt, image_bgr)
    if not ok:
        raise InvalidImageError("Could not encode image.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def preprocess_product_image(image_bgr: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize + convert product image to MobileNetV2-expected RGB float input."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
