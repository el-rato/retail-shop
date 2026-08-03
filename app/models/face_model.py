"""Face detection & encoding service.

Primary engine: ``face_recognition`` (dlib) for 128-d encodings.
If dlib is unavailable (e.g. missing build tools), we gracefully fall back
to OpenCV Haar cascades for detection-only mode with a synthetic encoding,
so the API stays usable on constrained/CI machines.
"""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - depends on optional native lib
    import face_recognition as _fr

    FACE_RECOGNITION_AVAILABLE = True
    ENCODING_DIM = 128
except ImportError:  # pragma: no cover
    _fr = None
    FACE_RECOGNITION_AVAILABLE = False
    ENCODING_DIM = 128


class FaceRecognitionError(RuntimeError):
    """Raised when no faces can be processed in an image."""


_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _yunet_model_path() -> object | None:
    """Download (once) and return the YuNet ONNX model path, or None on failure."""
    import urllib.request  # noqa: PLC0415

    from app.core.config import settings  # noqa: PLC0415

    dest = settings.DATA_DIR / "yunet" / "face_detection_yunet.onnx"
    try:
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading YuNet face detector model...")
            urllib.request.urlretrieve(_YUNET_MODEL_URL, dest)
        return dest
    except Exception as exc:
        logger.error("Could not obtain YuNet model: %s", exc)
        return None


class FaceDetector:
    """Detect faces and produce encodings, using dlib when present.

    Fallback (no dlib): OpenCV FaceDetectorYN (YuNet) for detection. Encoding
    extraction without dlib uses a deterministic pseudo-encoding, which only
    supports demo/registration flows; install face_recognition for production
    face matching.
    """

    def __init__(self) -> None:
        self._detector = None
        self._model_path = None
        if not FACE_RECOGNITION_AVAILABLE:

            self._model_path = _yunet_model_path()
            logger.warning(
                "face_recognition unavailable; using OpenCV YuNet fallback "
                "(detection only)."
            )
        logger.info(
            "FaceDetector ready (engine=%s)",
            "dlib" if FACE_RECOGNITION_AVAILABLE else "opencv-yunet",
        )

    @property
    def using_dlib(self) -> bool:
        return FACE_RECOGNITION_AVAILABLE

    def find_face_locations(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return list of (top, right, bottom, left) face locations."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise FaceRecognitionError("Face detection requires a 3-channel BGR image.")

        if FACE_RECOGNITION_AVAILABLE:
            try:
                rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
                return _fr.face_locations(rgb, model=settings.FACE_ENCODING_MODEL)
            except Exception as exc:
                logger.warning("dlib location failed (%s); retrying with HOG", exc)
                rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
                return _fr.face_locations(rgb, model="hog")

        # OpenCV YuNet fallback
        import cv2  # noqa: PLC0415

        if self._model_path is None:
            raise FaceRecognitionError(
                "Face detection unavailable: no dlib and YuNet model could not be downloaded."
            )
        if self._detector is None:
            height, width = image_bgr.shape[:2]
            self._detector = cv2.FaceDetectorYN.create(
                str(self._model_path), "", (width, height), score_threshold=0.6
            )
        height, width = image_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _ok, faces = self._detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return []
        locations: list[tuple[int, int, int, int]] = []
        for face in faces:
            x, y, w, h = (int(v) for v in face[:4])
            locations.append((y, x + w, y + h, x))
        return locations

    def face_encodings(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Compute 128-d encodings for every detected face."""
        locations = self.find_face_locations(image_bgr)
        if not locations:
            return []
        if FACE_RECOGNITION_AVAILABLE:
            rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
            try:
                return _fr.face_encodings(rgb, known_face_locations=locations)
            except Exception as exc:  # pragma: no cover
                raise FaceRecognitionError(f"Could not compute face encodings: {exc}") from exc
        # Fallback: deterministic pseudo-encoding derived from the face crop
        encodings: list[np.ndarray] = []
        for top, right, bottom, left in locations:
            crop = image_bgr[top:bottom, left:right]
            resized = cv2_resize(crop, (32, 32))
            flat = resized.ravel().astype(np.float32) / 255.0
            padded = np.zeros(ENCODING_DIM, dtype=np.float32)
            padded[: min(flat.size, ENCODING_DIM)] = flat[: ENCODING_DIM]
            encodings.append(padded / (np.linalg.norm(padded) + 1e-8))
        return encodings

    def face_distance(
        self, known_encodings: list[np.ndarray], query_encoding: np.ndarray
    ) -> list[float]:
        """Euclidean distance between a query encoding and known encodings."""
        return [float(np.linalg.norm(query_encoding - k)) for k in known_encodings]


def cv2_resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Safe resize that works even when cv2 is fully imported."""
    import cv2  # noqa: PLC0415

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def best_match(
    encodings: list[np.ndarray], reference: np.ndarray, tolerance: float | None = None
) -> tuple[int, float] | None:
    """Return (index, distance) of the closest encoding below tolerance."""
    if not encodings:
        return None
    tol = settings.FACE_TOLERANCE if tolerance is None else tolerance
    distances = [float(np.linalg.norm(reference - k)) for k in encodings]
    index = int(np.argmin(distances))
    if distances[index] <= tol:
        return index, distances[index]
    return None
