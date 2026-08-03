"""Tests for image utilities and face recognition orchestration."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.exceptions import InvalidImageError
from app.database.models import Customer
from app.services import face_service
from app.utils.image_utils import decode_base64_image, preprocess_product_image


def _fake_encoding(seed: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed * 1000))
    enc = rng.random(128, dtype=np.float32)
    return enc / np.linalg.norm(enc)


def _image_base64() -> str:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (200, 120, 40)
    import cv2

    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class TestImageUtils:
    def test_decode_valid_base64(self):
        b64 = _image_base64()
        image = decode_base64_image(b64)
        assert image.ndim == 3 and image.shape[2] == 3

    def test_decode_invalid_base64_raises(self):
        with pytest.raises(InvalidImageError):
            decode_base64_image("not!!base64!!")

    def test_decode_empty_raises(self):
        with pytest.raises(InvalidImageError):
            decode_base64_image("")

    def test_preprocess_product(self):
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        x = preprocess_product_image(image)
        assert x.shape == (224, 224, 3)
        assert 0.0 <= x.min() <= x.max() <= 1.0


class TestFaceRecognition:
    def _stub_pipeline(self, monkeypatch, known_encodings, encoding):
        detector = MagicMock()
        detector.find_face_locations.return_value = [(0, 100, 100, 0)]
        detector.face_encodings.return_value = [encoding]
        detector.face_distance.return_value = [0.2]

        fake_pipeline = MagicMock()
        fake_pipeline.face = detector
        monkeypatch.setattr(face_service, "get_pipeline", lambda: fake_pipeline)
        return fake_pipeline

    def test_registers_new_customer(self, db, monkeypatch):
        encoding = _fake_encoding(1.0)
        self._stub_pipeline(monkeypatch, known_encodings=[], encoding=encoding)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = face_service.recognize_or_register(
            db, image, enroll=True, customer_name="New Person", location="Main St"
        )
        assert result["recognized"] is True
        assert result["customer_name"] == "New Person"
        assert result["visits"] == 1
        assert db.query(Customer).filter(Customer.name == "New Person").first() is not None

    def test_unknown_face_no_register(self, db, monkeypatch):
        encoding = _fake_encoding(2.0)
        self._stub_pipeline(monkeypatch, known_encodings=[], encoding=encoding)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = face_service.recognize_or_register(
            db, image, enroll=False, customer_name=None, location=None
        )
        assert result["recognized"] is False
        assert result["message"].startswith("Unknown face")

    def test_recognizes_existing_customer(self, db, monkeypatch):
        reference = _fake_encoding(3.0)
        customer = Customer(name="Known", face_encodings=[reference.tolist()])
        db.add(customer)
        db.commit()
        db.refresh(customer)

        encoding = _fake_encoding(3.0)
        self._stub_pipeline(monkeypatch, known_encodings=[encoding], encoding=encoding)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = face_service.recognize_or_register(
            db, image, enroll=False, customer_name=None, location=None
        )
        assert result["recognized"] is True
        assert result["customer_name"] == "Known"
        assert result["visits"] >= 1

    def test_no_face_raises(self, db, monkeypatch):
        detector = MagicMock()
        detector.find_face_locations.return_value = [(0, 100, 100, 0)]
        detector.face_encodings.return_value = []
        fake_pipeline = MagicMock()
        fake_pipeline.face = detector
        monkeypatch.setattr(face_service, "get_pipeline", lambda: fake_pipeline)

        with pytest.raises(face_service.FaceRecognitionError):
            face_service.recognize_or_register(
                db, np.zeros((100, 100, 3), dtype=np.uint8),
                enroll=False, customer_name=None, location=None,
            )
