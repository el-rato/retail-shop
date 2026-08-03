"""Face recognition orchestration: recognise / register / record visits."""

from __future__ import annotations

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.database.models import Customer, Visit
from app.models.face_model import FaceDetector, FaceRecognitionError, best_match
from app.services.pipeline import get_pipeline

logger = get_logger(__name__)


def _extract_first_encoding(detector: FaceDetector, image_bgr: np.ndarray) -> np.ndarray:
    encodings = detector.face_encodings(image_bgr)
    if not encodings:
        raise FaceRecognitionError("No face detected in the provided image.")
    return encodings[0]


def recognize_or_register(
    db: Session,
    image_bgr: np.ndarray,
    *,
    enroll: bool,
    customer_name: str | None,
    location: str | None,
) -> dict:
    """Match a face to a customer; optionally enrol a new one.

    Returns a dict with recognized, customer info, confidence, visits count
    and number of faces detected.
    """
    pipeline = get_pipeline()
    detector = pipeline.face
    locations = detector.find_face_locations(image_bgr)
    faces_detected = len(locations)
    encoding = _extract_first_encoding(detector, image_bgr)

    # Gather all stored encodings and their owners
    customers = db.execute(select(Customer)).scalars().all()
    known_encodings: list[np.ndarray] = []
    owner_index: list[tuple[Customer, int]] = []
    for customer in customers:
        for idx, enc in enumerate(customer.face_encodings or []):
            known_encodings.append(np.asarray(enc, dtype=np.float32))
            owner_index.append((customer, idx))

    match = best_match(known_encodings, encoding) if known_encodings else None

    if match is not None:
        index, distance = match
        customer = owner_index[index][0]
        # Convert distance to a 0-1 similarity score
        similarity = max(0.0, min(1.0, 1.0 - distance / 1.5))
        _record_visit(db, customer, similarity, location)
        visit_count = _customer_visit_count(db, customer.id)
        logger.info("Recognised customer id=%s name=%s sim=%.3f", customer.id, customer.name, similarity)
        return {
            "recognized": True,
            "customer_id": customer.id,
            "customer_name": customer.name,
            "confidence": round(similarity, 4),
            "visits": visit_count,
            "faces_detected": faces_detected,
            "message": f"Welcome back, {customer.name}!",
        }

    if not enroll:
        return {
            "recognized": False,
            "customer_id": None,
            "customer_name": None,
            "confidence": None,
            "visits": 0,
            "faces_detected": faces_detected,
            "message": "Unknown face. Set register=true to enrol this customer.",
        }

    # Register a new customer
    name = (customer_name or "Customer").strip() or "Customer"
    customer = Customer(name=name, face_encodings=[encoding.tolist()])
    db.add(customer)
    db.commit()
    db.refresh(customer)
    _record_visit(db, customer, 1.0, location)
    logger.info("Registered new customer id=%s name=%s", customer.id, name)
    return {
        "recognized": True,
        "customer_id": customer.id,
        "customer_name": name,
        "confidence": 1.0,
        "visits": 1,
        "faces_detected": faces_detected,
        "message": f"New customer '{name}' enrolled and visit recorded.",
    }


def _record_visit(db: Session, customer: Customer, confidence: float, location: str | None) -> None:
    visit = Visit(customer_id=customer.id, confidence=confidence, location=location)
    db.add(visit)
    db.commit()
    db.refresh(visit)


def _customer_visit_count(db: Session, customer_id: int) -> int:
    return db.execute(
        select(func.count(Visit.id)).where(Visit.customer_id == customer_id)
    ).scalar_one()
