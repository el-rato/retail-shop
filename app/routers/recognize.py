"""POST /recognize-face endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.database.session import get_db
from app.models.face_model import FaceRecognitionError
from app.schemas.api import FaceRecognitionRequest, FaceRecognitionResponse
from app.services import face_service
from app.utils.image_utils import decode_base64_image

router = APIRouter(prefix="/recognize-face", tags=["Face Recognition"])


@router.post("", response_model=FaceRecognitionResponse)
def recognize_face(
    payload: FaceRecognitionRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Detect and recognise a returning customer from a base64 image."""
    image_bgr = decode_base64_image(payload.image_base64)
    try:
        return face_service.recognize_or_register(
            db,
            image_bgr,
            enroll=payload.enroll,
            customer_name=payload.customer_name,
            location=payload.location,
        )
    except FaceRecognitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
