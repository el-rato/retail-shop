"""POST /classify-product endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.database.session import get_db
from app.schemas.api import ClassifyProductRequest, ClassifyProductResponse
from app.services import product_service
from app.utils.image_utils import decode_base64_image

router = APIRouter(prefix="/classify-product", tags=["Product Classification"])


@router.post("", response_model=ClassifyProductResponse)
def classify_product(
    payload: ClassifyProductRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(require_api_key),
) -> dict:
    """Predict the category of a product from a base64 image."""
    image_bgr = decode_base64_image(payload.image_base64)
    return product_service.classify_product(db, image_bgr)
