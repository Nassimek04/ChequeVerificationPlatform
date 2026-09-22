"""Cheque OCR endpoint — lazy PaddleOCR, structured 503 on unavailable."""

import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.cheque_ocr import ChequeOcrResponse, OcrFields, OcrLine
from app.services.cheque_ocr_service import (
    MAX_OCR_FILE_SIZE_BYTES,
    OcrImageError,
    OcrUnavailableError,
    get_ocr_status,
    run_ocr,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@router.post(
    "/api/cheques/ocr",
    response_model=ChequeOcrResponse,
    tags=["cheques"],
    status_code=status.HTTP_200_OK,
)
async def cheque_ocr_endpoint(file: UploadFile = File(...)) -> ChequeOcrResponse:
    # Content-Type allowlist (additional, not trusted alone)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("OCR rejected unsupported content type: %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Type de contenu non supporté. Formats acceptés : image/jpeg, image/png.",
        )

    content = await file.read()
    if len(content) == 0:
        logger.warning("OCR rejected empty file: %s", file.filename)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Le fichier est vide.")
    if len(content) > MAX_OCR_FILE_SIZE_BYTES:
        logger.warning("OCR rejected oversized file: %s bytes", len(content))
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Le fichier dépasse la taille maximale autorisée (10 Mo).",
        )

    # Do not log image bytes
    start = time.perf_counter()
    try:
        result = run_ocr(content)
    except OcrImageError as exc:
        logger.warning("OCR image error: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OcrUnavailableError as exc:
        # Structured 503 like AI unavailable
        status_info = get_ocr_status()
        logger.warning("OCR unavailable (503): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc) if str(exc) else "Service OCR indisponible.",
        ) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected OCR error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'OCR.",
        ) from exc

    # Normalize to DTO (already normalized in service, but ensure Pydantic)
    lines = [OcrLine(text=l["text"], confidence=l["confidence"], box=l["box"]) for l in result["lines"]]
    fields = OcrFields(**result["fields"])

    return ChequeOcrResponse(
        success=True,
        message=result["message"],
        full_text=result["full_text"],
        lines=lines,
        fields=fields,
        processing_ms=result["processing_ms"],
        lang=result["lang"],
        device=result["device"],
    )


@router.get(
    "/api/cheques/ocr-status",
    tags=["cheques"],
    status_code=status.HTTP_200_OK,
)
async def cheque_ocr_status_endpoint() -> dict:
    return get_ocr_status()
