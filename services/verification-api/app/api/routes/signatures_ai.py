"""AI V2 signature-verification endpoints (dedicated, additive).

The existing OpenCV baseline endpoint (/api/signatures/compare) is NOT modified.
The AI endpoint returns a raw cosine similarity (no threshold, no probability).
When the AI model is disabled or fails to load, the endpoint reports a
controlled 503 "unavailable" response and the OpenCV baseline keeps working.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.schemas.ai_signature_comparison import (
    AiSignatureComparisonResponse,
    AiStatusResponse,
)
from app.services.ai_signature_verification_service import (
    EMBEDDING_DIMENSION,
    METHOD,
    MODEL_NAME,
    VERSION,
    AIError,
    AIImageError,
    AISignatureVerificationService,
    get_ai_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


def _read_upload(file: UploadFile, field_name: str) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected unsupported content type for %s: %s", field_name, file.content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Type de contenu non supporté. Formats acceptés : image/jpeg, image/png.",
        )

    content = file.file.read()
    if len(content) == 0:
        logger.warning("Rejected empty file for %s", field_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Le fichier {field_name} est vide.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        logger.warning("Rejected oversized file %s: %s bytes", field_name, len(content))
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Le fichier {field_name} dépasse la taille maximale autorisée (10 Mo).",
        )

    return content


def _unavailable(detail: str) -> JSONResponse:
    """Controlled 'AI unavailable' response (OpenCV baseline stays up)."""
    payload = AiSignatureComparisonResponse(
        success=False,
        similarity_score=None,
        method=METHOD,
        version=VERSION,
        model=MODEL_NAME,
        embedding_dimension=EMBEDDING_DIMENSION,
        device="unavailable",
        message=detail,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.post(
    "/api/signatures/compare-ai",
    response_model=AiSignatureComparisonResponse,
    tags=["signatures"],
    status_code=status.HTTP_200_OK,
)
async def compare_ai_endpoint(
    extracted_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
) -> AiSignatureComparisonResponse:
    extracted_bytes = _read_upload(extracted_file, "extracted_file")
    reference_bytes = _read_upload(reference_file, "reference_file")

    settings = get_settings()
    try:
        service = get_ai_service(settings)
        result = service.compare_signatures_ai(extracted_bytes, reference_bytes)
    except AIImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AIError as exc:
        logger.warning("AI comparison unavailable: %s", exc)
        return _unavailable(str(exc))
    except Exception as exc:  # pragma: no cover - unexpected internal error
        logger.exception("Unexpected error during AI signature comparison")
        return _unavailable("Erreur interne lors de la comparaison IA des signatures.")

    logger.info(
        "AI comparison OK: similarity=%.6f (device=%s)",
        result["similarity_score"],
        result["device"],
    )

    return AiSignatureComparisonResponse(
        success=True,
        similarity_score=result["similarity_score"],
        method=result["method"],
        version=result["version"],
        model=result["model"],
        embedding_dimension=result["embedding_dimension"],
        device=result["device"],
        message="Comparaison IA effectuée.",
    )


@router.get(
    "/api/signatures/ai-status",
    response_model=AiStatusResponse,
    tags=["signatures"],
)
async def ai_status_endpoint() -> AiStatusResponse:
    settings = get_settings()
    if not settings.signature_ai_enabled:
        return AiStatusResponse(
            enabled=False,
            loaded=False,
            model=MODEL_NAME,
            version=VERSION,
            device="disabled",
        )

    try:
        service = get_ai_service(settings)
    except AIError:
        return AiStatusResponse(
            enabled=True,
            loaded=False,
            model=MODEL_NAME,
            version=VERSION,
            device="unavailable",
        )

    return AiStatusResponse(
        enabled=True,
        loaded=True,
        model=service.model_name,
        version=service.version,
        device=service.device_str,
    )