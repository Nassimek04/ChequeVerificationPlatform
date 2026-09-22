import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.signature_comparison import (
    SignatureComparisonMetrics,
    SignatureComparisonResponse,
)
from app.services.signature_comparison_service import (
    METHOD,
    VERSION,
    SignatureComparisonError,
    SignatureDecodeError,
    compare_signatures,
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


@router.post(
    "/api/signatures/compare",
    response_model=SignatureComparisonResponse,
    tags=["signatures"],
    status_code=status.HTTP_200_OK,
)
async def compare_signatures_endpoint(
    extracted_file: UploadFile = File(...),
    reference_file: UploadFile = File(...),
) -> SignatureComparisonResponse:
    extracted_bytes = _read_upload(extracted_file, "extracted_file")
    reference_bytes = _read_upload(reference_file, "reference_file")

    settings = get_settings()

    try:
        result = compare_signatures(
            extracted_bytes,
            reference_bytes,
            canvas_width=settings.signature_comparison_canvas_width,
            canvas_height=settings.signature_comparison_canvas_height,
        )
    except SignatureDecodeError as exc:
        logger.warning("Signature comparison failed (undecodable): %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SignatureComparisonError as exc:
        logger.warning("Signature comparison failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected internal error
        logger.exception("Unexpected error during signature comparison")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la comparaison des signatures.",
        ) from exc

    logger.info(
        "Comparison OK: score %.6f (overlap=%.4f, corr=%.4f, density=%.4f)",
        result.similarity_score,
        result.mask_overlap,
        result.normalized_correlation,
        result.density_similarity,
    )

    return SignatureComparisonResponse(
        success=True,
        similarity_score=result.similarity_score,
        method=METHOD,
        version=VERSION,
        metrics=SignatureComparisonMetrics(
            mask_overlap=result.mask_overlap,
            normalized_correlation=result.normalized_correlation,
            density_similarity=result.density_similarity,
        ),
    )