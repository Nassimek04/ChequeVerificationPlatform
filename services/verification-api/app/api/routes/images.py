import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.image_analysis import (
    ImageAnalysisResponse,
    ImageProcessingInfo,
)
from app.services.image_processing_service import (
    ImageAnalysisResult,
    ImageProcessingError,
    analyze_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@router.post(
    "/api/images/analyze",
    response_model=ImageAnalysisResponse,
    tags=["images"],
    status_code=status.HTTP_200_OK,
)
async def analyze_image_endpoint(file: UploadFile = File(...)) -> ImageAnalysisResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected unsupported content type: %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Type de contenu non supporté. Formats acceptés : image/jpeg, image/png.",
        )

    content = await file.read()
    if len(content) == 0:
        logger.warning("Rejected empty file: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier est vide.",
        )

    if len(content) > MAX_FILE_SIZE_BYTES:
        logger.warning("Rejected oversized file: %s bytes", len(content))
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Le fichier dépasse la taille maximale autorisée (10 Mo).",
        )

    try:
        result: ImageAnalysisResult = analyze_image(content)
    except ImageProcessingError as exc:
        logger.warning("Image decoding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected internal error
        logger.exception("Unexpected error during image analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'analyse de l'image.",
        ) from exc

    return ImageAnalysisResponse(
        success=True,
        width=result.width,
        height=result.height,
        channels=result.channels,
        content_type=file.content_type,
        processing=ImageProcessingInfo(
            decoded=True,
            grayscale_ready=result.grayscale_ready,
        ),
    )