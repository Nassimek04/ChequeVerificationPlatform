import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.signature_extraction import (
    BoundingBox,
    SignatureExtractionResponse,
)
from app.services.image_processing_service import ImageProcessingError, decode_image
from app.services.signature_extraction_service import (
    SignatureExtractionError,
    SignatureExtractionResult,
    extract_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@router.post(
    "/api/signatures/extract",
    response_model=SignatureExtractionResponse,
    tags=["signatures"],
    status_code=status.HTTP_200_OK,
)
async def extract_signature_endpoint(
    file: UploadFile = File(...),
) -> SignatureExtractionResponse:
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
        image = decode_image(content)
    except ImageProcessingError as exc:
        logger.warning("Image decoding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    settings = get_settings()

    try:
        result: SignatureExtractionResult = extract_signature(
            image,
            roi_x_start=settings.signature_roi_x_start,
            roi_y_start=settings.signature_roi_y_start,
            roi_x_end=settings.signature_roi_x_end,
            roi_y_end=settings.signature_roi_y_end,
            bbox_margin=settings.signature_bbox_margin,
            micr_zone_ratio=settings.signature_bottom_exclusion_ratio,
            min_component_area_ratio=settings.signature_min_component_area_ratio,
            max_component_area_ratio=settings.signature_max_component_area_ratio,
            min_component_width_ratio=settings.signature_min_component_width_ratio,
            min_component_height_ratio=settings.signature_min_component_height_ratio,
            micr_max_height_ratio=settings.signature_micr_max_height_ratio,
            micr_min_aspect_ratio=settings.signature_micr_min_aspect_ratio,
            micr_min_width_ratio=settings.signature_micr_min_width_ratio,
            component_merge_distance_ratio=settings.signature_component_merge_distance_ratio,
            group_min_components=settings.signature_group_min_components,
            morph_kernel_size=settings.signature_morph_kernel_size,
            min_credible_quality=settings.signature_hybrid_min_credible_quality,
        )
    except SignatureExtractionError as exc:
        logger.warning("Signature extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected internal error
        logger.exception("Unexpected error during signature extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de l'extraction de la signature.",
        ) from exc

    logger.info(
        "Extraction OK: image %dx%d, ROI %dx%d, quality %.4f",
        result.original_width,
        result.original_height,
        result.candidate_roi.width,
        result.candidate_roi.height,
        result.extraction_quality,
    )

    return SignatureExtractionResponse(
        success=True,
        original_width=result.original_width,
        original_height=result.original_height,
        candidate_roi=BoundingBox(
            x=result.candidate_roi.x,
            y=result.candidate_roi.y,
            width=result.candidate_roi.width,
            height=result.candidate_roi.height,
        ),
        signature_bbox=BoundingBox(
            x=result.signature_bbox.x,
            y=result.signature_bbox.y,
            width=result.signature_bbox.width,
            height=result.signature_bbox.height,
        ),
        extraction_quality=result.extraction_quality,
        image_format=result.image_format,
        signature_image_base64=result.signature_image_base64,
        localization_mode=result.localization_mode,
        fallback_candidate_count=result.fallback_candidate_count,
        fallback_selected_index=result.fallback_selected_index,
        fallback_selected_score=result.fallback_selected_score,
        fallback_selection_reason=result.fallback_selection_reason,
        final_crop_width=result.final_crop_width,
        final_crop_height=result.final_crop_height,
    )