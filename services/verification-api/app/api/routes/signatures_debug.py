"""Development-only diagnostic endpoint for signature extraction.

This router is intentionally registered ONLY when the application runs in the
``development`` environment (see ``app.main``). In any other environment the
route is not registered at all, so a request to ``/api/signatures/debug`` is
answered with 404 by the framework.

It exposes annotated debug images (ROI + bounding box drawn on copies) used to
calibrate the signature search region. It never modifies the source image and
does NOT constitute a conformity/authenticity check.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.routes.signatures import (
    ALLOWED_CONTENT_TYPES,
    MAX_FILE_SIZE_BYTES,
)
from app.core.config import get_settings
from app.schemas.signature_debug import (
    CropCompleteness,
    DirectionalAcceptDetail,
    FallbackCandidate,
    FallbackRejectedCandidate,
    RecoveredStroke,
    SignatureDebugResponse,
)
from app.schemas.signature_extraction import BoundingBox
from app.services.image_processing_service import ImageProcessingError, decode_image
from app.services.signature_debug_service import (
    SignatureDebugResult,
    build_signature_debug,
)
from app.services.signature_extraction_service import SignatureExtractionError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/api/signatures/debug",
    response_model=SignatureDebugResponse,
    tags=["signatures"],
    status_code=status.HTTP_200_OK,
)
async def signature_debug_endpoint(
    file: UploadFile = File(...),
) -> SignatureDebugResponse:
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
        result: SignatureDebugResult = build_signature_debug(
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
            core_ink_distance_ratio=settings.signature_core_ink_distance_ratio,
            core_max_h_gap_ratio=settings.signature_core_max_h_gap_ratio,
            core_max_v_gap_ratio=settings.signature_core_max_v_gap_ratio,
            core_min_height_ratio=settings.signature_core_min_height_ratio,
            core_directional_h_gap_ratio=settings.signature_core_directional_h_gap_ratio,
            core_directional_ink_distance_ratio=settings.signature_core_directional_ink_distance_ratio,
            completeness_min_factor=settings.signature_completeness_min_factor,
            min_credible_quality=settings.signature_hybrid_min_credible_quality,
        )
    except SignatureExtractionError as exc:
        logger.warning("Debug extraction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected internal error
        logger.exception("Unexpected error during debug extraction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors du diagnostic d'extraction.",
        ) from exc

    return SignatureDebugResponse(
        success=True,
        extraction_pipeline_version=result.extraction_pipeline_version,
        original_width=result.original_width,
        original_height=result.original_height,
        candidate_roi=BoundingBox(
            x=result.candidate_roi.x,
            y=result.candidate_roi.y,
            width=result.candidate_roi.width,
            height=result.candidate_roi.height,
        ),
        analysis_zone=(
            BoundingBox(
                x=result.analysis_zone.x,
                y=result.analysis_zone.y,
                width=result.analysis_zone.width,
                height=result.analysis_zone.height,
            )
            if result.analysis_zone is not None
            else None
        ),
        micr_band=(
            BoundingBox(
                x=result.micr_band.x,
                y=result.micr_band.y,
                width=result.micr_band.width,
                height=result.micr_band.height,
            )
            if result.micr_band is not None
            else None
        ),
        signature_bbox=(
            BoundingBox(
                x=result.signature_bbox.x,
                y=result.signature_bbox.y,
                width=result.signature_bbox.width,
                height=result.signature_bbox.height,
            )
            if result.signature_bbox is not None
            else None
        ),
        total_component_count=result.total_component_count,
        retained_component_count=result.retained_component_count,
        rejected_component_count=result.rejected_component_count,
        micr_rejected_count=result.micr_rejected_count,
        group_count=result.group_count,
        groups=[
            {
                "index": g.index,
                "component_count": g.component_count,
                "bbox": BoundingBox(
                    x=g.bbox.x,
                    y=g.bbox.y,
                    width=g.bbox.width,
                    height=g.bbox.height,
                ),
                "ink_area": g.ink_area,
                "score": g.score,
                "selected": g.selected,
            }
            for g in result.groups
        ],
        selected_group_index=result.selected_group_index,
        selection_reason=result.selection_reason,
        rejected_component_reasons=result.rejected_component_reasons,
        component_count=result.retained_component_count,
        dominant_component_bbox=(
            BoundingBox(
                x=result.dominant_component_bbox.x,
                y=result.dominant_component_bbox.y,
                width=result.dominant_component_bbox.width,
                height=result.dominant_component_bbox.height,
            )
            if result.dominant_component_bbox is not None
            else None
        ),
        dominant_component_ink=result.dominant_component_ink,
        dominant_component_score=result.dominant_component_score,
        refined_component_count=result.refined_component_count,
        core_component_indices=result.core_component_indices,
        discarded_from_selected_group_indices=result.discarded_from_selected_group_indices,
        refined_signature_bbox=(
            BoundingBox(
                x=result.refined_signature_bbox.x,
                y=result.refined_signature_bbox.y,
                width=result.refined_signature_bbox.width,
                height=result.refined_signature_bbox.height,
            )
            if result.refined_signature_bbox is not None
            else None
        ),
        refinement_reason=result.refinement_reason,
        directional_accepted_indices=list(result.directional_accepted_indices),
        directional_accept_details=[
            DirectionalAcceptDetail(
                component_index=d.component_index,
                anchor_index=d.anchor_index,
                h_gap=d.h_gap,
                v_gap=d.v_gap,
                ink_distance=d.ink_distance,
                h_gap_limit=d.h_gap_limit,
                ink_distance_limit=d.ink_distance_limit,
                height=d.height,
            )
            for d in result.directional_accept_details
        ],
        completeness_score=result.completeness_score,
        completeness_refined_ink=result.completeness_refined_ink,
        completeness_reference_ink=result.completeness_reference_ink,
        quality_base=result.quality_base,
        quality_completeness_factor=result.quality_completeness_factor,
        extraction_quality=result.extraction_quality,
        message=result.message,
        localization_mode=result.localization_mode,
        localization_label=result.localization_label,
        roi_failure_reason=result.roi_failure_reason,
        fallback_candidate_count=result.fallback_candidate_count,
        fallback_selected_index=result.fallback_selected_index,
        fallback_selected_score=result.fallback_selected_score,
        fallback_selection_reason=result.fallback_selection_reason,
        final_crop_width=result.final_crop_width,
        final_crop_height=result.final_crop_height,
        fallback_rejected_candidates=[
            FallbackRejectedCandidate(
                index=c.index,
                reason=c.reason,
                bbox=BoundingBox(
                    x=c.bbox.x,
                    y=c.bbox.y,
                    width=c.bbox.width,
                    height=c.bbox.height,
                ),
            )
            for c in result.fallback_rejected_candidates
        ],
        fallback_candidates=[
            FallbackCandidate(
                index=c.index,
                bbox=BoundingBox(
                    x=c.bbox.x,
                    y=c.bbox.y,
                    width=c.bbox.width,
                    height=c.bbox.height,
                ),
                width=c.width,
                height=c.height,
                aspect=c.aspect,
                true_density=c.true_density,
                component_count=c.component_count,
                ink_relative=c.ink_relative,
                vertical_extent=c.vertical_extent,
                score=c.score,
                status=c.status,
                selected=c.selected,
            )
            for c in result.fallback_candidates
        ],
        primary_roi=(
            BoundingBox(
                x=result.primary_roi.x,
                y=result.primary_roi.y,
                width=result.primary_roi.width,
                height=result.primary_roi.height,
            )
            if result.primary_roi is not None
            else None
        ),
        roi_candidate_found=result.roi_candidate_found,
        fallback_executed=result.fallback_executed,
        crop_completeness=(
            CropCompleteness(
                initial_bbox=(
                    BoundingBox(
                        x=result.crop_completeness.initial_bbox.x,
                        y=result.crop_completeness.initial_bbox.y,
                        width=result.crop_completeness.initial_bbox.width,
                        height=result.crop_completeness.initial_bbox.height,
                    )
                    if result.crop_completeness.initial_bbox is not None
                    else None
                ),
                final_bbox=(
                    BoundingBox(
                        x=result.crop_completeness.final_bbox.x,
                        y=result.crop_completeness.final_bbox.y,
                        width=result.crop_completeness.final_bbox.width,
                        height=result.crop_completeness.final_bbox.height,
                    )
                    if result.crop_completeness.final_bbox is not None
                    else None
                ),
                initial_component_count=result.crop_completeness.initial_component_count,
                recovered_component_count=result.crop_completeness.recovered_component_count,
                recovered_left=result.crop_completeness.recovered_left,
                recovered_right=result.crop_completeness.recovered_right,
                recovered_other=result.crop_completeness.recovered_other,
                iterations=result.crop_completeness.iterations,
                expansion_ratio=result.crop_completeness.expansion_ratio,
                status=result.crop_completeness.status,
                recovered_strokes=[
                    RecoveredStroke(
                        direction=s.direction,
                        bbox=BoundingBox(
                            x=s.bbox.x,
                            y=s.bbox.y,
                            width=s.bbox.width,
                            height=s.bbox.height,
                        ),
                        h_gap=s.h_gap,
                        v_gap=s.v_gap,
                        reason=s.reason,
                    )
                    for s in result.crop_completeness.recovered_strokes
                ],
            )
            if result.crop_completeness is not None
            else None
        ),
        image_format=result.image_format,
        original_with_roi_base64=result.original_with_roi_base64,
        roi_image_base64=result.roi_image_base64,
        mask_image_base64=result.mask_image_base64,
        components_all_base64=result.components_all_base64,
        components_rejected_base64=result.components_rejected_base64,
        components_image_base64=result.components_image_base64,
        group_image_base64=result.group_image_base64,
        groups_image_base64=result.groups_image_base64,
        refined_group_image_base64=result.refined_group_image_base64,
        fallback_candidates_image_base64=result.fallback_candidates_image_base64,
        signature_image_base64=result.signature_image_base64,
    )