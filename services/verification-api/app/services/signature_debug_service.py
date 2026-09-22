"""Development-only diagnostic service for the signature extraction pipeline.

Draws the candidate ROI, the MICR risk zone and the detected bounding box on a
*copy* of the original image so that the search region can be visually
calibrated. The source image is never modified.

The detection logic is strictly the same as the one used by ``extract_signature``
(it reuses the shared ``run_hybrid_localization`` orchestration), so the debug
rectangles always match what the extraction endpoint would use.

Additional diagnostic images (mask, all/rejected/retained components, all
groups, selected signature group) are returned as PNG bytes encoded in Base64,
along with component counts, per-group scores, the selected group index and the
selection/rejection reasons. When the full-document fallback runs, one extra
image shows the fallback candidate regions (graphically rejected groups in
gray, surviving groups in colors) and the selected region in red.

Diagnostic only: this does NOT recognize a signature, does NOT compute an
authenticity/conformity score and does NOT compare signatures.
"""

import base64
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.signature_extraction_service import (
    DEFAULT_MIN_CREDIBLE_QUALITY,
    EXTRACTION_PIPELINE_VERSION,
    LOCALIZATION_LABELS,
    LOCALIZATION_MODE_GLOBAL_FALLBACK,
    LOCALIZATION_MODE_ROI,
    CropCompleteness,
    FallbackCandidateRow,
    Rect,
    SignatureExtractionError,
    _build_final_crop_png,
    run_hybrid_localization,
    DirectionalAcceptDetail,
)

logger = logging.getLogger(__name__)

ROI_COLOR = (0, 255, 0)           # green  (candidate ROI)
MICR_BAND_COLOR = (255, 0, 255)   # magenta (MICR risk zone)
BBOX_COLOR = (0, 0, 255)          # red    (final bbox)
RETAINED_COLOR = (255, 0, 0)      # blue   (kept components)
REJECTED_COLOR = (96, 96, 96)     # gray   (rejected components)
MICR_REJECTED_COLOR = (0, 165, 255)  # orange (rejected as MICR / printed)
CORE_COLOR = (0, 255, 255)        # yellow (V2.3 dominant signature core)
REFINED_COLOR = (255, 128, 0)     # blue-ish (V2.3 refined kept components)
DISCARDED_REFINE_COLOR = (160, 160, 160)  # gray (V2.3 discarded from selected group)
COMPONENT_COLORS = [
    (255, 0, 0),   # blue
    (0, 0, 255),   # red
    (0, 255, 255), # yellow
    (255, 0, 255), # magenta
    (0, 128, 255), # orange
    (128, 0, 255), # violet
]
LINE_THICKNESS = 3

NO_CONTENT_MESSAGE = "Aucun contenu exploitable détecté dans la ROI candidate."

REJECTION_REASON_LABELS = {
    "area_min": "aire_trop_petite",
    "area_max": "aire_trop_grande",
    "thin": "trop_fin",
    "micr": "micr_texte_imprime",
}


@dataclass
class DebugGroupInfo:
    """Per-group diagnostic info (ROI-local bounding box + score)."""

    index: int
    component_count: int
    bbox: Rect
    ink_area: int
    score: float
    selected: bool


@dataclass
class FallbackRejectedCandidate:
    """A full-document fallback group rejected as an obvious document graphic.

    ``index`` is the group index in the pre-filter fallback grouping,
    ``reason`` the truthful French gate reason (see
    ``_graphic_rejection_reason``) and ``bbox`` the rejected union box
    (absolute document coordinates), drawn gray in the candidates image.
    """

    index: int
    reason: str
    bbox: Rect


@dataclass
class SignatureDebugResult:
    extraction_pipeline_version: str
    original_width: int
    original_height: int
    candidate_roi: Rect
    analysis_zone: Rect | None
    micr_band: Rect | None
    signature_bbox: Rect | None
    total_component_count: int
    retained_component_count: int
    rejected_component_count: int
    micr_rejected_count: int
    group_count: int
    groups: list[DebugGroupInfo]
    selected_group_index: int
    selection_reason: str
    rejected_component_reasons: dict
    extraction_quality: float
    # --- V2.3 dominant-core refinement fields ---
    dominant_component_bbox: Rect | None
    dominant_component_ink: int
    dominant_component_score: float
    refined_component_count: int
    core_component_indices: list[int]
    discarded_from_selected_group_indices: list[int]
    refined_signature_bbox: Rect | None
    refinement_reason: str
    # --- V2.4 directional continuation + completeness quality fields ---
    directional_accepted_indices: list[int]
    directional_accept_details: list[DirectionalAcceptDetail]
    completeness_score: float
    completeness_refined_ink: int
    completeness_reference_ink: int
    quality_base: float
    quality_completeness_factor: float
    message: str | None
    # --- Hybrid localization (ROI first, full-document fallback) ---
    # Machine mode ("roi" / "global_fallback") + French label for the Bilan.
    localization_mode: str
    localization_label: str
    # Why the ROI candidate was rejected (empty when the ROI won).
    roi_failure_reason: str
    # Full-document fallback evidence (0 / -1 / "" when the ROI won).
    fallback_candidate_count: int
    fallback_selected_index: int
    fallback_selected_score: float
    fallback_selection_reason: str
    final_crop_width: int
    final_crop_height: int
    original_with_roi_png_bytes: bytes
    roi_image_png_bytes: bytes
    mask_image_png_bytes: bytes | None
    components_all_png_bytes: bytes | None
    components_rejected_png_bytes: bytes | None
    components_retained_png_bytes: bytes | None
    group_image_png_bytes: bytes | None
    groups_image_png_bytes: bytes | None
    refined_group_png_bytes: bytes | None
    # Full-document candidates (each group color) + selected bbox red.
    # Only produced when the fallback ran; None when the ROI won.
    fallback_candidates_png_bytes: bytes | None
    signature_png_bytes: bytes | None
    image_format: str
    # Whole fallback groups rejected as obvious graphics, with truthful
    # reasons (empty when the ROI won or nothing was graphically rejected).
    fallback_rejected_candidates: list[FallbackRejectedCandidate] = field(
        default_factory=list
    )
    # Explainability table: one row per pre-filter fallback group (empty
    # when the ROI won and the fallback never ran).
    fallback_candidates: list[FallbackCandidateRow] = field(default_factory=list)
    # The attempted primary ROI (always set, even when the fallback won).
    primary_roi: Rect | None = None
    # True when the ROI stage produced any candidate bbox (even rejected).
    roi_candidate_found: bool = False
    # True as soon as the full-document fallback executed.
    fallback_executed: bool = False
    # Crop-completeness evidence for the final candidate (None when no
    # refined bbox exists). Shared dataclass with the extraction service.
    crop_completeness: CropCompleteness | None = None

    @property
    def original_with_roi_base64(self) -> str:
        return base64.b64encode(self.original_with_roi_png_bytes).decode("ascii")

    @property
    def roi_image_base64(self) -> str:
        return base64.b64encode(self.roi_image_png_bytes).decode("ascii")

    @property
    def mask_image_base64(self) -> str | None:
        if self.mask_image_png_bytes is None:
            return None
        return base64.b64encode(self.mask_image_png_bytes).decode("ascii")

    @property
    def components_all_base64(self) -> str | None:
        if self.components_all_png_bytes is None:
            return None
        return base64.b64encode(self.components_all_png_bytes).decode("ascii")

    @property
    def components_rejected_base64(self) -> str | None:
        if self.components_rejected_png_bytes is None:
            return None
        return base64.b64encode(self.components_rejected_png_bytes).decode("ascii")

    @property
    def components_image_base64(self) -> str | None:
        """Retained components (kept field name for backward compatibility)."""
        if self.components_retained_png_bytes is None:
            return None
        return base64.b64encode(self.components_retained_png_bytes).decode("ascii")

    @property
    def group_image_base64(self) -> str | None:
        if self.group_image_png_bytes is None:
            return None
        return base64.b64encode(self.group_image_png_bytes).decode("ascii")

    @property
    def groups_image_base64(self) -> str | None:
        if self.groups_image_png_bytes is None:
            return None
        return base64.b64encode(self.groups_image_png_bytes).decode("ascii")

    @property
    def refined_group_image_base64(self) -> str | None:
        if self.refined_group_png_bytes is None:
            return None
        return base64.b64encode(self.refined_group_png_bytes).decode("ascii")

    @property
    def fallback_candidates_image_base64(self) -> str | None:
        if self.fallback_candidates_png_bytes is None:
            return None
        return base64.b64encode(self.fallback_candidates_png_bytes).decode("ascii")

    @property
    def signature_image_base64(self) -> str | None:
        if self.signature_png_bytes is None:
            return None
        return base64.b64encode(self.signature_png_bytes).decode("ascii")


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise SignatureExtractionError("Impossible d'encoder l'image en PNG.")
    return encoded.tobytes()


def _draw_rectangle(
    image: np.ndarray,
    rect: Rect,
    color: tuple[int, int, int],
    thickness: int = LINE_THICKNESS,
) -> None:
    """Draw a rectangle onto ``image`` in place."""
    cv2.rectangle(
        image,
        (rect.x, rect.y),
        (rect.x + rect.width - 1, rect.y + rect.height - 1),
        color,
        thickness,
    )


def _draw_micr_band(
    image: np.ndarray, band: Rect, fill_alpha: float = 0.15
) -> None:
    """Draw the MICR risk zone on ``image`` in place (semi-transparent fill +
    boundary line). ``band`` is expressed in the image coordinate space."""
    if band.height <= 0:
        return

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (band.x, band.y),
        (band.x + band.width - 1, band.y + band.height - 1),
        MICR_BAND_COLOR,
        -1,
    )
    image[:] = cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0)
    cv2.line(
        image,
        (band.x, band.y),
        (band.x + band.width - 1, band.y),
        MICR_BAND_COLOR,
        1,
    )


def _draw_components(
    image: np.ndarray,
    components: list[Rect],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    for component in components:
        _draw_rectangle(image, component, color, thickness)


def build_signature_debug(
    image: np.ndarray,
    roi_x_start: float,
    roi_y_start: float,
    roi_x_end: float,
    roi_y_end: float,
    bbox_margin: float = 0.08,
    micr_zone_ratio: float = 0.10,
    min_component_area_ratio: float = 0.0005,
    max_component_area_ratio: float = 0.60,
    min_component_width_ratio: float = 0.01,
    min_component_height_ratio: float = 0.01,
    micr_max_height_ratio: float = 0.06,
    micr_min_aspect_ratio: float = 6.0,
    micr_min_width_ratio: float = 0.03,
    component_merge_distance_ratio: float = 0.10,
    group_min_components: int = 1,
    morph_kernel_size: int = 3,
    core_ink_distance_ratio: float = 0.018,
    core_max_h_gap_ratio: float = 0.15,
    core_max_v_gap_ratio: float = 0.15,
    core_min_height_ratio: float = 0.08,
    core_directional_h_gap_ratio: float = 0.08,
    core_directional_ink_distance_ratio: float = 0.04,
    completeness_min_factor: float = 0.5,
    min_credible_quality: float = DEFAULT_MIN_CREDIBLE_QUALITY,
) -> SignatureDebugResult:
    """Run the exact extraction pipeline and return annotated debug images.

    Never modifies ``image``: all drawings are made on copies. The hybrid
    localization (``run_hybrid_localization``) is shared verbatim with
    ``extract_signature``: the existing ROI stage images always document the
    primary ROI attempt, and when the full-document fallback runs a single
    additional candidates image shows the fallback groups + selected region.

    If no usable content is found by either stage, the result still contains
    the annotated original (ROI + MICR band drawn), the ROI crop and the ink
    mask, plus an explicit message; ``signature_bbox`` and
    ``signature_png_bytes`` are then None.

    Raises SignatureExtractionError only for hard input failures (undecoded
    image, image too small, invalid ROI).
    """
    if image is None:
        raise SignatureExtractionError("Image non décodée.")

    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise SignatureExtractionError(
            f"Image trop petite pour le diagnostic ({width}x{height})."
        )

    outcome = run_hybrid_localization(
        image,
        roi_x_start,
        roi_y_start,
        roi_x_end,
        roi_y_end,
        bbox_margin=bbox_margin,
        micr_zone_ratio=micr_zone_ratio,
        min_component_area_ratio=min_component_area_ratio,
        max_component_area_ratio=max_component_area_ratio,
        min_component_width_ratio=min_component_width_ratio,
        min_component_height_ratio=min_component_height_ratio,
        micr_max_height_ratio=micr_max_height_ratio,
        micr_min_aspect_ratio=micr_min_aspect_ratio,
        micr_min_width_ratio=micr_min_width_ratio,
        component_merge_distance_ratio=component_merge_distance_ratio,
        group_min_components=group_min_components,
        morph_kernel_size=morph_kernel_size,
        core_ink_distance_ratio=core_ink_distance_ratio,
        core_max_h_gap_ratio=core_max_h_gap_ratio,
        core_max_v_gap_ratio=core_max_v_gap_ratio,
        core_min_height_ratio=core_min_height_ratio,
        core_directional_h_gap_ratio=core_directional_h_gap_ratio,
        core_directional_ink_distance_ratio=core_directional_ink_distance_ratio,
        completeness_min_factor=completeness_min_factor,
        min_credible_quality=min_credible_quality,
    )
    roi = outcome.roi
    if roi.width < 4 or roi.height < 4:
        raise SignatureExtractionError("ROI candidate invalide ou trop petite.")
    primary_roi = outcome.roi
    roi_candidate_found = outcome.roi_analysis.bbox is not None
    fallback_executed = outcome.fallback_analysis is not None
    fallback_candidates: list[FallbackCandidateRow] = (
        list(outcome.fallback_analysis.fallback_candidate_table)
        if outcome.fallback_analysis is not None
        else []
    )

    roi_crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    # Existing stage images always document the PRIMARY ROI attempt.
    roi_analysis = outcome.roi_analysis
    analysis = outcome.roi_analysis
    message = None
    bbox = None
    quality = 0.0
    localization_mode = outcome.localization_mode
    roi_failure_reason = outcome.roi_failure_reason
    fallback_candidate_count = 0
    fallback_selected_index = -1
    fallback_selected_score = 0.0
    fallback_selection_reason = ""
    final_crop_width = 0
    final_crop_height = 0
    fallback_rejected_candidates: list[FallbackRejectedCandidate] = []

    if outcome.final_analysis is not None and outcome.final_analysis.bbox is not None:
        analysis = outcome.final_analysis
        bbox = analysis.bbox
        quality = analysis.quality
        if localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK:
            fallback_candidate_count = len(analysis.groups)
            fallback_selected_index = analysis.selected_group_index
            if 0 <= fallback_selected_index < len(analysis.group_scores):
                fallback_selected_score = round(
                    analysis.group_scores[fallback_selected_index], 6
                )
            fallback_selection_reason = analysis.selection_reason
    else:
        if outcome.fallback_analysis is not None:
            message = (
                "Aucune signature exploitable détectée (ni dans la ROI "
                "principale ni dans l'ensemble du document)."
            )
            fallback_candidate_count = len(outcome.fallback_analysis.groups)
        else:
            message = NO_CONTENT_MESSAGE

    micr_band_abs = Rect(
        x=roi.x,
        y=roi.y + roi_analysis.micr_band_top,
        width=roi.width,
        height=roi.height - roi_analysis.micr_band_top,
    )
    micr_band_local = Rect(
        x=0,
        y=roi_analysis.micr_band_top,
        width=roi.width,
        height=roi.height - roi_analysis.micr_band_top,
    )

    # --- Annotated copy of the original: ROI (green) + MICR band + bbox ---
    # The final bbox is ROI-local in ROI mode and absolute (= full-document
    # coordinates) when the fallback won.
    annotated = image.copy()
    _draw_rectangle(annotated, roi, ROI_COLOR, LINE_THICKNESS)
    _draw_micr_band(annotated, micr_band_abs)
    if bbox is not None:
        if localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK:
            bbox_abs = Rect(
                x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height
            )
        else:
            bbox_abs = Rect(
                x=roi.x + bbox.x,
                y=roi.y + bbox.y,
                width=bbox.width,
                height=bbox.height,
            )
        _draw_rectangle(
            annotated,
            bbox_abs,
            BBOX_COLOR,
            LINE_THICKNESS,
        )
    original_png = _encode_png(annotated)

    # Geometry of the WINNING stage: ROI-local in ROI mode, absolute
    # (= full-document-local) when the fallback won. In ROI mode this is
    # byte-identical to the historical behavior.
    stage_is_full = outcome.final_is_full_image and bbox is not None
    stage_base = image.copy() if stage_is_full else roi_crop.copy()
    stage_roi = (
        Rect(x=0, y=0, width=width, height=height)
        if stage_is_full
        else roi
    )
    stage_band_local = Rect(
        x=0,
        y=analysis.micr_band_top,
        width=stage_roi.width,
        height=stage_roi.height - analysis.micr_band_top,
    )

    # --- ROI crop with MICR band boundary + bbox in the same coordinate space ---
    # Documents the primary ROI attempt: the fallback bbox (absolute) is never
    # drawn here.
    roi_annotated = roi_crop.copy()
    _draw_micr_band(roi_annotated, micr_band_local)
    if bbox is not None and not stage_is_full:
        _draw_rectangle(roi_annotated, bbox, BBOX_COLOR, LINE_THICKNESS)
    roi_png = _encode_png(roi_annotated)

    # --- Binary ink mask of the analyzed zone ---
    mask_png = _encode_png(analysis.binary_mask)

    # --- All components detected (distinct colors) ---
    components_all_img = stage_base.copy()
    for index, component in enumerate(analysis.all_components):
        color = COMPONENT_COLORS[index % len(COMPONENT_COLORS)]
        _draw_rectangle(components_all_img, component, color, 2)
    components_all_png = _encode_png(components_all_img)

    # --- Rejected components (gray; MICR-rejected in orange) ---
    micr_rejected_set = set(analysis.micr_rejected_components)
    components_rejected_img = stage_base.copy()
    for component in analysis.rejected_components:
        color = MICR_REJECTED_COLOR if component in micr_rejected_set else REJECTED_COLOR
        _draw_rectangle(components_rejected_img, component, color, 2)
    components_rejected_png = _encode_png(components_rejected_img)

    # --- Retained components (blue) + bbox red ---
    components_retained_img = stage_base.copy()
    _draw_components(components_retained_img, analysis.kept_components, RETAINED_COLOR, 2)
    if bbox is not None:
        _draw_rectangle(components_retained_img, bbox, BBOX_COLOR, LINE_THICKNESS)
    components_retained_png = _encode_png(components_retained_img)

    # --- Final signature group (distinct colors) + bbox red ---
    group_image_img = stage_base.copy()
    for index, component in enumerate(analysis.primary_group):
        color = COMPONENT_COLORS[index % len(COMPONENT_COLORS)]
        _draw_rectangle(group_image_img, component, color, 2)
    if bbox is not None:
        _draw_rectangle(group_image_img, bbox, BBOX_COLOR, LINE_THICKNESS)
    group_image_png = _encode_png(group_image_img)

    # --- All generated groups (each in a distinct color) + selected group red ---
    groups_image_img = stage_base.copy()
    for gi, group in enumerate(analysis.groups):
        color = COMPONENT_COLORS[gi % len(COMPONENT_COLORS)]
        for component in group:
            _draw_rectangle(groups_image_img, component, color, 2)
    if bbox is not None:
        _draw_rectangle(groups_image_img, bbox, BBOX_COLOR, LINE_THICKNESS)
    groups_image_png = _encode_png(groups_image_img)

    # --- V2.3 refined signature structure (dominant core + refined vs discarded) ---
    refined_group_png = None
    if analysis.primary_group:
        discarded_set = set(analysis.discarded_from_selected_group_indices)
        refined_group_img = stage_base.copy()
        for index, component in enumerate(analysis.primary_group):
            if index in discarded_set:
                _draw_rectangle(
                    refined_group_img, component, DISCARDED_REFINE_COLOR, 2
                )
            elif index == analysis.dominant_core_index_in_group:
                _draw_rectangle(refined_group_img, component, CORE_COLOR, 2)
            else:
                _draw_rectangle(refined_group_img, component, REFINED_COLOR, 2)
        if bbox is not None:
            _draw_rectangle(refined_group_img, bbox, BBOX_COLOR, LINE_THICKNESS)
        refined_group_png = _encode_png(refined_group_img)

    # --- Full-document fallback candidates ---
    # Minimum additional evidence: only produced when the fallback ran.
    # Graphically rejected groups are drawn gray, surviving groups each in a
    # distinct color, and the selected region red.
    fallback_candidates_png = None
    if outcome.fallback_analysis is not None:
        fb = outcome.fallback_analysis
        for rej_index, rej_reason, rej_box in zip(
            fb.graphic_rejected_group_indices,
            fb.graphic_rejected_group_reasons,
            fb.graphic_rejected_group_boxes,
        ):
            fallback_rejected_candidates.append(
                FallbackRejectedCandidate(
                    index=rej_index, reason=rej_reason, bbox=rej_box
                )
            )
        fb_img = image.copy()
        for rej_box in fb.graphic_rejected_group_boxes:
            _draw_rectangle(fb_img, rej_box, REJECTED_COLOR, 2)
        for gi, group in enumerate(fb.groups):
            color = COMPONENT_COLORS[gi % len(COMPONENT_COLORS)]
            for component in group:
                _draw_rectangle(fb_img, component, color, 2)
        if fb.bbox is not None and localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK:
            _draw_rectangle(fb_img, fb.bbox, BBOX_COLOR, LINE_THICKNESS)
        fallback_candidates_png = _encode_png(fb_img)

    # --- Per-group diagnostic info ---
    group_info: list[DebugGroupInfo] = []
    for gi, group in enumerate(analysis.groups):
        x0 = min(c.x for c in group)
        y0 = min(c.y for c in group)
        x1 = max(c.x + c.width for c in group)
        y1 = max(c.y + c.height for c in group)
        ink_area = sum(c.width * c.height for c in group)
        score = (
            analysis.group_scores[gi]
            if gi < len(analysis.group_scores)
            else 0.0
        )
        group_info.append(
            DebugGroupInfo(
                index=gi,
                component_count=len(group),
                bbox=Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0),
                ink_area=ink_area,
                score=score,
                selected=(gi == analysis.selected_group_index),
            )
        )

    # --- Rejected component reasons (internal keys -> French labels) ---
    rejected_component_reasons = {
        REJECTION_REASON_LABELS.get(key, key): count
        for key, count in analysis.rejection_reasons.items()
    }

    selection_reason = analysis.selection_reason

    # --- Final crop (same margin logic as extract_signature) ---
    signature_png = None
    if bbox is not None:
        crop_source = image if stage_is_full else roi_crop
        try:
            crop_bytes, final_crop_width, final_crop_height = (
                _build_final_crop_png(crop_source, bbox, bbox_margin)
            )
            signature_png = crop_bytes
        except SignatureExtractionError:
            signature_png = None
            bbox = None
            message = NO_CONTENT_MESSAGE
            quality = 0.0

    logger.info(
        "Debug extraction V2.4: image %dx%d, ROI %dx%d, mode=%s, micr_band_top=%d, "
        "bbox %s, all=%d kept=%d rejected=%d micr_rejected=%d groups=%d, "
        "selected_group=%d, core=(%d, score %.5f), refined=%d, directional=%d, "
        "discarded=%d, completeness %.3f, quality %.4f, message=%s",
        width,
        height,
        roi.width,
        roi.height,
        localization_mode,
        analysis.micr_band_top,
        bbox,
        len(analysis.all_components),
        len(analysis.kept_components),
        len(analysis.rejected_components),
        len(analysis.micr_rejected_components),
        len(analysis.groups),
        analysis.selected_group_index,
        analysis.dominant_core_index_in_group,
        analysis.dominant_component_score,
        len(analysis.refined_components),
        len(analysis.directional_accepted_indices),
        len(analysis.discarded_from_selected_group_indices),
        analysis.completeness_score,
        quality,
        message,
    )

    return SignatureDebugResult(
        extraction_pipeline_version=EXTRACTION_PIPELINE_VERSION,
        original_width=width,
        original_height=height,
        candidate_roi=stage_roi,
        analysis_zone=Rect(0, 0, stage_roi.width, stage_roi.height),
        micr_band=stage_band_local,
        signature_bbox=bbox,
        localization_mode=localization_mode,
        localization_label=LOCALIZATION_LABELS.get(
            localization_mode, localization_mode
        ),
        roi_failure_reason=roi_failure_reason,
        fallback_candidate_count=fallback_candidate_count,
        fallback_selected_index=fallback_selected_index,
        fallback_selected_score=fallback_selected_score,
        fallback_selection_reason=fallback_selection_reason,
        final_crop_width=final_crop_width,
        final_crop_height=final_crop_height,
        fallback_rejected_candidates=fallback_rejected_candidates,
        fallback_candidates=fallback_candidates,
        crop_completeness=analysis.crop_completeness,
        primary_roi=primary_roi,
        roi_candidate_found=roi_candidate_found,
        fallback_executed=fallback_executed,
        total_component_count=len(analysis.all_components),
        retained_component_count=len(analysis.kept_components),
        rejected_component_count=len(analysis.rejected_components),
        micr_rejected_count=len(analysis.micr_rejected_components),
        group_count=len(analysis.groups),
        groups=group_info,
        selected_group_index=analysis.selected_group_index,
        selection_reason=selection_reason,
        rejected_component_reasons=rejected_component_reasons,
        extraction_quality=round(quality, 4),
        dominant_component_bbox=analysis.dominant_component_bbox,
        dominant_component_ink=analysis.dominant_component_ink,
        dominant_component_score=round(analysis.dominant_component_score, 6),
        refined_component_count=len(analysis.refined_components),
        core_component_indices=list(analysis.core_component_indices),
        discarded_from_selected_group_indices=list(
            analysis.discarded_from_selected_group_indices
        ),
        refined_signature_bbox=analysis.refined_signature_bbox,
        refinement_reason=analysis.refinement_reason,
        directional_accepted_indices=list(analysis.directional_accepted_indices),
        directional_accept_details=list(analysis.directional_accept_details),
        completeness_score=round(analysis.completeness_score, 4),
        completeness_refined_ink=analysis.completeness_refined_ink,
        completeness_reference_ink=analysis.completeness_reference_ink,
        quality_base=round(analysis.quality_base, 4),
        quality_completeness_factor=round(analysis.quality_completeness_factor, 4),
        message=message,
        original_with_roi_png_bytes=original_png,
        roi_image_png_bytes=roi_png,
        mask_image_png_bytes=mask_png,
        components_all_png_bytes=components_all_png,
        components_rejected_png_bytes=components_rejected_png,
        components_retained_png_bytes=components_retained_png,
        group_image_png_bytes=group_image_png,
        groups_image_png_bytes=groups_image_png,
        refined_group_png_bytes=refined_group_png,
        fallback_candidates_png_bytes=fallback_candidates_png,
        signature_png_bytes=signature_png,
        image_format="png",
    )
