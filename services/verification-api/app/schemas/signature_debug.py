from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.signature_extraction import BoundingBox


class DebugGroupInfo(BaseModel):
    index: int
    component_count: int
    bbox: BoundingBox
    ink_area: int
    score: float
    selected: bool


class FallbackRejectedCandidate(BaseModel):
    """A full-document fallback group rejected as an obvious document graphic."""

    index: int
    reason: str = ""
    bbox: BoundingBox


class FallbackCandidate(BaseModel):
    """One row of the fallback explainability table (every pre-filter group).

    ``score`` is the ACTUAL signature-likeness value used by the ranking.
    ``selected`` marks the top-ranked group (False for every row when the
    final credibility check rejected the winner or nothing was selected).
    """

    index: int
    bbox: BoundingBox
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    aspect: float = 0.0
    true_density: float = Field(ge=0.0, le=1.0)
    component_count: int = 0
    ink_relative: float = 0.0
    vertical_extent: float = Field(ge=0.0, le=2.0)
    score: float = 0.0
    status: str = ""
    selected: bool = False


class RecoveredStroke(BaseModel):
    """One detached stroke re-attached by local stroke recovery."""

    direction: str = ""
    bbox: BoundingBox
    h_gap: int = 0
    v_gap: int = 0
    reason: str = ""


class CropCompleteness(BaseModel):
    """Crop-completeness evidence for the refined signature candidate."""

    initial_bbox: Optional[BoundingBox] = None
    final_bbox: Optional[BoundingBox] = None
    initial_component_count: int = 0
    recovered_component_count: int = 0
    recovered_left: int = 0
    recovered_right: int = 0
    recovered_other: int = 0
    iterations: int = 0
    expansion_ratio: float = 1.0
    status: str = ""
    recovered_strokes: list[RecoveredStroke] = Field(default_factory=list)


class DirectionalAcceptDetail(BaseModel):
    """V2.4 diagnostic detail for one directional continuation acceptance."""

    component_index: int
    anchor_index: int
    h_gap: int
    v_gap: int
    ink_distance: int
    h_gap_limit: int
    ink_distance_limit: int
    height: int


class SignatureDebugResponse(BaseModel):
    success: bool
    # Development-only marker: which pipeline version produced this response.
    extraction_pipeline_version: str = ""
    original_width: int
    original_height: int
    candidate_roi: BoundingBox
    # Region actually analyzed (the full ROI in V2.1/V2.2).
    analysis_zone: Optional[BoundingBox] = None
    # MICR risk zone (ROI-local coordinates). Flagged, NOT physically removed.
    micr_band: Optional[BoundingBox] = None
    signature_bbox: Optional[BoundingBox] = None
    # Every connected component found in the ROI.
    total_component_count: int = 0
    # Components kept after geometric filtering.
    retained_component_count: int = 0
    # Components dropped (all reasons).
    rejected_component_count: int = 0
    # Components dropped specifically as MICR / printed text.
    micr_rejected_count: int = 0
    # Number of spatial groups built from the kept components.
    group_count: int = 0
    # Per-group diagnostic info (bbox, ink, signature-likeness score).
    groups: list[DebugGroupInfo] = Field(default_factory=list)
    # Index of the group selected as the signature candidate (-1 if none).
    selected_group_index: int = -1
    # Human-readable reason for the group selection.
    selection_reason: str = ""
    # Count of rejected components per reason (French labels).
    rejected_component_reasons: dict = Field(default_factory=dict)
    # Kept for backward compatibility with V2 (== retained_component_count).
    component_count: int = 0
    # --- V2.3 dominant-core refinement ---
    # Bbox of the dominant signature core (ROI-local).
    dominant_component_bbox: Optional[BoundingBox] = None
    # True ink pixels of the dominant core.
    dominant_component_ink: int = 0
    # Dominant-core score (ink x vertical extent x density, relative to ROI).
    dominant_component_score: float = 0.0
    # Number of components kept by the V2.3 core-connected refinement.
    refined_component_count: int = 0
    # Indices (into the selected V2.2 group) kept by the refinement.
    core_component_indices: list[int] = Field(default_factory=list)
    # Indices (into the selected V2.2 group) discarded by the refinement.
    discarded_from_selected_group_indices: list[int] = Field(default_factory=list)
    # Bbox of the refined signature structure (== signature_bbox in V2.3).
    refined_signature_bbox: Optional[BoundingBox] = None
    # Human-readable reason for the V2.3 refinement.
    refinement_reason: str = ""
    # --- V2.4 directional continuation + completeness quality ---
    # Indices (into the selected group) accepted by the V2.4 directional pass.
    directional_accepted_indices: list[int] = Field(default_factory=list)
    # Per-acceptance diagnostic details (anchor, gaps, ink distance, limits).
    directional_accept_details: list[DirectionalAcceptDetail] = Field(default_factory=list)
    # Completeness = refined ink / signature-like reference ink in [0, 1].
    completeness_score: float = Field(ge=0.0, le=1.0)
    # True ink pixels of the refined structure.
    completeness_refined_ink: int = 0
    # Refined ink + signature-like ink strictly right of the refined bbox.
    completeness_reference_ink: int = 0
    # V2.3 base quality before the completeness penalty.
    quality_base: float = Field(ge=0.0, le=1.0)
    # Multiplicative completeness factor actually applied to the base quality.
    quality_completeness_factor: float = Field(ge=0.0, le=1.0)
    extraction_quality: float = Field(ge=0.0, le=1.0)
    message: Optional[str] = None
    # --- Hybrid localization (ROI first, full-document fallback) ---
    # Machine mode ("roi" / "global_fallback") + French label for the Bilan.
    localization_mode: str = "roi"
    localization_label: str = "ROI principale"
    # Why the ROI candidate was rejected (empty when the ROI won).
    roi_failure_reason: str = ""
    # Full-document fallback evidence (0 / -1 / 0.0 / "" when ROI won).
    fallback_candidate_count: int = 0
    fallback_selected_index: int = -1
    fallback_selected_score: float = 0.0
    fallback_selection_reason: str = ""
    final_crop_width: int = 0
    final_crop_height: int = 0
    # Whole fallback groups rejected as obvious graphics, with truthful
    # reasons (empty when the ROI won or nothing was graphically rejected).
    fallback_rejected_candidates: list[FallbackRejectedCandidate] = Field(
        default_factory=list
    )
    # Explainability table: one row per pre-filter fallback group, in index
    # order, with the ACTUAL ranking scores (empty when fallback never ran).
    fallback_candidates: list[FallbackCandidate] = Field(default_factory=list)
    # The attempted primary ROI (always set, even when the fallback won and
    # candidate_roi describes the full-document scope instead).
    primary_roi: Optional[BoundingBox] = None
    # True when the ROI stage produced any candidate bbox (even rejected).
    roi_candidate_found: bool = False
    # True as soon as the full-document fallback executed.
    fallback_executed: bool = False
    # Crop-completeness evidence for the final candidate (None when no
    # refined bbox exists).
    crop_completeness: Optional[CropCompleteness] = None
    image_format: str
    original_with_roi_base64: str
    roi_image_base64: str
    # Binary ink mask of the full ROI (255 = ink).
    mask_image_base64: Optional[str] = None
    # ROI crop with every detected component drawn (distinct colors).
    components_all_base64: Optional[str] = None
    # ROI crop with the rejected components drawn (gray; MICR ones in orange).
    components_rejected_base64: Optional[str] = None
    # ROI crop with the retained components drawn (kept name for compat).
    components_image_base64: Optional[str] = None
    # ROI crop with the final signature group drawn (distinct colors) + bbox.
    group_image_base64: Optional[str] = None
    # ROI crop with all generated groups (each in a distinct color) + bbox.
    groups_image_base64: Optional[str] = None
    # ROI crop with the V2.3 refined structure (core yellow, refined kept
    # components, discarded-from-selected-group in gray) + refined bbox red.
    refined_group_image_base64: Optional[str] = None
    # Full-document fallback candidates (each group a distinct color, selected
    # region red). Only present when the fallback ran.
    fallback_candidates_image_base64: Optional[str] = None
    signature_image_base64: Optional[str] = None
