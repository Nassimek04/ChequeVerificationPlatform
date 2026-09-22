"""Experimental signature extraction service (V2.4 + hybrid localization).

Strategy overview:
1. decode the image (already done upstream) ;
2. compute the candidate ROI from configured relative ratios ;
3. crop the ROI ;
4. convert to grayscale and apply light preprocessing (noise reduction) ;
5. HYBRID LOCALIZATION (conservative extension, V2.4 pipeline unchanged):
   - PRIMARY: run the full V2.4 pipeline inside the expected cheque ROI,
     then apply the ROI-only signature plausibility gate (a high-quality crop
     can still be a printed-text fragment: merged text slab or tiny solid
     speck). A rejected ROI candidate triggers the fallback ;
   - FALLBACK (only when the ROI yields no credible candidate): rerun the
     SAME V2.4 pipeline (``_analyze_ink`` + dominant-core refinement +
     directional continuation + completeness quality) on the FULL document,
     reject whole groups that are structurally impossible as handwriting
     (full-bleed bars, near-solid blocks, long thin rules), rank the
     surviving candidate groups with the existing signature-likeness score,
     keep the strongest credible candidate, allow the guarded leftward
     mirror of the directional pass (same limits, mirrored) so a detached
     left stroke of the selected signature survives in the final crop, and
     apply a final credibility check to the top-ranked winner (ranking
     first never implies credibility) ;
   - V5-A is NEVER used for localization: it remains comparison-only after a
     successful extraction. When both stages fail, extraction raises
     ``SignatureExtractionError`` (no forced crop, no fabricated quality).
5. binarize with Otsu + ``THRESH_BINARY_INV`` so that ink (dark) becomes the
   foreground (255) ;
6. analyze the FULL ROI (no rigid bottom crop). The bottom band of the ROI is
   flagged as a "MICR risk zone" (configurable ratio) but is NOT physically
   removed: signature strokes that descend into it remain available ;
7. apply a small, configurable morphology close to reconnect broken strokes of
   the same trace (deliberately NOT inflated) ;
8. run connected-components analysis over the whole ROI ;
9. filter components with *relative* geometric criteria (area / width / height) ;
10. reject only components that look like MICR / printed text: located in the
    risk zone AND (very horizontal OR short-and-wide). A long diagonal
    signature stroke is never rejected for being tall or slanted ;
11. group the kept components by spatial proximity (nearest component, so
    distinct clusters do not chain into a single huge group) ;
12. select the signature group with a *signature-likeness* score based on
    relative ink mass x vertical extent x ink density — NOT the number of
    components (printed text can contain many small characters) ;
13. [V2.3] inside the selected group, identify the *dominant signature core*:
    the component with the highest dominant-core score (ink relative to ROI x
    vertical extent relative to ROI x ink density) ;
14. [V2.3] *core-connected refinement*: starting from that dominant core,
    re-expand the signature structure ONLY towards components whose actual ink
    is genuinely close in 2D to an already-accepted component (Chebyshev ink
    distance within a relative threshold AND bounding-box same-pair rule, see
    ``_is_component_2d_near``). Printed text / "(28)"-like components / bottom
    row MICR content that touch the selected group by bounding box but whose
    ink is separated from the signature are discarded. Legitimate disconnected
    strokes (main loop, nearby strokes, long descending diagonals) are kept ;
15. [V2.4] *directional continuation*: a second guarded pass AFTER the V2.3
    refinement rescues a genuine rightward signature continuation (a large
    loop + long rightward stroke, CHQ-0003) that V2.3 rejected only because
    its ink was farther than the conservative core ink-distance threshold.
    The candidate must be STRICTLY RIGHT of an already-accepted component,
    vertically aligned with it (v_gap == 0), tall enough, and its ink must
    come within a more generous, directional-only distance of the ink of that
    SAME component (single-pair rule, so no chaining through intermediates) ;
16. [V2.4] *completeness-aware quality*: the extraction quality is penalized
    when the refined structure misses signature-like ink strictly to its
    right; a complete signature keeps exactly the V2.3 quality ;
17. build the final bounding box as the union of the REFINED components ;
17b. [crop completeness] local stroke recovery: inspect kept components
     around the accepted union for detached strokes of the same handwriting
     (bounded, deterministic, both paths ; untouched result when nothing
     qualifies) and extend the final bbox accordingly ;
18. add a small margin and produce the final crop, encoded as PNG.

This is a technical heuristic. It does NOT recognize a signature, does NOT
compute an authenticity/conformity score, does NOT compare signatures and does
NOT use any OCR / AI model.
"""

import base64
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Development-only diagnostic marker. Exposed ONLY in the dev debug response
# so the UI can confirm which pipeline version actually ran.
EXTRACTION_PIPELINE_VERSION = "2.4"

# Hybrid localization modes (machine values, stable for the API / Bilan).
LOCALIZATION_MODE_ROI = "roi"
LOCALIZATION_MODE_GLOBAL_FALLBACK = "global_fallback"

# Explainability-table statuses (single source; the API route reuses the
# selected marker instead of re-matching the French label).
FALLBACK_STATUS_SELECTED = "Sélectionné"
FALLBACK_STATUS_MICR = "Rejeté — MICR"
FALLBACK_STATUS_NON_RETAINED = "Non retenu — score inférieur"
FALLBACK_STATUS_GROUP_DROPPED = "Non retenu — groupe écarté"
FALLBACK_STATUS_FINAL_REJECT = "Écarté — contrôle final"

# Human-readable French labels for the Bilan diagnostic.
LOCALIZATION_LABELS = {
    LOCALIZATION_MODE_ROI: "ROI principale",
    LOCALIZATION_MODE_GLOBAL_FALLBACK: "Recherche globale de secours",
}

# Default credibility gate: an ROI candidate is kept as-is only when its V2.4
# quality reaches this threshold. Below it (or with no bbox at all) the
# full-document fallback is attempted. Kept low on purpose so every ROI
# extraction that succeeds today (qualities ~0.6+) keeps byte-identical
# behavior; only weak/junk ROI candidates trigger the fallback.
DEFAULT_MIN_CREDIBLE_QUALITY = 0.20

# Hybrid FALLBACK-ONLY structural impossibility guards (never applied to the
# primary ROI path). A whole candidate group is rejected as an obvious
# document graphic — not handwriting — when its UNION bounding box (geometry
# relative to the full document + TRUE ink pixel density) matches one of
# these generic, coordinate-free patterns:
# - full-bleed bar: spans >= 90% of the document width or height (header /
#   footer banner, page border). A genuine signature is always localized.
# - near-solid block: TRUE ink density >= 0.85 of its bbox (solid banner,
#   filled graphic block, solid rule). Handwriting always leaves most of its
#   bbox empty, even with a thick pen ; the threshold sits clearly above a
#   fully filled ellipse (pi/4 ~= 0.785) so only truly solid graphics trip it.
# - long thin rule: aspect >= 15 with thickness <= 3% of the document
#   (horizontal / vertical rules, border lines). A straight rule is not
#   writing; a signature group containing an underline also spans taller ink.
FALLBACK_MAX_WIDTH_RATIO = 0.90
FALLBACK_MAX_HEIGHT_RATIO = 0.90
FALLBACK_MAX_SOLID_DENSITY = 0.85
FALLBACK_RULE_MIN_ASPECT = 15.0
FALLBACK_RULE_MAX_THIN_RATIO = 0.03

# Fallback WINNER final credibility gate (never applied to the ROI path).
# Ranking first is not enough: the top-ranked surviving group must still be
# plausibly handwritten. Full-document calibrated (a full-page signature is
# small relative to the page: Nexora 209x111 -> aspect 1.88, h_ratio 0.074 ;
# synthetic blobs -> aspect ~1.7, h_ratio ~0.14) :
# - FW_SLAB: at most 2 refined components AND aspect >= 6.0 AND refined
#   height < 6% of the document (merged printed-text slab).
# - FW_ROW: at least 4 refined components of UNIFORM height (max/min <= 1.6),
#   vertically aligned (center spread <= 0.30 of bbox height), jointly wide
#   (aspect >= 4.0) and all short (< 12% of the document) : a row of
#   glyphs, not varied handwriting strokes.
# - FW_LINE: extreme aspect (>= 10 or <= 0.1) whatever the rest : a rule or a
#   straight fragment, never a complete signature.
FW_SLAB_MIN_ASPECT = 6.0
FW_SLAB_MAX_HEIGHT_RATIO = 0.06
FW_SLAB_MAX_COMPONENTS = 2
FW_ROW_MIN_COMPONENTS = 4
FW_ROW_MAX_HEIGHT_UNIFORMITY = 1.6
FW_ROW_MAX_VSPREAD = 0.30
FW_ROW_MIN_ASPECT = 4.0
FW_ROW_MAX_HEIGHT_RATIO = 0.12
FW_LINE_MIN_ASPECT = 10.0

# ROI-ONLY signature plausibility gate (never applied to the fallback path).
# The V2.4 extraction quality measures crop characteristics, not whether the
# object looks like handwriting: a small printed-text fragment (e.g. a table
# header merged into one shallow slab, ~116x19 px at ~80% quality) can score
# ~0.80 and wrongly block the fallback. The gate below rejects a refined ROI
# candidate ONLY on combined text-like evidence. Calibrated on 27 genuine ROI
# signatures (single-component aspect <= 3.9, any aspect <= 5.48 with 4-5
# components, h_ratio >= 0.096, true density <= 0.432):
# - P1 merged text slab: at most 2 refined components AND aspect >= 6.0 AND
#   bbox height < 10% of the ROI height (true density is deliberately NOT
#   required: merged print spans 0.3-1.0 depending on glyph packing).
# - P1b dense shallow slab: at most 2 refined components AND aspect >= 4.5
#   AND bbox height < 8% of the ROI height AND true ink density >= 0.85.
# - P0 tiny solid speck: exactly 1 refined component AND bbox width < 4% AND
#   bbox height < 9% of the ROI (a lone printed glyph) AND density >= 0.85.
# - P2 lone solid rule: exactly 1 refined component AND extreme aspect
#   (>= 10 or <= 0.1, horizontal or vertical rule crossing the ROI) AND
#   true density >= 0.70. A single straight rule is never a complete
#   signature (27 genuine samples stay within aspect [0.38, 5.48]).
# Genuine signatures — even compact/shallow ones — always fail at least one
# term of every conjunction (real wide signatures have 4-5 components, real
# shallow ones are far less dense, real small ones are far less solid, real
# single strokes are far less thin).
ROI_PLAUS_SLAB_MIN_ASPECT = 6.0
ROI_PLAUS_SLAB_MAX_HEIGHT_RATIO = 0.10
ROI_PLAUS_SLAB_MAX_COMPONENTS = 2
ROI_PLAUS_DENSE_MIN_ASPECT = 4.5
ROI_PLAUS_DENSE_MAX_HEIGHT_RATIO = 0.08
ROI_PLAUS_DENSE_MIN_DENSITY = 0.85
ROI_PLAUS_SPECK_MAX_WIDTH_RATIO = 0.04
ROI_PLAUS_SPECK_MAX_HEIGHT_RATIO = 0.09
ROI_PLAUS_SPECK_MIN_DENSITY = 0.85
ROI_PLAUS_RULE_MIN_ASPECT = 10.0
ROI_PLAUS_RULE_MIN_DENSITY = 0.70

# Crop-completeness local stroke recovery (both paths, post-refinement).
# After a candidate is accepted, inspect a bounded neighborhood of the
# refined signature union for nearby kept ink components plausibly belonging
# to the same handwriting. All limits are relative to the INITIAL refined
# union (U0) so they adapt to signature size; no fixed pixel padding.
# - side attach (left/right): vertical overlap (v_gap == 0), horizontal gap
#   <= 20% of U0 width, candidate height in [35%, 150%] of U0 height and
#   width <= U0 width (tall detached flourish; body-text glyphs are far
#   shorter than a third of a signature envelope).
# - above/below marks: >= 50% horizontal overlap of the candidate width,
#   vertical gap <= 15% of U0 height, candidate at most half the union size
#   and compact (aspect in [0.5, 2.0] : i dots, not text slabs or stems).
# - ink gate: Chebyshev ink distance <= 20% of max(U0 width, U0 height).
# - exclusions: MICR-like components and graphic-guard hits (bars, solid
#   blocks, long rules) are never attached — existing logic reused.
# - bounds: at most 4 expansion passes, final union area <= 2.5x initial.
#   Deterministic: fixed candidate order, first-come union updates.
REC_MAX_PASSES = 4
REC_MAX_H_GAP_RATIO = 0.20
REC_MIN_HEIGHT_RATIO = 0.35
REC_MAX_HEIGHT_RATIO = 1.5
REC_MAX_WIDTH_RATIO = 1.0
REC_MAX_INK_RATIO = 0.20
REC_ABOVE_BELOW_MAX_V_GAP_RATIO = 0.15
REC_ABOVE_BELOW_MAX_SIZE_RATIO = 0.5
REC_ABOVE_BELOW_MIN_OVERLAP_RATIO = 0.5
REC_ABOVE_BELOW_MIN_ASPECT = 0.5
REC_ABOVE_BELOW_MAX_ASPECT = 2.0
REC_MAX_AREA_GROWTH = 2.5


class SignatureExtractionError(Exception):
    """Raised when signature extraction cannot be performed."""


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RecoveredStroke:
    """One detached stroke re-attached by local stroke recovery.

    ``direction`` is "gauche", "droite", "haut" or "bas" (relative to the
    signature union at attach time). ``h_gap``/``v_gap`` are the measured
    bounding-box gaps in pixels, ``reason`` the concise acceptance motive.
    """

    direction: str
    bbox: Rect
    h_gap: int
    v_gap: int
    reason: str


@dataclass
class CropCompleteness:
    """Crop-completeness evidence for the refined signature candidate."""

    initial_bbox: Rect | None
    final_bbox: Rect | None
    initial_component_count: int
    recovered_component_count: int
    recovered_left: int
    recovered_right: int
    recovered_other: int
    iterations: int
    expansion_ratio: float
    status: str
    recovered_strokes: list[RecoveredStroke] = field(default_factory=list)


@dataclass
class FallbackCandidateRow:
    """One full-document fallback candidate for the explainability table.

    Covers EVERY pre-filter group (rejected or not) in pre-filter index
    order. ``score`` is the ACTUAL ``_group_signature_score`` value used by
    the ranking (the selected row carries the exact selected score).
    ``status`` is one of the ``FALLBACK_STATUS_*`` labels ("Sélectionné",
    "Rejeté — MICR", "Rejeté — bloc graphique", "Rejeté — règle",
    "Non retenu — score inférieur", "Non retenu — groupe écarté" or
    "Écarté — contrôle final").
    """

    index: int
    bbox: Rect
    width: int
    height: int
    aspect: float
    true_density: float
    component_count: int
    ink_relative: float
    vertical_extent: float
    score: float
    status: str
    selected: bool = False


@dataclass
class InkAnalysis:
    """Result of the ink analysis stage (shared by extract and debug).

    V2.1 analyzes the FULL ROI. ``micr_band_top`` is the ROI-local y
    coordinate where the MICR risk zone begins: components located there are
    only rejected when they look like printed text / MICR, signature strokes
    that enter the band are preserved.

    V2.2 selects the signature group with a signature-likeness score
    (relative ink mass x vertical extent x ink density) instead of the number
    of components, so printed text (many small characters) no longer wins.

    V2.3 adds a second-stage *dominant-core refinement* on top of the selected
    V2.2 group: the dominant component of the group becomes the signature core
    and the refined structure is re-expanded only towards components whose
    actual ink is genuinely close in 2D (same-pair Chebyshev distance + bbox
    rule). ``bbox`` and ``quality`` are computed on the REFINED structure.

    V2.4 adds a directional continuation pass and a completeness-aware quality
    (see ``_refine_signature_group`` and ``_compute_extraction_quality_v24``).
    """

    roi: Rect
    micr_band_top: int
    binary_mask: np.ndarray          # ink mask of the full ROI (255 = ink)
    all_components: list[Rect]       # every connected component found
    kept_components: list[Rect]      # retained after geometric filtering
    rejected_components: list[Rect]  # dropped components (all reasons)
    micr_rejected_components: list[Rect]  # subset dropped as MICR / text
    groups: list[list[Rect]]         # spatial groups of kept components
    group_scores: list[float]        # signature-likeness score per group
    selected_group_index: int        # index of the selected group (-1 if none)
    selection_reason: str            # why this group was selected
    rejection_reasons: dict          # reason -> count for dropped components
    primary_group: list[Rect]        # signature group used for the bbox
    # --- V2.3 dominant-core refinement ---
    dominant_core_index_in_group: int   # index into primary_group (-1 if none)
    dominant_component_bbox: Rect | None   # bbox of the dominant core
    dominant_component_ink: int        # true ink pixels of the dominant core
    dominant_component_score: float    # dominant-core score of the core
    core_component_indices: list[int]  # indices into primary_group kept by V2.3
    discarded_from_selected_group_indices: list[int]  # indices dropped by V2.3
    refined_components: list[Rect]     # kept components after V2.3 refinement
    refined_signature_bbox: Rect | None  # union of refined components
    refinement_reason: str             # why components were kept/discarded
    # --- V2.4 directional continuation + completeness quality ---
    directional_accepted_indices: list[int]          # indices accepted by V2.4
    directional_accept_details: list["DirectionalAcceptDetail"]  # per-accept info
    completeness_score: float          # refined_ink / signature-like reference ink
    completeness_refined_ink: int      # true ink of the refined structure
    completeness_reference_ink: int    # refined ink + signature-like right content
    quality_base: float                # V2.3 base quality before the penalty
    quality_completeness_factor: float  # multiplicative completeness factor in [0,1]
    bbox: Rect | None                # union of the refined components (None if none)
    quality: float
    # --- Hybrid fallback-only graphic rejections (empty on the ROI path) ---
    # Parallel lists describing whole candidate groups rejected as obvious
    # document graphics before ranking (see ``_graphic_rejection_reason``).
    # Indices are pre-filter group indices (position in the grouping before
    # any rejection).
    graphic_rejected_group_indices: list[int] = field(default_factory=list)
    graphic_rejected_group_reasons: list[str] = field(default_factory=list)
    graphic_rejected_group_boxes: list[Rect] = field(default_factory=list)
    # --- Hybrid fallback candidate table (empty on the ROI path) ---
    # One row per pre-filter group with the ACTUAL ranking scores and a
    # truthful status (see ``FallbackCandidateRow``). Built only when the
    # fallback rejection flags are on.
    fallback_candidate_table: list[FallbackCandidateRow] = field(
        default_factory=list
    )
    # Final fallback credibility verdict on the top-ranked winner ("" when
    # the check did not run or the winner was accepted).
    winner_reject_reason: str = ""
    # --- Crop-completeness local stroke recovery (both paths) ---
    # Components re-attached around the refined union after acceptance.
    recovered_components: list[Rect] = field(default_factory=list)
    # Crop-completeness evidence (None when no refined bbox exists).
    crop_completeness: CropCompleteness | None = None


@dataclass
class SignatureExtractionResult:
    original_width: int
    original_height: int
    candidate_roi: Rect
    signature_bbox: Rect
    extraction_quality: float
    image_format: str
    signature_png_bytes: bytes
    # --- Hybrid localization (defaults preserve the historical ROI-only shape) ---
    # ``localization_mode`` is "roi" or "global_fallback" (see constants above).
    # ``signature_bbox`` is expressed in ``candidate_roi`` coordinates: ROI-local
    # in ROI mode, absolute (= full-document-local) in fallback mode where the
    # candidate ROI is the whole document.
    localization_mode: str = LOCALIZATION_MODE_ROI
    fallback_candidate_count: int = 0
    fallback_selected_index: int = -1
    fallback_selected_score: float = 0.0
    fallback_selection_reason: str = ""
    final_crop_width: int = 0
    final_crop_height: int = 0

    @property
    def signature_image_base64(self) -> str:
        return base64.b64encode(self.signature_png_bytes).decode("ascii")


@dataclass
class HybridLocalizationOutcome:
    """Two-stage localization evidence shared by extract and debug.

    ``roi_analysis`` is ALWAYS the V2.4 analysis of the expected cheque ROI.
    ``fallback_analysis`` is None when the ROI candidate was credible (no
    fallback attempted) ; otherwise it is the V2.4 analysis of the FULL
    document. ``final_analysis`` is the analysis the final crop was built
    from (None when both stages failed). ``final_is_full_image`` tells whether
    the final crop coordinates refer to the full image (fallback) or to the
    ROI crop (primary mode).
    """

    roi: Rect
    roi_analysis: InkAnalysis
    roi_credible: bool
    roi_failure_reason: str
    fallback_analysis: InkAnalysis | None
    localization_mode: str
    final_analysis: InkAnalysis | None
    final_is_full_image: bool
    # Why the top-ranked fallback winner was finally rejected ("" when the
    # fallback was not attempted, found nothing credible, or succeeded).
    fallback_reject_reason: str = ""


@dataclass(frozen=True)
class DirectionalAcceptDetail:
    """V2.4 diagnostic detail for one directional continuation acceptance.

    ``component_index`` is the index (into the selected group) of the accepted
    component, ``anchor_index`` the index of the already-accepted component it
    anchors to (the SAME component used for the horizontal gap AND the ink
    distance: single-pair rule), ``h_gap`` / ``v_gap`` the bounding-box gaps,
    ``ink_distance`` the Chebyshev distance between the two inks,
    ``h_gap_limit`` / ``ink_distance_limit`` the applicable thresholds and
    ``height`` the component height (compared to the V2.3 minimum).
    """

    component_index: int
    anchor_index: int
    h_gap: int
    v_gap: int
    ink_distance: int
    h_gap_limit: int
    ink_distance_limit: int
    height: int


@dataclass
class CoreRefinement:
    """V2.3/V2.4 second-stage refinement of the selected V2.2 group.

    ``core_index`` is the index (into the group) of the dominant signature
    core, ``refined_indices`` the indices of the components kept by the
    core-connected refinement (core first) and ``discarded_indices`` the
    indices dropped because they are printed text / "(28)" / bottom-row MICR
    content whose ink is not genuinely close in 2D.

    V2.4 fields: ``directional_accepted_indices`` are the group indices
    additionally accepted by the directional continuation pass, with one
    ``DirectionalAcceptDetail`` per acceptance in ``directional_accept_details``.
    ``completeness_score`` = ``completeness_refined_ink`` /
    ``completeness_reference_ink`` measures how much signature-like ink of the
    selected group lies strictly to the right of the refined structure.
    """

    core_index: int
    core_ink: int
    core_score: float
    refined_indices: list[int]
    discarded_indices: list[int]
    reason: str
    # --- V2.4 ---
    directional_accepted_indices: list[int]
    directional_accept_details: list[DirectionalAcceptDetail]
    completeness_score: float
    completeness_refined_ink: int
    completeness_reference_ink: int


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _compute_roi(
    width: int,
    height: int,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
) -> Rect:
    """Convert relative ratios into an absolute, clamped ROI rectangle."""
    x0 = int(round(x_start * width))
    y0 = int(round(y_start * height))
    x1 = int(round(x_end * width))
    y1 = int(round(y_end * height))

    x0 = _clip(x0, 0, width - 1)
    y0 = _clip(y0, 0, height - 1)
    x1 = _clip(x1, x0 + 1, width)
    y1 = _clip(y1, y0 + 1, height)

    return Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _compute_extraction_quality(
    binary_roi: np.ndarray,
    bbox: Rect | None,
    roi_area: int,
    n_components: int,
) -> float:
    """Compute a purely technical ink-focus heuristic in [0, 1].

    Since V2.3 this shared function is called with the REFINED bounding box and
    the REFINED component set (see ``_compute_extraction_quality_v23``), i.e.
    the printed text / "(28)" / bottom-row MICR content discarded by the
    dominant-core refinement no longer inflates the metric.

    Definition (documented, NOT a probability / authenticity / conformity /
    comparison score):

        quality = 0.30 * component_score
                + 0.40 * density_score
                + 0.30 * focus_score

    - ``component_score = min(1, n_components / 3)`` rewards a *coherent
      group* of several strokes (a real signature is rarely a single blob).
    - ``density_score = min(1, ink_density * 8)`` rewards a bbox reasonably
      filled with ink (density >= ~0.125) and never rewards an empty bbox.
    - ``focus_score = max(0, 1 - bbox_coverage)`` penalizes a bbox that spans
      almost the whole ROI (bad isolation) WITHOUT zeroing the whole metric:
      a legitimate, wide signature still receives a useful non-zero value.

    The metric never returns a high value for an empty result (bbox None or no
    component => 0.0).
    """
    if bbox is None or roi_area <= 0 or n_components <= 0:
        return 0.0

    bbox_area = max(1, bbox.width * bbox.height)
    bbox_area = min(bbox_area, roi_area)

    region = binary_roi[
        bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width
    ]
    ink_pixels = int(np.count_nonzero(region))
    ink_density = ink_pixels / bbox_area

    component_score = min(1.0, n_components / 3.0)
    density_score = min(1.0, ink_density * 8.0)
    coverage = bbox_area / roi_area
    focus_score = max(0.0, 1.0 - coverage)

    quality = 0.30 * component_score + 0.40 * density_score + 0.30 * focus_score
    return round(min(1.0, quality), 4)


def _compute_extraction_quality_v23(
    binary_roi: np.ndarray,
    bbox: Rect | None,
    roi_area: int,
    n_components: int,
) -> float:
    """V2.3 extraction quality, recomputed on the REFINED signature structure.

    Exact V2.3 formula (same documented sub-scores as V2.2 but applied to the
    refined bbox and the refined component set):

        quality = 0.30 * component_score + 0.40 * density_score + 0.30 * focus_score
        component_score = min(1, n_refined_components / 3)
        density_score   = min(1, ink_density_in_refined_bbox * 8)
        focus_score     = max(0, 1 - refined_bbox_area / roi_area)

    Technical heuristic only: it is NOT an authenticity, similarity,
    conformity or fraud-probability score.
    """
    return _compute_extraction_quality(binary_roi, bbox, roi_area, n_components)


def _compute_extraction_quality_v24(
    binary_roi: np.ndarray,
    bbox: Rect | None,
    roi_area: int,
    n_components: int,
    completeness_score: float,
    completeness_min_factor: float,
) -> tuple[float, float, float]:
    """V2.4 extraction quality = V2.3 base quality x completeness factor.

    Exact V2.4 formula:

        quality = base * (min_factor + (1 - min_factor) * completeness)
        base   = V2.3 quality (see ``_compute_extraction_quality_v23``)
        factor = min_factor + (1 - min_factor) * completeness

    - ``completeness_score`` in [0, 1]: refined_ink / signature-like reference
      ink (see ``_refine_signature_group``). A complete signature scores 1.0
      and keeps exactly the V2.3 base quality; a signature truncated on its
      right (signature-like ink strictly to the right of the refined bbox was
      missed) is penalized multiplicatively.
    - ``completeness_min_factor`` in [0, 1] bounds the penalty: with the
      default 0.5 a fully-truncated signature still keeps 50% of its base
      quality, so the metric can never drop to an arbitrary 0.

    Returns ``(quality, quality_base, completeness_factor)`` for diagnostics.
    Technical heuristic only: it is NOT an authenticity, similarity,
    conformity or fraud-probability score.
    """
    base = _compute_extraction_quality(binary_roi, bbox, roi_area, n_components)
    completeness = min(1.0, max(0.0, completeness_score))
    factor = completeness_min_factor + (1.0 - completeness_min_factor) * completeness
    quality = round(min(1.0, base * factor), 4)
    return quality, base, round(factor, 4)


def _is_micr_like(
    rect: Rect,
    roi: Rect,
    micr_band_top: int,
    micr_max_height_ratio: float,
    micr_min_aspect_ratio: float,
    micr_min_width_ratio: float,
) -> bool:
    """Return True when ``rect`` looks like MICR / printed text.

    Only components located (by vertical center) inside the MICR risk zone are
    candidates for rejection. A component is considered parasitic when it is:
    - very horizontal: ``width / height > micr_min_aspect_ratio`` (a long
      rule, a merged line) ; or
    - short and wide: ``height_ratio < micr_max_height_ratio`` AND
      ``width_ratio >= micr_min_width_ratio`` (a printed character row).

    A tall or slanted component (diagonal signature stroke) is never rejected
    merely for touching the band.
    """
    center_y = rect.y + rect.height / 2
    if center_y < micr_band_top:
        return False

    height_ratio = rect.height / max(1, roi.height)
    width_ratio = rect.width / max(1, roi.width)
    aspect = rect.width / max(1, rect.height)

    if aspect > micr_min_aspect_ratio:
        return True
    if height_ratio < micr_max_height_ratio and width_ratio >= micr_min_width_ratio:
        return True
    return False


def _axis_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Signed distance between two 1D intervals (0 when they overlap)."""
    if b_start > a_end:
        return b_start - a_end
    if a_start > b_end:
        return a_start - b_end
    return 0


def _group_components(
    components: list[Rect], roi: Rect, merge_x: int, merge_y: int
) -> list[list[Rect]]:
    """Group components by spatial proximity.

    A component joins a group when both the horizontal and the vertical gap
    between its bbox and the **nearest** component already in the group are
    within the relative merge distances. Using the nearest component (instead
    of the group's union bounding box) prevents "skip merges" where a
    component chains onto the far edge of a large group and merges distinct
    clusters together. Groups collect strokes that belong to the same
    signature even when they are not connected by ink.
    """
    groups: list[list[Rect]] = []
    for comp in components:
        best_index: int | None = None
        best_gap = None
        for gi, group in enumerate(groups):
            h_gap = min(
                _axis_gap(comp.x, comp.x + comp.width, c.x, c.x + c.width)
                for c in group
            )
            v_gap = min(
                _axis_gap(comp.y, comp.y + comp.height, c.y, c.y + c.height)
                for c in group
            )
            if h_gap <= merge_x and v_gap <= merge_y:
                combined = h_gap + v_gap
                if best_gap is None or combined < best_gap:
                    best_index = gi
                    best_gap = combined
        if best_index is not None:
            groups[best_index].append(comp)
        else:
            groups.append([comp])
    return groups


def _group_signature_score(group: list[Rect], roi: Rect) -> float:
    """Signature-likeness of a group, purely relative to the ROI.

    score = ink_frac * height_frac * density

    - ``ink_frac = ink_area / roi_area`` : the group concentrates ink (a real
      signature carries much more ink than a row of printed characters) ;
    - ``height_frac = bbox_height / roi_height`` : the group spans a
      significant vertical range (handwriting strokes, long descending
      diagonals) ; printed text rows are short ;
    - ``density = ink_area / bbox_area`` : the ink is concentrated inside the
      group bbox (compact handwriting) rather than spread over a wide region.

    Deliberately NOT based on the number of components: printed text can
    contain many small characters.
    """
    ink_area = sum(c.width * c.height for c in group)
    x0 = min(c.x for c in group)
    x1 = max(c.x + c.width for c in group)
    y0 = min(c.y for c in group)
    y1 = max(c.y + c.height for c in group)

    gbox_area = max(1, (x1 - x0) * (y1 - y0))
    ink_frac = ink_area / max(1, roi.width * roi.height)
    height_frac = (y1 - y0) / max(1, roi.height)
    density = ink_area / gbox_area
    return ink_frac * height_frac * density


def _group_union_bbox(group: list[Rect]) -> Rect:
    """Union bounding box of a component group."""
    return Rect(
        x=min(c.x for c in group),
        y=min(c.y for c in group),
        width=max(c.x + c.width for c in group) - min(c.x for c in group),
        height=max(c.y + c.height for c in group) - min(c.y for c in group),
    )


def _graphic_rejection_reason(
    group: list[Rect], zone: Rect, true_ink_pixels: int
) -> str | None:
    """Return a French rejection reason when a whole candidate group is
    structurally incompatible with a handwritten signature, else None.

    Hybrid FALLBACK-ONLY guard. All criteria are relative to the analyzed
    zone dimensions (never absolute pixels, never color, never fixed page
    positions) and use the TRUE ink pixel count of the group (from the
    connected-component label areas), not bbox-area estimates — so a solid
    banner (true density ~1.0) is caught while genuinely dense handwriting
    (true density well below 1.0) is never affected.
    """
    union = _group_union_bbox(group)
    union_area = max(1, union.width * union.height)
    true_density = true_ink_pixels / union_area
    width_ratio = union.width / max(1, zone.width)
    height_ratio = union.height / max(1, zone.height)
    aspect = union.width / max(1, union.height)

    if width_ratio >= FALLBACK_MAX_WIDTH_RATIO:
        return (
            "Rejeté : largeur excessive (%.2f de la largeur du document)."
            % width_ratio
        )
    if height_ratio >= FALLBACK_MAX_HEIGHT_RATIO:
        return (
            "Rejeté : hauteur excessive (%.2f de la hauteur du document)."
            % height_ratio
        )
    if true_density >= FALLBACK_MAX_SOLID_DENSITY:
        return (
            "Rejeté : bloc trop dense (densité d'encre réelle %.3f, "
            "forme rectangulaire pleine)."
            % true_density
        )
    if (
        aspect >= FALLBACK_RULE_MIN_ASPECT
        and height_ratio <= FALLBACK_RULE_MAX_THIN_RATIO
    ):
        return (
            "Rejeté : règle horizontale (rapport %.1f, épaisseur relative %.3f)."
            % (aspect, height_ratio)
        )
    if (
        aspect > 0
        and (1.0 / aspect) >= FALLBACK_RULE_MIN_ASPECT
        and width_ratio <= FALLBACK_RULE_MAX_THIN_RATIO
    ):
        return (
            "Rejeté : règle verticale (rapport %.1f, épaisseur relative %.3f)."
            % (1.0 / aspect, width_ratio)
        )
    return None


def _select_primary_group(
    groups: list[list[Rect]], roi: Rect, micr_band_top: int
) -> tuple[list[Rect], int, list[float], str]:
    """Choose the signature group used to build the final bbox.

    Groups whose components are mostly inside the MICR risk zone are strongly
    penalized (they are likely MICR / printed content). Among the remaining
    groups the one with the highest signature-likeness score wins (see
    ``_group_signature_score``) — NOT the one with the most components.

    Returns ``(primary_group, selected_index, scores, reason)``.
    """
    scores = [_group_signature_score(g, roi) for g in groups]

    candidates: list[int] = []
    fallback: int | None = None
    for index, group in enumerate(groups):
        in_band = sum(
            1 for c in group if c.y + c.height / 2 >= micr_band_top
        )
        fraction_in_band = in_band / len(group)
        if fraction_in_band >= 0.8:
            if fallback is None or scores[index] > scores[fallback]:
                fallback = index
        else:
            candidates.append(index)

    if candidates:
        selected = max(candidates, key=lambda i: scores[i])
    elif fallback is not None:
        selected = fallback
    else:
        return [], -1, scores, "Aucun groupe éligible."

    primary = groups[selected]
    reason = (
        "Groupe %d sélectionné : score de ressemblance signature %.5f "
        "(encre relative %.3f, étendue verticale %.3f, densité %.3f)."
        % (
            selected,
            scores[selected],
            sum(c.width * c.height for c in primary) / max(1, roi.width * roi.height),
            (max(c.y + c.height for c in primary) - min(c.y for c in primary))
            / max(1, roi.height),
            sum(c.width * c.height for c in primary)
            / max(
                1,
                (
                    max(c.x + c.width for c in primary) - min(c.x for c in primary)
                )
                * (
                    max(c.y + c.height for c in primary) - min(c.y for c in primary)
                ),
            ),
        )
    )
    if not candidates:
        reason += " Aucun groupe hors bande MICR : repli sur le meilleur score."
    return primary, selected, scores, reason


def _dominant_component_score(rect: Rect, ink_area: int, roi: Rect) -> float:
    """Dominant-core score of a single component, purely relative to the ROI.

        score = ink_frac * height_frac * density

    - ``ink_frac = ink_area / roi_area`` : the component concentrates ink (the
      main handwritten loop carries much more ink than a printed character) ;
    - ``height_frac = rect.height / roi_height`` : the component spans a large
      vertical range (handwriting strokes, long descending diagonals) ;
    - ``density = ink_area / (rect.width * rect.height)`` : the ink is compact
      inside the component bbox.

    ``ink_area`` is the TRUE number of ink pixels of the component (from the
    connected-component label mask), not its bounding-box area, so a long thin
    diagonal stroke is not over-weighted. CHQ-0002 coordinates are NEVER
    hard-coded here.
    """
    roi_area = max(1, roi.width * roi.height)
    bbox_area = max(1, rect.width * rect.height)
    ink_frac = ink_area / roi_area
    height_frac = rect.height / max(1, roi.height)
    density = ink_area / bbox_area
    return ink_frac * height_frac * density


def _is_component_2d_near(
    candidate: Rect,
    accepted: Rect,
    roi: Rect,
    max_h_gap_ratio: float,
    max_v_gap_ratio: float,
) -> bool:
    """True when the candidate is genuinely close in 2D to ``accepted``.

    CRITICAL RULE: the horizontal gap AND the vertical gap are evaluated
    against the SAME ``accepted`` component. This deliberately does NOT
    reproduce the V2.2 grouping behaviour where the minimum horizontal gap
    could come from component A while the minimum vertical gap came from
    component B (which let printed clusters chain into the signature group).
    """
    h_gap = _axis_gap(
        candidate.x, candidate.x + candidate.width,
        accepted.x, accepted.x + accepted.width,
    )
    v_gap = _axis_gap(
        candidate.y, candidate.y + candidate.height,
        accepted.y, accepted.y + accepted.height,
    )
    max_h = max_h_gap_ratio * roi.width
    max_v = max_v_gap_ratio * roi.height
    return h_gap <= max_h and v_gap <= max_v


def _chebyshev_ink_distance(ink: np.ndarray) -> np.ndarray:
    """Chebyshev (L-infinity) distance from every pixel to the nearest ink pixel.

    ``ink`` is a 0/255 uint8 mask (255 = ink). The returned map contains, for
    each pixel, ``max(abs(dx), abs(dy))`` to the closest ink pixel: horizontal
    AND vertical separation are thus both within the value, against the same
    ink.
    """
    return cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_C, 3)


def _refine_signature_group(
    group: list[Rect],
    labels: np.ndarray,
    stats: np.ndarray,
    roi: Rect,
    ink_distance_ratio: float,
    max_h_gap_ratio: float,
    max_v_gap_ratio: float,
    min_height_ratio: float,
    directional_h_gap_ratio: float,
    directional_ink_distance_ratio: float,
    allow_leftward_continuation: bool = False,
) -> CoreRefinement:
    """V2.3/V2.4 second-stage refinement of the selected V2.2 group.

    V2.3 steps:
    1. the dominant core is the group component with the highest
       ``_dominant_component_score`` ;
    2. starting from that core, the refined structure re-expands
       CONSERVATIVELY: a candidate component joins only when
       - its true ink comes within ``ink_distance_ratio * roi.width``
         (Chebyshev) of the ink of an already-accepted component, AND
       - its bbox is within the same-pair gap thresholds of at least one
         accepted component (``_is_component_2d_near``), AND
       - its height is at least ``min_height_ratio * roi.height`` (printed
         text / specks are short).

    V2.4 adds a directional continuation pass AFTER the V2.3 loop:
    3. a candidate component that V2.3 rejected is accepted when it is
       - STRICTLY RIGHT of an already-accepted component
         (``candidate.x >= anchor.x + anchor.width``), AND
       - vertically aligned with that SAME component (v_gap == 0), AND
       - tall enough (same ``min_height`` as V2.3), AND
       - within ``directional_h_gap_ratio * roi.width`` of that component's
         bounding box, AND
       - its ink comes within ``directional_ink_distance_ratio * roi.width``
         (Chebyshev) of the ink of that SAME component ONLY (per-component
         distance transform, never the union). This single-pair rule keeps the
         anti-chaining property: a far component can never be pulled in
         through an intermediate one.

    V2.4 completeness metric (``completeness_score``): the "signature-like
    reference ink" is refined_ink + the true ink of every group component that
    is STRICTLY RIGHT of the refined bbox, vertically overlapping it, tall
    enough, and within ``directional_h_gap_ratio * roi.width`` of it. Such a
    component is exactly what the directional continuation could have kept, so
    the ratio refined_ink / reference_ink truthfully measures how much of the
    signature was captured. Printed text above / left / far away never
    qualifies and therefore never penalizes the metric.

    Fallback-only leftward continuation (``allow_leftward_continuation``):
    the V2.4 ROI contract NEVER accepts leftward components, so this guarded
    mirror runs only when explicitly enabled (full-document fallback). A
    genuine detached left stroke (leading flourish) of the selected signature
    would otherwise be cut from the final crop with no rescue path (the V2.4
    pass is strictly rightward). Same single-pair rule, mirrored: STRICTLY
    LEFT of an already-accepted component, vertically aligned with that SAME
    component (v_gap == 0), tall enough, bbox gap and ink distance within the
    SAME directional-only limits. Runs after (never interleaved with) the
    rightward pass; the completeness/quality computation below is untouched.
    """
    if not group:
        return CoreRefinement(
            core_index=-1,
            core_ink=0,
            core_score=0.0,
            refined_indices=[],
            discarded_indices=[],
            reason="Groupe principal vide : aucun raffinement possible.",
            directional_accepted_indices=[],
            directional_accept_details=[],
            completeness_score=1.0,
            completeness_refined_ink=0,
            completeness_reference_ink=0,
        )

    stats_by_rect: dict[tuple[int, int, int, int], int] = {}
    for lab in range(1, len(stats)):
        x, y, w, h, _ = stats[lab]
        stats_by_rect[(int(x), int(y), int(w), int(h))] = lab

    group_labels: list[int | None] = []
    for comp in group:
        group_labels.append(stats_by_rect.get((comp.x, comp.y, comp.width, comp.height)))

    ink_masks: list[np.ndarray | None] = []
    for lab in group_labels:
        if lab is None:
            ink_masks.append(None)
        else:
            ink_masks.append(labels == lab)

    def _ink_area(mask: np.ndarray | None) -> int:
        return int(np.count_nonzero(mask)) if mask is not None else 0

    scores = [
        _dominant_component_score(comp, _ink_area(mask), roi)
        for comp, mask in zip(group, ink_masks)
    ]
    core_index = int(np.argmax(scores))
    core_ink = _ink_area(ink_masks[core_index])
    core_score = scores[core_index]

    max_ink = ink_distance_ratio * roi.width
    min_height = min_height_ratio * roi.height
    directional_max_h = directional_h_gap_ratio * roi.width
    directional_max_ink = directional_ink_distance_ratio * roi.width

    accepted_set: set[int] = {core_index}
    accepted_ink = ink_masks[core_index].astype(np.uint8)
    ink_dist = _chebyshev_ink_distance(accepted_ink)

    changed = True
    while changed:
        changed = False
        for index in range(len(group)):
            if index in accepted_set:
                continue
            mask = ink_masks[index]
            if mask is None:
                continue
            candidate = group[index]
            if candidate.height < min_height:
                continue
            ys, xs = np.where(mask)
            if ink_dist[ys, xs].min() > max_ink:
                continue
            if not any(
                _is_component_2d_near(candidate, group[j], roi, max_h_gap_ratio, max_v_gap_ratio)
                for j in accepted_set
            ):
                continue
            accepted_set.add(index)
            accepted_ink = np.logical_or(accepted_ink, mask).astype(np.uint8)
            ink_dist = _chebyshev_ink_distance(accepted_ink)
            changed = True

    # --- V2.4 directional continuation (single-pair, strictly rightward) ---
    directional_indices: list[int] = []
    directional_details: list[DirectionalAcceptDetail] = []

    changed = True
    while changed:
        changed = False
        for index in range(len(group)):
            if index in accepted_set:
                continue
            mask = ink_masks[index]
            if mask is None:
                continue
            candidate = group[index]
            if candidate.height < min_height:
                continue
            ys, xs = np.where(mask)
            best: tuple[float, int, int, int] | None = None
            for anchor_index in sorted(accepted_set):
                anchor = group[anchor_index]
                if candidate.x < anchor.x + anchor.width:
                    continue
                v_gap = _axis_gap(
                    candidate.y, candidate.y + candidate.height,
                    anchor.y, anchor.y + anchor.height,
                )
                if v_gap != 0:
                    continue
                h_gap = candidate.x - (anchor.x + anchor.width)
                if h_gap > directional_max_h:
                    continue
                anchor_mask = ink_masks[anchor_index].astype(np.uint8)
                anchor_dist = _chebyshev_ink_distance(anchor_mask)
                ink_distance = float(anchor_dist[ys, xs].min())
                if ink_distance > directional_max_ink:
                    continue
                if best is None or ink_distance < best[0]:
                    best = (ink_distance, anchor_index, h_gap, v_gap)
            if best is not None:
                ink_distance, anchor_index, h_gap, v_gap = best
                accepted_set.add(index)
                directional_indices.append(index)
                directional_details.append(
                    DirectionalAcceptDetail(
                        component_index=index,
                        anchor_index=anchor_index,
                        h_gap=h_gap,
                        v_gap=v_gap,
                        ink_distance=round(ink_distance),
                        h_gap_limit=round(directional_max_h),
                        ink_distance_limit=round(directional_max_ink),
                        height=candidate.height,
                    )
                )
                changed = True

    # --- Fallback-only leftward continuation (exact mirror, see docstring) ---
    leftward_indices: list[int] = []
    if allow_leftward_continuation:
        changed = True
        while changed:
            changed = False
            for index in range(len(group)):
                if index in accepted_set:
                    continue
                mask = ink_masks[index]
                if mask is None:
                    continue
                candidate = group[index]
                if candidate.height < min_height:
                    continue
                ys, xs = np.where(mask)
                best_left: tuple[float, int, int, int] | None = None
                for anchor_index in sorted(accepted_set):
                    anchor = group[anchor_index]
                    if candidate.x + candidate.width > anchor.x:
                        continue
                    v_gap = _axis_gap(
                        candidate.y, candidate.y + candidate.height,
                        anchor.y, anchor.y + anchor.height,
                    )
                    if v_gap != 0:
                        continue
                    h_gap = anchor.x - (candidate.x + candidate.width)
                    if h_gap > directional_max_h:
                        continue
                    anchor_mask = ink_masks[anchor_index].astype(np.uint8)
                    anchor_dist = _chebyshev_ink_distance(anchor_mask)
                    ink_distance = float(anchor_dist[ys, xs].min())
                    if ink_distance > directional_max_ink:
                        continue
                    if best_left is None or ink_distance < best_left[0]:
                        best_left = (ink_distance, anchor_index, h_gap, v_gap)
                if best_left is not None:
                    ink_distance, anchor_index, h_gap, v_gap = best_left
                    accepted_set.add(index)
                    directional_indices.append(index)
                    leftward_indices.append(index)
                    directional_details.append(
                        DirectionalAcceptDetail(
                            component_index=index,
                            anchor_index=anchor_index,
                            h_gap=h_gap,
                            v_gap=v_gap,
                            ink_distance=round(ink_distance),
                            h_gap_limit=round(directional_max_h),
                            ink_distance_limit=round(directional_max_ink),
                            height=candidate.height,
                        )
                    )
                    changed = True

    directional_accepted = sorted(set(directional_indices))
    refined_indices = sorted(accepted_set)
    discarded_indices = [i for i in range(len(group)) if i not in accepted_set]
    core_rect = group[core_index]

    # --- V2.4 completeness metric ---
    refined_ink = sum(_ink_area(ink_masks[i]) for i in refined_indices)
    reference_ink = refined_ink
    if refined_indices:
        rb_x0 = min(group[i].x for i in refined_indices)
        rb_x1 = max(group[i].x + group[i].width for i in refined_indices)
        rb_y0 = min(group[i].y for i in refined_indices)
        rb_y1 = max(group[i].y + group[i].height for i in refined_indices)
        for index in range(len(group)):
            if index in accepted_set:
                continue
            comp = group[index]
            if comp.height < min_height:
                continue
            if comp.x < rb_x1:
                continue
            if _axis_gap(comp.y, comp.y + comp.height, rb_y0, rb_y1) != 0:
                continue
            if comp.x - rb_x1 > directional_max_h:
                continue
            reference_ink += _ink_area(ink_masks[index])
    completeness_score = (
        1.0 if reference_ink <= 0 else min(1.0, refined_ink / reference_ink)
    )

    reason = (
        "Raffinement V2.3/V2.4 : composante dominante (indice groupe %d, bbox "
        "(%d,%d,%d,%d), encre %d px, score %.5f). %d composante(s) retenue(s) "
        "dont %d par continuation directionnelle, %d rejetée(s) du groupe "
        "sélectionné (V2.3: distance encre <= %.1f px, hauteur >= %.1f px, "
        "même composante ; V2.4: strictement à droite, v_gap=0, distance encre "
        "<= %.1f px, gap H <= %.1f px). Complétude %.3f (encre raffinée %d / "
        "encre de référence %d)."
        % (
            core_index,
            core_rect.x,
            core_rect.y,
            core_rect.width,
            core_rect.height,
            core_ink,
            core_score,
            len(refined_indices),
            len(directional_accepted),
            len(discarded_indices),
            max_ink,
            min_height,
            directional_max_ink,
            directional_max_h,
            completeness_score,
            refined_ink,
            reference_ink,
        )
    )
    if leftward_indices:
        reason += (
            " Continuation gauche (secours) : %d composante(s) ajoutée(s) "
            "(strictement à gauche, v_gap=0, mêmes limites directionnelles)."
            % len(leftward_indices)
        )

    return CoreRefinement(
        core_index=core_index,
        core_ink=core_ink,
        core_score=core_score,
        refined_indices=refined_indices,
        discarded_indices=discarded_indices,
        reason=reason,
        directional_accepted_indices=directional_accepted,
        directional_accept_details=directional_details,
        completeness_score=completeness_score,
        completeness_refined_ink=refined_ink,
        completeness_reference_ink=reference_ink,
    )


def _analyze_ink(
    gray_roi: np.ndarray,
    roi: Rect,
    micr_zone_ratio: float,
    min_component_area_ratio: float,
    max_component_area_ratio: float,
    min_component_width_ratio: float,
    min_component_height_ratio: float,
    micr_max_height_ratio: float,
    micr_min_aspect_ratio: float,
    micr_min_width_ratio: float,
    component_merge_distance_ratio: float,
    group_min_components: int,
    morph_kernel_size: int,
    core_ink_distance_ratio: float,
    core_max_h_gap_ratio: float,
    core_max_v_gap_ratio: float,
    core_min_height_ratio: float,
    core_directional_h_gap_ratio: float,
    core_directional_ink_distance_ratio: float,
    completeness_min_factor: float,
    reject_micr_like_groups: bool = False,
    reject_graphic_groups: bool = False,
    allow_leftward_continuation: bool = False,
    final_winner_check: bool = False,
) -> InkAnalysis:
    """Binarize the ROI, filter components, group them, then refine the
    selected group (V2.3 dominant-core refinement + V2.4 directional
    continuation) and compute bbox + quality.

    V2.2 differences vs V2:
    - the whole ROI is analyzed (no rigid bottom crop) ;
    - only MICR / printed-looking components inside the risk zone are rejected ;
    - kept components are grouped by proximity to their nearest neighbor (no
      chaining of distinct clusters) ;
    - the signature group is selected with a signature-likeness score
      (ink x vertical extent x density), not by component count.

    V2.3 adds a second stage on top of the selected group: the dominant
    component becomes the signature core and the group is re-expanded only
    towards components whose actual ink is genuinely close in 2D (see
    ``_refine_signature_group``).

    V2.4 adds a directional continuation pass (rescues a rightward signature
    continuation rejected by V2.3) and applies the completeness-aware quality
    (see ``_compute_extraction_quality_v24``). ``bbox`` and ``quality`` are
    computed on the refined structure.

    ``reject_micr_like_groups`` (hybrid fallback only, default False) drops
    whole groups whose UNION bounding box is itself MICR / printed-text-like
    per the unchanged ``_is_micr_like`` predicate (e.g. a full MICR row whose
    isolated characters each pass the per-component filter but whose merged
    row is obviously printed content). Genuine handwriting (tall/slanted
    union, or outside the risk band) is never affected.

    ``reject_graphic_groups`` (hybrid fallback only, default False) drops
    whole groups that are structurally incompatible with handwriting
    (full-bleed bars, near-solid blocks, long thin rules — see
    ``_graphic_rejection_reason``). Rejected groups are recorded with their
    reason in ``graphic_rejected_group_*`` for diagnostics. When every group
    is rejected, no candidate is selected (safe failure, never "least-bad").

    ``allow_leftward_continuation`` (hybrid fallback only, default False)
    enables the guarded leftward mirror of the V2.4 directional pass inside
    ``_refine_signature_group`` so a detached left stroke of the selected
    signature survives in the final crop. The ROI path always keeps False
    (V2.4 contract: the directional pass never accepts left components).

    ``final_winner_check`` (hybrid fallback only, default False) applies the
    final credibility gate (``_fallback_winner_plausibility``) to the
    top-ranked surviving candidate and fills ``fallback_candidate_table``
    (one row per pre-filter group with the ACTUAL ranking scores) plus
    ``winner_reject_reason`` when the winner is finally rejected.

    All geometric criteria are relative to the ROI dimensions / area.
    """
    roi_width = roi.width
    roi_height = roi.height
    roi_area = roi_width * roi_height
    micr_band_top = int(round(roi_height * (1.0 - micr_zone_ratio)))

    # Binarization: Otsu + THRESH_BINARY_INV => ink (dark) is the foreground.
    _, binary = cv2.threshold(
        gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    zone_mask = binary.copy()

    # Small morphology close: reconnects broken strokes of the same trace.
    # Kept small on purpose: it must not merge signature + text + MICR into a
    # single giant component.
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    zone_mask = cv2.morphologyEx(zone_mask, cv2.MORPH_CLOSE, kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        zone_mask, connectivity=8
    )

    min_area = min_component_area_ratio * roi_area
    max_area = max_component_area_ratio * roi_area
    min_width = min_component_width_ratio * roi_width
    min_height = min_component_height_ratio * roi_height

    merge_x = int(round(component_merge_distance_ratio * roi_width))
    merge_y = int(round(component_merge_distance_ratio * roi_height))

    all_components: list[Rect] = []
    kept: list[Rect] = []
    rejected: list[Rect] = []
    micr_rejected: list[Rect] = []
    rejection_reasons: dict = {"area_min": 0, "area_max": 0, "thin": 0, "micr": 0}
    # TRUE ink pixel count per component bbox (label areas from
    # connectedComponents, accumulated for safety on duplicate bboxes).
    true_ink_by_rect: dict[tuple[int, int, int, int], int] = {}

    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        rect = Rect(x=int(x), y=int(y), width=int(w), height=int(h))
        all_components.append(rect)
        key = (int(x), int(y), int(w), int(h))
        true_ink_by_rect[key] = true_ink_by_rect.get(key, 0) + int(area)

        if area < min_area:
            rejected.append(rect)
            rejection_reasons["area_min"] += 1
            continue
        if area > max_area:
            rejected.append(rect)
            rejection_reasons["area_max"] += 1
            continue
        if w < min_width or h < min_height:
            rejected.append(rect)
            rejection_reasons["thin"] += 1
            continue
        if _is_micr_like(
            rect,
            roi,
            micr_band_top,
            micr_max_height_ratio,
            micr_min_aspect_ratio,
            micr_min_width_ratio,
        ):
            rejected.append(rect)
            micr_rejected.append(rect)
            rejection_reasons["micr"] += 1
            continue

        kept.append(rect)

    groups: list[list[Rect]] = []
    group_scores: list[float] = []
    selected_group_index = -1
    selection_reason = ""
    primary: list[Rect] = []
    dominant_core_index_in_group = -1
    dominant_component_bbox: Rect | None = None
    dominant_component_ink = 0
    dominant_component_score = 0.0
    core_component_indices: list[int] = []
    discarded_from_selected_group_indices: list[int] = []
    refined_components: list[Rect] = []
    refinement_reason = ""
    directional_accepted_indices: list[int] = []
    directional_accept_details: list[DirectionalAcceptDetail] = []
    completeness_score = 1.0
    completeness_refined_ink = 0
    completeness_reference_ink = 0
    quality_base = 0.0
    quality_completeness_factor = 1.0
    bbox: Rect | None = None
    refined_signature_bbox: Rect | None = None
    quality = 0.0
    recovered_components: list[Rect] = []
    crop_completeness: CropCompleteness | None = None

    graphic_rejected_group_indices: list[int] = []
    graphic_rejected_group_reasons: list[str] = []
    graphic_rejected_group_boxes: list[Rect] = []
    fallback_candidate_table: list[FallbackCandidateRow] = []
    winner_reject_reason = ""
    micr_rejected_group_indices: list[int] = []
    micr_rejected_group_boxes: list[Rect] = []
    # Pre-filter group index -> survivor position tracking. ``groups`` below
    # keeps the historical content/order (survivors only); ``survivor_orig``
    # maps each survivor position back to its pre-filter group index so the
    # explainability table and the selection stay aligned.
    survivor_orig: list[int] = []
    eligible_orig: list[int] = []
    selected_orig: int | None = None

    if kept:
        pre_groups = _group_components(kept, roi, merge_x, merge_y)
        survivor_orig = list(range(len(pre_groups)))
        if reject_micr_like_groups:
            remaining: list[int] = []
            for gi in survivor_orig:
                union = _group_union_bbox(pre_groups[gi])
                if _is_micr_like(
                    union,
                    roi,
                    micr_band_top,
                    micr_max_height_ratio,
                    micr_min_aspect_ratio,
                    micr_min_width_ratio,
                ):
                    micr_rejected_group_indices.append(gi)
                    micr_rejected_group_boxes.append(union)
                else:
                    remaining.append(gi)
            survivor_orig = remaining
        groups = [pre_groups[gi] for gi in survivor_orig]
        if reject_graphic_groups:
            remaining = []
            for gi in survivor_orig:
                group = pre_groups[gi]
                true_ink = sum(
                    true_ink_by_rect.get(
                        (c.x, c.y, c.width, c.height), c.width * c.height
                    )
                    for c in group
                )
                reason = _graphic_rejection_reason(group, roi, true_ink)
                if reason is None:
                    remaining.append(gi)
                else:
                    graphic_rejected_group_indices.append(gi)
                    graphic_rejected_group_reasons.append(reason)
                    graphic_rejected_group_boxes.append(
                        _group_union_bbox(group)
                    )
            survivor_orig = remaining
            groups = [pre_groups[gi] for gi in survivor_orig]
        eligible = [g for g in groups if len(g) >= group_min_components]
        eligible_orig = [
            survivor_orig[k]
            for k, g in enumerate(groups)
            if len(g) >= group_min_components
        ]
        if eligible:
            primary, selected_group_index, group_scores, selection_reason = (
                _select_primary_group(eligible, roi, micr_band_top)
            )
            selected_orig = eligible_orig[selected_group_index]

        # Explainability table: one row per pre-filter group, in index order,
        # with the ACTUAL ranking scores (the selected row carries the exact
        # selected score). Built only on the fallback path.
        if reject_micr_like_groups or reject_graphic_groups:
            micr_set = set(micr_rejected_group_indices)
            graphic_status = {
                gi: _graphic_short_status(reason)
                for gi, reason in zip(
                    graphic_rejected_group_indices,
                    graphic_rejected_group_reasons,
                )
            }
            eligible_set = set(eligible_orig)
            score_by_orig = {
                eligible_orig[k]: group_scores[k]
                for k in range(len(group_scores))
            }
            for gi, group in enumerate(pre_groups):
                union = _group_union_bbox(group)
                union_area = max(1, union.width * union.height)
                ink_area = sum(c.width * c.height for c in group)
                true_ink = sum(
                    true_ink_by_rect.get(
                        (c.x, c.y, c.width, c.height), c.width * c.height
                    )
                    for c in group
                )
                if gi in micr_set:
                    status = FALLBACK_STATUS_MICR
                    is_selected = False
                elif gi in graphic_status:
                    status = graphic_status[gi]
                    is_selected = False
                elif selected_orig is not None and gi == selected_orig:
                    status = FALLBACK_STATUS_SELECTED
                    is_selected = True
                elif gi in eligible_set:
                    status = FALLBACK_STATUS_NON_RETAINED
                    is_selected = False
                else:
                    status = FALLBACK_STATUS_GROUP_DROPPED
                    is_selected = False
                fallback_candidate_table.append(
                    FallbackCandidateRow(
                        index=gi,
                        bbox=union,
                        width=union.width,
                        height=union.height,
                        aspect=round(union.width / max(1, union.height), 3),
                        true_density=round(true_ink / union_area, 4),
                        component_count=len(group),
                        ink_relative=round(
                            ink_area / max(1, roi.width * roi.height), 6
                        ),
                        vertical_extent=round(
                            union.height / max(1, roi.height), 4
                        ),
                        score=round(
                            score_by_orig.get(
                                gi, _group_signature_score(group, roi)
                            ),
                            6,
                        ),
                        status=status,
                        selected=is_selected,
                    )
                )

    if primary:
        refinement = _refine_signature_group(
            primary,
            labels,
            stats,
            roi,
            ink_distance_ratio=core_ink_distance_ratio,
            max_h_gap_ratio=core_max_h_gap_ratio,
            max_v_gap_ratio=core_max_v_gap_ratio,
            min_height_ratio=core_min_height_ratio,
            directional_h_gap_ratio=core_directional_h_gap_ratio,
            directional_ink_distance_ratio=core_directional_ink_distance_ratio,
            allow_leftward_continuation=allow_leftward_continuation,
        )
        dominant_core_index_in_group = refinement.core_index
        refinement_reason = refinement.reason
        refined_indices = refinement.refined_indices
        discarded_from_selected_group_indices = list(refinement.discarded_indices)
        core_component_indices = list(refinement.refined_indices)
        refined_components = [primary[i] for i in refined_indices]
        directional_accepted_indices = list(refinement.directional_accepted_indices)
        directional_accept_details = list(refinement.directional_accept_details)
        completeness_score = refinement.completeness_score
        completeness_refined_ink = refinement.completeness_refined_ink
        completeness_reference_ink = refinement.completeness_reference_ink

        if dominant_core_index_in_group >= 0:
            core_rect = primary[dominant_core_index_in_group]
            dominant_component_bbox = core_rect
            dominant_component_ink = refinement.core_ink
            dominant_component_score = refinement.core_score

        if refined_components:
            x0 = min(c.x for c in refined_components)
            y0 = min(c.y for c in refined_components)
            x1 = max(c.x + c.width for c in refined_components)
            y1 = max(c.y + c.height for c in refined_components)
            bbox = Rect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
            refined_signature_bbox = bbox
            quality, quality_base, quality_completeness_factor = (
                _compute_extraction_quality_v24(
                    zone_mask,
                    bbox,
                    roi_area,
                    len(refined_components),
                    completeness_score,
                    completeness_min_factor,
                )
            )

    # Final fallback credibility verdict on the top-ranked winner: ranking
    # first does not imply credibility. Only evaluated on the fallback path;
    # on rejection the refined result still stands (diagnostics stay
    # truthful) but the orchestration must treat it as a clean failure.
    if (
        final_winner_check
        and selected_orig is not None
        and bbox is not None
        and refined_components
    ):
        winner_reject_reason = _fallback_winner_plausibility(
            bbox, refined_components, zone_mask, roi
        ) or ""
        if winner_reject_reason:
            for row in fallback_candidate_table:
                if row.index == selected_orig:
                    row.status = FALLBACK_STATUS_FINAL_REJECT
                    row.selected = False

    return InkAnalysis(
        roi=roi,
        micr_band_top=micr_band_top,
        binary_mask=zone_mask,
        all_components=all_components,
        kept_components=kept,
        rejected_components=rejected,
        micr_rejected_components=micr_rejected,
        groups=groups,
        group_scores=group_scores,
        selected_group_index=selected_group_index,
        selection_reason=selection_reason,
        rejection_reasons=rejection_reasons,
        primary_group=primary,
        dominant_core_index_in_group=dominant_core_index_in_group,
        dominant_component_bbox=dominant_component_bbox,
        dominant_component_ink=dominant_component_ink,
        dominant_component_score=dominant_component_score,
        core_component_indices=core_component_indices,
        discarded_from_selected_group_indices=discarded_from_selected_group_indices,
        refined_components=refined_components,
        refined_signature_bbox=refined_signature_bbox,
        refinement_reason=refinement_reason,
        directional_accepted_indices=directional_accepted_indices,
        directional_accept_details=directional_accept_details,
        completeness_score=completeness_score,
        completeness_refined_ink=completeness_refined_ink,
        completeness_reference_ink=completeness_reference_ink,
        quality_base=quality_base,
        quality_completeness_factor=quality_completeness_factor,
        bbox=bbox,
        quality=quality,
        recovered_components=recovered_components,
        crop_completeness=crop_completeness,
        graphic_rejected_group_indices=graphic_rejected_group_indices,
        graphic_rejected_group_reasons=graphic_rejected_group_reasons,
        graphic_rejected_group_boxes=graphic_rejected_group_boxes,
        fallback_candidate_table=fallback_candidate_table,
        winner_reject_reason=winner_reject_reason,
    )


def _check_candidate_credibility(
    analysis: InkAnalysis, min_credible_quality: float
) -> tuple[bool, str]:
    """Decide whether a V2.4 ``InkAnalysis`` is a credible signature candidate.

    A candidate is credible only when the UNCHANGED V2.4 pipeline produced a
    refined bounding box, kept at least one refined component and reached the
    minimum technical quality. No new AI score is invented: the gate reuses the
    existing ``quality`` heuristic.
    """
    if analysis.bbox is None:
        return False, "aucun contenu détecté (bbox None)."
    if not analysis.refined_components:
        return False, "aucune composante raffinée conservée."
    if analysis.quality < min_credible_quality:
        return (
            False,
            "qualité %.4f sous le seuil de crédibilité %.2f."
            % (analysis.quality, min_credible_quality),
        )
    return True, "candidat crédible."


def _graphic_short_status(reason: str) -> str:
    """Compact table status for a graphic-guard rejection reason."""
    if "règle" in reason:
        return "Rejeté — règle"
    return "Rejeté — bloc graphique"


def _fallback_winner_plausibility(
    bbox: Rect,
    refined_components: list[Rect],
    binary_mask: np.ndarray,
    roi: Rect,
) -> str | None:
    """Fallback-ONLY final credibility check on the top-ranked winner.

    Ranking first is not enough: the surviving group with the best
    signature-likeness score must still be plausibly handwritten. Uses
    existing measurable features only (refined bbox geometry relative to the
    full document, refined-component count/uniformity/alignment, TRUE ink
    density) — never OCR content, color, absolute coordinates or layout
    knowledge. Returns a French rejection reason, else None (accepted).
    """
    n_refined = len(refined_components)
    if n_refined <= 0:
        return None

    height_ratio = bbox.height / max(1, roi.height)
    aspect = bbox.width / max(1, bbox.height)

    region = binary_mask[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
    true_density = int(np.count_nonzero(region)) / max(
        1, bbox.width * bbox.height
    )
    heights = [c.height for c in refined_components]
    cys = [c.y + c.height / 2 for c in refined_components]
    vspread = (
        (max(cys) - min(cys)) / max(1, bbox.height) if n_refined >= 2 else 0.0
    )
    uniformity = max(heights) / max(1, min(heights))
    max_height_ratio = max(heights) / max(1, roi.height)

    # FW_SLAB: merged printed-text slab that slipped past the guards.
    if (
        n_refined <= FW_SLAB_MAX_COMPONENTS
        and aspect >= FW_SLAB_MIN_ASPECT
        and height_ratio < FW_SLAB_MAX_HEIGHT_RATIO
    ):
        return (
            "Candidat final écarté : texte probable (bloc %dx%d, allure %.1f, "
            "hauteur relative %.3f, %d composante(s), densité d'encre "
            "réelle %.3f)."
            % (
                bbox.width, bbox.height, aspect, height_ratio,
                n_refined, true_density,
            )
        )

    # FW_ROW: row of uniform, aligned, short glyphs (not varied strokes).
    if (
        n_refined >= FW_ROW_MIN_COMPONENTS
        and uniformity <= FW_ROW_MAX_HEIGHT_UNIFORMITY
        and vspread <= FW_ROW_MAX_VSPREAD
        and aspect >= FW_ROW_MIN_ASPECT
        and max_height_ratio < FW_ROW_MAX_HEIGHT_RATIO
    ):
        return (
            "Candidat final écarté : ligne de caractères probable "
            "(%d composantes uniformes, allure %.1f, dispersion verticale "
            "%.3f)."
            % (n_refined, aspect, vspread)
        )

    # FW_LINE: extreme aspect in either direction (rule or straight
    # fragment), never a complete signature.
    if aspect >= FW_LINE_MIN_ASPECT or aspect <= 1.0 / FW_LINE_MIN_ASPECT:
        return (
            "Candidat final écarté : trait incompatible (allure %.1f, "
            "bloc %dx%d)."
            % (aspect, bbox.width, bbox.height)
        )

    return None


def _roi_signature_plausibility(analysis: InkAnalysis) -> str | None:
    """ROI-ONLY gate: reject a refined candidate that looks like printed text.

    Called only for the primary ROI decision, after ``_check_candidate_credibility``
    passed. Uses existing measurable properties only (refined bbox geometry
    relative to the ROI, refined-component count, TRUE ink density from the
    binary mask) — never OCR content, color or absolute coordinates.

    Returns a French rejection reason when the candidate matches one of the
    conservative text-like conjunctions (see ``ROI_PLAUS_*`` constants),
    else None (plausible signature, keep the ROI result as-is).
    """
    bbox = analysis.bbox
    roi = analysis.roi
    if bbox is None or not analysis.refined_components:
        return None

    n_refined = len(analysis.refined_components)
    height_ratio = bbox.height / max(1, roi.height)
    width_ratio = bbox.width / max(1, roi.width)
    aspect = bbox.width / max(1, bbox.height)

    mask = analysis.binary_mask
    region = mask[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
    true_density = int(np.count_nonzero(region)) / max(
        1, bbox.width * bbox.height
    )

    # P1: merged printed-text slab (e.g. a table header fused into one wide,
    # shallow block). Genuine wide signatures always carry more components
    # (4-5) or a much lower aspect; genuine shallow ones are never this wide.
    if (
        n_refined <= ROI_PLAUS_SLAB_MAX_COMPONENTS
        and aspect >= ROI_PLAUS_SLAB_MIN_ASPECT
        and height_ratio < ROI_PLAUS_SLAB_MAX_HEIGHT_RATIO
    ):
        return (
            "ROI écartée : candidat trop compact / probablement textuel "
            "(bloc %dx%d, allure %.1f, hauteur relative %.3f, %d composante(s), "
            "densité d'encre réelle %.3f)."
            % (
                bbox.width, bbox.height, aspect, height_ratio,
                n_refined, true_density,
            )
        )

    # P1b: dense shallow slab with slightly lower aspect. No genuine sample
    # combines this width with this shallowness, solidity and few components.
    if (
        n_refined <= ROI_PLAUS_SLAB_MAX_COMPONENTS
        and aspect >= ROI_PLAUS_DENSE_MIN_ASPECT
        and height_ratio < ROI_PLAUS_DENSE_MAX_HEIGHT_RATIO
        and true_density >= ROI_PLAUS_DENSE_MIN_DENSITY
    ):
        return (
            "ROI écartée : candidat trop compact / probablement textuel "
            "(bloc dense %dx%d, allure %.1f, hauteur relative %.3f, "
            "densité d'encre réelle %.3f)."
            % (
                bbox.width, bbox.height, aspect, height_ratio, true_density,
            )
        )

    # P0: tiny solid speck (e.g. a lone printed glyph). Far smaller and more
    # solid than any genuine single-stroke signature.
    if (
        n_refined == 1
        and width_ratio < ROI_PLAUS_SPECK_MAX_WIDTH_RATIO
        and height_ratio < ROI_PLAUS_SPECK_MAX_HEIGHT_RATIO
        and true_density >= ROI_PLAUS_SPECK_MIN_DENSITY
    ):
        return (
            "ROI écartée : géométrie peu compatible avec une signature "
            "(bloc %dx%d, %d composante, densité d'encre réelle %.3f)."
            % (bbox.width, bbox.height, n_refined, true_density)
        )

    # P2: lone solid rule crossing the ROI (a form gridline, table border or
    # underline without its signature). Straight and solid, never a complete
    # handwritten signature.
    if (
        n_refined == 1
        and (
            aspect >= ROI_PLAUS_RULE_MIN_ASPECT
            or aspect <= 1.0 / ROI_PLAUS_RULE_MIN_ASPECT
        )
        and true_density >= ROI_PLAUS_RULE_MIN_DENSITY
    ):
        return (
            "ROI écartée : trait incompatible avec une signature "
            "(bloc %dx%d, allure %.1f, densité d'encre réelle %.3f)."
            % (bbox.width, bbox.height, aspect, true_density)
        )

    return None


def _recover_local_strokes(
    refined: list[Rect],
    pool_rects: list[Rect],
    labels: np.ndarray,
    stats: np.ndarray,
    roi: Rect,
    micr_band_top: int,
    micr_max_height_ratio: float,
    micr_min_aspect_ratio: float,
    micr_min_width_ratio: float,
) -> tuple[list[Rect], CropCompleteness]:
    """Conservative local stroke recovery around an accepted signature.

    After a candidate is accepted as a credible signature, inspect kept ink
    components near the refined union for detached strokes plausibly
    belonging to the same handwriting (leading flourish left, finishing
    stroke right, small mark above/below). This runs on BOTH paths (ROI +
    fallback) because the ROI directional pass is strictly rightward and the
    V2.3 ink gate can strand a genuinely adjacent stroke.

    ``pool_rects`` holds kept components outside the refined set in
    deterministic (x, y) order. Attachment uses only relative geometry (see
    ``REC_*`` constants) plus the existing MICR-like and graphic-guard
    rejections. Expansion is iterative but strictly bounded
    (``REC_MAX_PASSES`` passes, union area capped at ``REC_MAX_AREA_GROWTH``
    x the initial union). Deterministic: fixed candidate order, union
    updated in place.

    Returns ``(recovered_rects, completeness_evidence)``.
    """
    initial_union = _group_union_bbox(refined)
    initial_area = max(1, initial_union.width * initial_union.height)
    u0w = max(1, initial_union.width)
    u0h = max(1, initial_union.height)
    max_h_gap = REC_MAX_H_GAP_RATIO * u0w
    max_ink_gap = REC_MAX_INK_RATIO * max(u0w, u0h)
    max_v_gap = REC_ABOVE_BELOW_MAX_V_GAP_RATIO * u0h

    stats_by_rect: dict[tuple[int, int, int, int], int] = {}
    for lab in range(1, len(stats)):
        x, y, w, h, _ = stats[lab]
        stats_by_rect[(int(x), int(y), int(w), int(h))] = lab

    def mask_of(rect: Rect) -> np.ndarray | None:
        lab = stats_by_rect.get((rect.x, rect.y, rect.width, rect.height))
        if lab is None:
            return None
        return labels == lab

    refined_masks = [mask_of(c) for c in refined]
    pool: list[tuple[Rect, np.ndarray | None, int]] = []
    for rect in pool_rects:
        mask = mask_of(rect)
        true_ink = int(np.count_nonzero(mask)) if mask is not None else rect.width * rect.height
        pool.append((rect, mask, true_ink))
    # Cache candidate pixel coordinates once (masks are zone-sized).
    pool_pixels: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray] | None] = {}
    for rect, mask, _ink in pool:
        if mask is None:
            pool_pixels[(rect.x, rect.y, rect.width, rect.height)] = None
        else:
            pool_pixels[(rect.x, rect.y, rect.width, rect.height)] = np.where(mask)

    current = list(refined)
    current_masks: list[np.ndarray | None] = list(refined_masks)
    accepted: set[tuple[int, int, int, int]] = set()
    recovered: list[Rect] = []
    strokes: list[RecoveredStroke] = []
    iterations = 0

    for _ in range(REC_MAX_PASSES):
        union = _group_union_bbox(current)
        union_mask: np.ndarray | None = None
        for m in current_masks:
            if m is not None:
                union_mask = m.copy() if union_mask is None else np.logical_or(union_mask, m)
        if union_mask is None:
            break
        ink_dist = _chebyshev_ink_distance(union_mask.astype(np.uint8))
        added_this_pass = False
        for rect, mask, true_ink in pool:
            key = (rect.x, rect.y, rect.width, rect.height)
            if key in accepted:
                continue
            pixels = pool_pixels[key]
            verdict = _recovery_verdict(
                rect, mask, pixels, true_ink, union, u0w, u0h,
                max_h_gap, max_v_gap, max_ink_gap, ink_dist,
                roi, micr_band_top, micr_max_height_ratio,
                micr_min_aspect_ratio, micr_min_width_ratio,
            )
            if verdict is None:
                continue
            direction, h_gap, v_gap, reason = verdict
            tentative = _group_union_bbox(current + [rect])
            if (tentative.width * tentative.height) > REC_MAX_AREA_GROWTH * initial_area:
                continue
            accepted.add(key)
            current.append(rect)
            current_masks.append(mask)
            recovered.append(rect)
            strokes.append(
                RecoveredStroke(
                    direction=direction, bbox=rect,
                    h_gap=h_gap, v_gap=v_gap, reason=reason,
                )
            )
            added_this_pass = True
        if not added_this_pass:
            break
        iterations += 1

    final_union = _group_union_bbox(current)
    final_area = max(1, final_union.width * final_union.height)
    n_left = sum(1 for s in strokes if s.direction == "gauche")
    n_right = sum(1 for s in strokes if s.direction == "droite")
    n_other = len(strokes) - n_left - n_right
    if strokes:
        status = (
            "Recadrage étendu : %d trait(s) voisin(s) rattaché(s)."
            % len(strokes)
        )
    else:
        status = "Recadrage complet — aucun trait voisin crédible à rattacher."
    completeness = CropCompleteness(
        initial_bbox=initial_union,
        final_bbox=final_union,
        initial_component_count=len(refined),
        recovered_component_count=len(recovered),
        recovered_left=n_left,
        recovered_right=n_right,
        recovered_other=n_other,
        iterations=iterations,
        expansion_ratio=round(final_area / initial_area, 3),
        status=status,
        recovered_strokes=strokes,
    )
    return recovered, completeness


def _apply_local_stroke_recovery(
    analysis: InkAnalysis,
    micr_max_height_ratio: float,
    micr_min_aspect_ratio: float,
    micr_min_width_ratio: float,
) -> None:
    """Extend an already accepted candidate's crop without changing V2.4.

    Credibility, plausibility, ranking and quality are decided before this
    function runs. Only the final crop bbox and crop-completeness diagnostics
    are updated.
    """
    if analysis.refined_signature_bbox is None or not analysis.refined_components:
        return

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        analysis.binary_mask, connectivity=8
    )
    refined_keys = {
        (c.x, c.y, c.width, c.height) for c in analysis.refined_components
    }
    recovery_pool = sorted(
        (
            c
            for c in analysis.kept_components
            if (c.x, c.y, c.width, c.height) not in refined_keys
        ),
        key=lambda c: (c.x, c.y),
    )
    recovered, completeness = _recover_local_strokes(
        analysis.refined_components,
        recovery_pool,
        labels,
        stats,
        analysis.roi,
        analysis.micr_band_top,
        micr_max_height_ratio,
        micr_min_aspect_ratio,
        micr_min_width_ratio,
    )
    analysis.recovered_components = recovered
    analysis.crop_completeness = completeness
    analysis.bbox = completeness.final_bbox


def _recovery_verdict(
    rect: Rect,
    mask: np.ndarray | None,
    pixels: tuple[np.ndarray, np.ndarray] | None,
    true_ink: int,
    union: Rect,
    u0w: int,
    u0h: int,
    max_h_gap: float,
    max_v_gap: float,
    max_ink_gap: float,
    ink_dist: np.ndarray,
    roi: Rect,
    micr_band_top: int,
    micr_max_height_ratio: float,
    micr_min_aspect_ratio: float,
    micr_min_width_ratio: float,
) -> tuple[str, int, int, str] | None:
    """Single-candidate attachment verdict: (direction, h_gap, v_gap, reason)
    or None. Never absorbs MICR-like components, graphic-guard hits (bars,
    solid blocks, long rules), distant text, or oversized components."""
    if mask is None or pixels is None:
        return None
    if _is_micr_like(
        rect, roi, micr_band_top, micr_max_height_ratio,
        micr_min_aspect_ratio, micr_min_width_ratio,
    ):
        return None
    if _graphic_rejection_reason([rect], roi, true_ink) is not None:
        return None

    ys, xs = pixels
    if xs.size == 0:
        return None

    v_gap = _axis_gap(
        rect.y, rect.y + rect.height, union.y, union.y + union.height,
    )

    if rect.x + rect.width <= union.x:
        direction = "gauche"
        h_gap = union.x - (rect.x + rect.width)
    elif rect.x >= union.x + union.width:
        direction = "droite"
        h_gap = rect.x - (union.x + union.width)
    else:
        # Horizontally overlapping the envelope: only a small mark strictly
        # above/below qualifies (i dot and the like), never a rule or label.
        # Anything vertically overlapping the union is already inside the
        # cropped area: nothing to recover.
        if rect.y + rect.height <= union.y:
            direction = "haut"
            v_gap = union.y - (rect.y + rect.height)
        elif rect.y >= union.y + union.height:
            direction = "bas"
            v_gap = rect.y - (union.y + union.height)
        else:
            return None
        overlap = min(rect.x + rect.width, union.x + union.width) - max(rect.x, union.x)
        if overlap < REC_ABOVE_BELOW_MIN_OVERLAP_RATIO * rect.width:
            return None
        aspect_mark = rect.width / max(1, rect.height)
        if not (REC_ABOVE_BELOW_MIN_ASPECT <= aspect_mark <= REC_ABOVE_BELOW_MAX_ASPECT):
            return None
        if v_gap > max_v_gap:
            return None
        if rect.height > REC_ABOVE_BELOW_MAX_SIZE_RATIO * u0h:
            return None
        if rect.width > REC_ABOVE_BELOW_MAX_SIZE_RATIO * u0w:
            return None
        if ink_dist[ys, xs].min() > max_ink_gap:
            return None
        return (
            direction, 0, v_gap,
            "petite marque proche (%s, écart V %d px, recouvrement %.2f)."
            % (direction, v_gap, overlap / max(1, rect.width)),
        )

    # Side attach (left/right detached stroke): vertical overlap, small
    # horizontal gap, stroke-like height, bounded width, close ink.
    if v_gap != 0:
        return None
    if h_gap > max_h_gap:
        return None
    if rect.height < REC_MIN_HEIGHT_RATIO * u0h:
        return None
    if rect.height > REC_MAX_HEIGHT_RATIO * u0h:
        return None
    if rect.width > REC_MAX_WIDTH_RATIO * u0w:
        return None
    if ink_dist[ys, xs].min() > max_ink_gap:
        return None
    return (
        direction, h_gap, v_gap,
        "trait voisin plausible à %s (écart H %d px, hauteur %d px)."
        % ("gauche" if direction == "gauche" else "droite", h_gap, rect.height),
    )


def _build_final_crop_png(
    source_crop: np.ndarray, bbox: Rect, bbox_margin: float
) -> tuple[bytes, int, int]:
    """Build the final crop PNG with the standard margin logic.

    Shared verbatim by the ROI path and the fallback path so both produce the
    same margin behavior. ``source_crop`` is the ROI crop (primary mode) or
    the full image (fallback mode) ; ``bbox`` uses the same coordinates.
    Returns ``(png_bytes, crop_width, crop_height)``.
    """
    src_h, src_w = source_crop.shape[:2]
    margin_x = int(round(bbox.width * bbox_margin))
    margin_y = int(round(bbox.height * bbox_margin))

    crop_x = _clip(bbox.x - margin_x, 0, src_w - 1)
    crop_y = _clip(bbox.y - margin_y, 0, src_h - 1)
    crop_x1 = _clip(bbox.x + bbox.width + margin_x, crop_x + 1, src_w)
    crop_y1 = _clip(bbox.y + bbox.height + margin_y, crop_y + 1, src_h)

    final_crop = source_crop[crop_y:crop_y1, crop_x:crop_x1]
    if final_crop.size == 0 or final_crop.shape[0] < 2 or final_crop.shape[1] < 2:
        raise SignatureExtractionError("Crop final vide.")

    ok, encoded = cv2.imencode(".png", final_crop)
    if not ok:
        raise SignatureExtractionError("Impossible d'encoder le crop en PNG.")
    return encoded.tobytes(), int(final_crop.shape[1]), int(final_crop.shape[0])


def run_hybrid_localization(
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
) -> HybridLocalizationOutcome:
    """Run the hybrid localization: expected ROI first, full document fallback.

    Both stages reuse the UNCHANGED V2.4 ``_analyze_ink`` pipeline (same
    grayscale/blur, Otsu, morphology, connected components, MICR filtering,
    grouping, signature-likeness ranking, dominant-core refinement, directional
    continuation and completeness quality). The fallback only reruns that same
    logic on the full document with the same relative thresholds, plus a
    group-scale MICR guard (a merged MICR row rejected via the unchanged
    ``_is_micr_like`` predicate applied to the group union bbox),
    structural graphic guards (full-bleed bars, near-solid blocks and long
    thin rules rejected via ``_graphic_rejection_reason`` before ranking),
    the guarded leftward continuation mirror, and a final credibility check
    on the top-ranked winner (``_fallback_winner_plausibility``) ; surviving
    candidate groups are ranked by the UNCHANGED existing signature-likeness
    score (``_select_primary_group`` is deterministic: first maximum wins
    ties). If every group is rejected — or the winner is finally implausible
    — no candidate is selected (safe failure).

    Never raises for a mere absence of content: inspect ``final_analysis``
    (None when neither stage found a credible candidate).
    """
    height, width = image.shape[:2]

    roi = _compute_roi(
        width, height, roi_x_start, roi_y_start, roi_x_end, roi_y_end
    )

    def _analyze(
        gray: np.ndarray,
        zone: Rect,
        reject_micr_like_groups: bool = False,
        reject_graphic_groups: bool = False,
        allow_leftward_continuation: bool = False,
        final_winner_check: bool = False,
    ) -> InkAnalysis:
        return _analyze_ink(
            gray,
            zone,
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
            reject_micr_like_groups=reject_micr_like_groups,
            reject_graphic_groups=reject_graphic_groups,
            allow_leftward_continuation=allow_leftward_continuation,
            final_winner_check=final_winner_check,
        )

    roi_crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    gray_roi = cv2.GaussianBlur(
        cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY), (3, 3), 0
    )
    roi_analysis = _analyze(gray_roi, roi)
    roi_credible, roi_failure_reason = _check_candidate_credibility(
        roi_analysis, min_credible_quality
    )
    if roi_credible:
        # Plausibility gate (ROI only): a high-quality crop can still be a
        # printed-text fragment. Rejecting here marks the ROI as not credible
        # so the existing full-document fallback runs; the V2.4 core stages
        # above are untouched.
        plausibility_rejection = _roi_signature_plausibility(roi_analysis)
        if plausibility_rejection is not None:
            roi_credible = False
            roi_failure_reason = plausibility_rejection
    if roi_credible:
        _apply_local_stroke_recovery(
            roi_analysis,
            micr_max_height_ratio,
            micr_min_aspect_ratio,
            micr_min_width_ratio,
        )
        return HybridLocalizationOutcome(
            roi=roi,
            roi_analysis=roi_analysis,
            roi_credible=True,
            roi_failure_reason="",
            fallback_analysis=None,
            localization_mode=LOCALIZATION_MODE_ROI,
            final_analysis=roi_analysis,
            final_is_full_image=False,
        )

    # --- Full-document fallback: same V2.4 logic on the whole document ---
    # plus a group-scale MICR guard (a merged MICR row is obvious printed
    # content even when its isolated characters pass the per-component
    # filter at full-document scale), structural graphic guards (solid
    # banners, full-bleed bars and long thin rules can never be handwriting),
    # the guarded leftward continuation mirror, and a final credibility check
    # on the top-ranked winner (ranking first never implies credibility).
    full_zone = Rect(x=0, y=0, width=width, height=height)
    gray_full = cv2.GaussianBlur(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0
    )
    fallback_analysis = _analyze(
        gray_full,
        full_zone,
        reject_micr_like_groups=True,
        reject_graphic_groups=True,
        allow_leftward_continuation=True,
        final_winner_check=True,
    )
    fallback_credible, _ = _check_candidate_credibility(
        fallback_analysis, min_credible_quality
    )
    fallback_reject_reason = ""
    if fallback_credible and fallback_analysis.winner_reject_reason:
        # The best surviving candidate is still clearly implausible: clean
        # failure, never persist a random object. The analysis (with its
        # explainability table) is kept for diagnostics.
        fallback_credible = False
        fallback_reject_reason = fallback_analysis.winner_reject_reason
    if fallback_credible:
        _apply_local_stroke_recovery(
            fallback_analysis,
            micr_max_height_ratio,
            micr_min_aspect_ratio,
            micr_min_width_ratio,
        )
        return HybridLocalizationOutcome(
            roi=roi,
            roi_analysis=roi_analysis,
            roi_credible=False,
            roi_failure_reason=roi_failure_reason,
            fallback_analysis=fallback_analysis,
            localization_mode=LOCALIZATION_MODE_GLOBAL_FALLBACK,
            final_analysis=fallback_analysis,
            final_is_full_image=True,
        )

    return HybridLocalizationOutcome(
        roi=roi,
        roi_analysis=roi_analysis,
        roi_credible=False,
        roi_failure_reason=roi_failure_reason,
        fallback_analysis=fallback_analysis,
        localization_mode=LOCALIZATION_MODE_ROI,
        final_analysis=None,
        final_is_full_image=False,
        fallback_reject_reason=fallback_reject_reason,
    )


def extract_signature(
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
) -> SignatureExtractionResult:
    """Extract a candidate signature crop from a decoded OpenCV image.

    Hybrid localization: the unchanged V2.4 pipeline runs inside the expected
    cheque ROI first ; only when it yields no credible candidate is the same
    pipeline rerun on the full document. V5-A is never involved here.

    Raises SignatureExtractionError for invalid input, empty crops, or when
    neither stage finds a credible candidate (no forced crop, no fabricated
    quality).
    """
    if image is None:
        raise SignatureExtractionError("Image non décodée.")

    height, width = image.shape[:2]
    if width < 32 or height < 32:
        raise SignatureExtractionError(
            f"Image trop petite pour l'extraction ({width}x{height})."
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

    analysis = outcome.final_analysis
    if analysis is None or analysis.bbox is None:
        raise SignatureExtractionError(
            "Aucune signature exploitable détectée (ni dans la ROI principale "
            "ni dans l'ensemble du document)."
        )

    bbox = analysis.bbox

    if outcome.final_is_full_image:
        source_crop = image
        candidate_roi = Rect(x=0, y=0, width=width, height=height)
    else:
        source_crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        candidate_roi = roi

    png_bytes, crop_w, crop_h = _build_final_crop_png(
        source_crop, bbox, bbox_margin
    )

    if outcome.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK:
        fb = outcome.fallback_analysis
        fallback_candidate_count = len(fb.groups) if fb is not None else 0
        fallback_selected_index = analysis.selected_group_index
        if (
            fb is not None
            and 0 <= fallback_selected_index < len(fb.group_scores)
        ):
            fallback_selected_score = round(
                fb.group_scores[fallback_selected_index], 6
            )
        else:
            fallback_selected_score = 0.0
        fallback_selection_reason = analysis.selection_reason
    else:
        fallback_candidate_count = 0
        fallback_selected_index = -1
        fallback_selected_score = 0.0
        fallback_selection_reason = ""

    logger.info(
        "Extraction V2.4: image %dx%d, ROI %dx%d, mode=%s, micr_band_top=%d, "
        "bbox %dx%d, all=%d kept=%d rejected=%d micr_rejected=%d groups=%d, "
        "selected_group=%d, core=(%d, encre %d, score %.5f), refined=%d, "
        "directional=%d, discarded=%d, completeness %.3f (%d/%d), quality "
        "%.4f (base %.4f x facteur %.3f), reason=%s",
        width,
        height,
        roi.width,
        roi.height,
        outcome.localization_mode,
        analysis.micr_band_top,
        bbox.width,
        bbox.height,
        len(analysis.all_components),
        len(analysis.kept_components),
        len(analysis.rejected_components),
        len(analysis.micr_rejected_components),
        len(analysis.groups),
        analysis.selected_group_index,
        analysis.dominant_core_index_in_group,
        analysis.dominant_component_ink,
        analysis.dominant_component_score,
        len(analysis.refined_components),
        len(analysis.directional_accepted_indices),
        len(analysis.discarded_from_selected_group_indices),
        analysis.completeness_score,
        analysis.completeness_refined_ink,
        analysis.completeness_reference_ink,
        analysis.quality,
        analysis.quality_base,
        analysis.quality_completeness_factor,
        analysis.refinement_reason,
    )

    return SignatureExtractionResult(
        original_width=width,
        original_height=height,
        candidate_roi=candidate_roi,
        signature_bbox=bbox,
        extraction_quality=analysis.quality,
        image_format="png",
        signature_png_bytes=png_bytes,
        localization_mode=outcome.localization_mode,
        fallback_candidate_count=fallback_candidate_count,
        fallback_selected_index=fallback_selected_index,
        fallback_selected_score=fallback_selected_score,
        fallback_selection_reason=fallback_selection_reason,
        final_crop_width=crop_w,
        final_crop_height=crop_h,
    )
