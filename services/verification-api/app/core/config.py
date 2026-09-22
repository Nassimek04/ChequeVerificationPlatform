from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "verification-api"
    app_version: str = "0.1.0"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5285", "http://127.0.0.1:5285"]

    # Candidate signature region of interest, expressed as relative ratios
    # (0..1) of the full image. The strategy assumes the signature sits in
    # the bottom-right area of a standard cheque. These values are NOT
    # universal: they must be calibrated on the real cheque formats used
    # during the internship.
    signature_roi_x_start: float = 0.40
    signature_roi_y_start: float = 0.40
    signature_roi_x_end: float = 0.98
    signature_roi_y_end: float = 0.98

    # Padding ratio added around the detected ink bounding box
    # (fraction of the bbox dimensions).
    signature_bbox_margin: float = 0.08

    # V2.1: bottom band of the ROI flagged as a "MICR risk zone" (fraction of
    # the ROI height). It is NOT physically removed: the whole ROI is analyzed
    # and only components that look like MICR / printed text inside this band
    # are rejected. Signature strokes that descend into the band are kept.
    # Name kept for compatibility (historical SIGNATURE_BOTTOM_EXCLUSION_RATIO).
    signature_bottom_exclusion_ratio: float = 0.10

    # Component filtering criteria, all RELATIVE to the ROI dimensions/area
    # (the whole ROI is the analysis zone in V2.1), never fixed pixels.
    # - drop components smaller than this fraction of the ROI area (specks) ;
    # - drop a component covering more than this fraction of the ROI area
    #   (whole-ROI blobs, e.g. merged grids/motifs) ;
    # - drop components thinner than these fractions of the ROI width/height.
    signature_min_component_area_ratio: float = 0.0005
    signature_max_component_area_ratio: float = 0.60
    signature_min_component_width_ratio: float = 0.01
    signature_min_component_height_ratio: float = 0.01

    # V2.2 MICR / printed-text rejection criteria (only evaluated for
    # components whose vertical center lies inside the MICR risk zone):
    # - a component is "very horizontal" when width/height exceeds
    #   signature_micr_min_aspect_ratio (e.g. a long rule or a merged line) ;
    # - a component is "short and wide" when its height ratio is below
    #   signature_micr_max_height_ratio AND its width ratio is at least
    #   signature_micr_min_width_ratio (e.g. a printed character row).
    # The "short" threshold is kept tight (V2.1: 0.12) so a long diagonal
    # signature stroke descending into the band is NOT mistaken for a printed
    # character: a genuine MICR row is only ~4% of the ROI height.
    signature_micr_max_height_ratio: float = 0.06
    signature_micr_min_aspect_ratio: float = 6.0
    signature_micr_min_width_ratio: float = 0.03

    # V2.2 spatial grouping of kept components: two components belong to the
    # same signature group when both the horizontal and the vertical gap
    # between their bounding boxes are within this fraction of the ROI
    # width / height respectively. Tighter than V2.1 (0.15) so an isolated
    # printed component near the signature does not merge into its group.
    signature_component_merge_distance_ratio: float = 0.10

    # V2.1: minimum number of components a group must contain to be eligible as
    # the primary signature group (1 = a lone component is acceptable).
    signature_group_min_components: int = 1

    # Morphology close kernel size (odd integer). Kept small on purpose so the
    # close reconnects broken strokes of the same trace without merging
    # signature + text + MICR into one giant component.
    signature_morph_kernel_size: int = 3

    # V2.3 second-stage refinement (dominant signature core).
    #
    # After the V2.2 group-selection stage, the selected group can still be
    # contaminated by printed text / "(28)"-like components / bottom-row MICR
    # content whose bounding boxes touch the signature. V2.3 starts from the
    # single dominant component of the selected group (highest dominant-core
    # score: ink relative to ROI x vertical extent relative to ROI x ink
    # density) and re-expands only towards components whose ACTUAL ink is
    # genuinely close in 2D.
    #
    # - signature_core_ink_distance_ratio: a candidate component joins the
    #   refined signature structure only when at least one of its ink pixels
    #   lies within this fraction of the ROI width (Chebyshev distance) of the
    #   ink of an ALREADY-ACCEPTED component. Chebyshev distance means the
    #   horizontal AND the vertical separation are evaluated together against
    #   that SAME component (never one gap from component A and the other from
    #   component B). Conservative: printed rows separated from the signature
    #   ink are not absorbed.
    # - signature_core_max_h_gap_ratio / signature_core_max_v_gap_ratio:
    #   bounding-box same-pair rule (see _is_component_2d_near): the candidate
    #   bbox must be within these fractions of the ROI width/height of at least
    #   one accepted component. Kept generous (the ink distance is the decisive
    #   gate) so long thin diagonal strokes are never rejected.
    # - signature_core_min_height_ratio: a candidate whose height is below this
    #   fraction of the ROI height is treated as printed text / a speck and is
    #   never added, even when its ink is close. Handwritten strokes (main
    #   loop, nearby detached strokes, long descending diagonals) are taller
    #   than printed characters on cheque scans.
    signature_core_ink_distance_ratio: float = 0.018
    signature_core_max_h_gap_ratio: float = 0.15
    signature_core_max_v_gap_ratio: float = 0.15
    signature_core_min_height_ratio: float = 0.08

    # V2.4 directional continuation (single-pair).
    #
    # V2.3 rejected a genuine signature stroke (main loop + long rightward
    # continuation) when its ink lies farther than the core ink-distance
    # threshold (CHQ-0003 case): a large connected component carrying most of
    # the signature ink is penalized by the Chebyshev gate even though it is
    # the visible continuation of the same trace. V2.4 adds a guarded second
    # pass AFTER the V2.3 refinement (see ``_refine_signature_group``): a
    # candidate component is accepted when it is STRICTLY RIGHT of an
    # already-accepted component, is vertically aligned with it (v_gap == 0),
    # is tall enough (same V2.3 minimum) and its ink comes within a more
    # generous, directional-only ink distance of the ink of that SAME
    # component (per-component distance transform, never the union). The
    # single-pair rule keeps the anti-chaining property: a far component can
    # never be pulled in through an intermediate one.
    #
    # - signature_core_directional_h_gap_ratio: max horizontal bounding-box gap
    #   between the candidate and the anchor component (fraction of the ROI
    #   width).
    # - signature_core_directional_ink_distance_ratio: max Chebyshev ink
    #   distance between the candidate and the anchor component's ink (fraction
    #   of the ROI width). Deliberately more generous than the V2.3 core ratio
    #   (0.018) because it is evaluated against a single component and only
    #   towards the right.
    # - signature_completeness_min_factor: the extraction quality is multiplied
    #   by (min_factor + (1 - min_factor) * completeness) where completeness =
    #   refined ink / signature-like reference ink (see ``_refine_signature_group``).
    #   A complete signature keeps exactly the V2.3 quality; a signature
    #   truncated on its right can never score excellent. 0.5 keeps the penalty
    #   soft (a truncated signature loses at most 50% of the base quality).
    signature_core_directional_h_gap_ratio: float = 0.08
    signature_core_directional_ink_distance_ratio: float = 0.04
    signature_completeness_min_factor: float = 0.5

    # Hybrid localization credibility gate (technical heuristic, NOT an
    # authenticity score): an ROI candidate is kept as-is only when its V2.4
    # extraction quality reaches this threshold. Below it (or with no refined
    # bbox at all) the same V2.4 pipeline is rerun on the full document.
    # Kept low (0.20) so every ROI extraction that succeeds today keeps
    # identical behavior; only weak/junk ROI candidates trigger the fallback.
    signature_hybrid_min_credible_quality: float = 0.20

    # Fixed canvas used to normalize signatures before comparison. Signatures
    # are resized preserving the aspect ratio, then centered with padding on a
    # canvas of this size (no distortion). Width > height matches the typical
    # wide shape of a signature.
    signature_comparison_canvas_width: int = 256
    signature_comparison_canvas_height: int = 128

    # --- AI V2 signature verification (experimental metric model) ---
    # The trained AI V2 model (ResNet18, 128-D L2 embeddings) can be enabled for
    # the dedicated /api/signatures/compare-ai endpoint. When disabled, the
    # service starts normally and the AI endpoint returns a controlled
    # "unavailable" response; the OpenCV baseline stays functional either way.
    signature_ai_enabled: bool = True
    # Checkpoint path (absolute, or relative to the service root). The default
    # points to the trained AI V5-A metric model (Phase 7 frozen).
    signature_ai_checkpoint: str = "ai/checkpoints/metric_resnet18_v5a.pt"
    # Device selection: "auto" (CUDA if available, otherwise CPU), "cuda" or
    # "cpu". CUDA is never required.
    signature_ai_device: str = "auto"

    # --- PaddleOCR Docker Linux worker (V1) ---
    # The main Windows FastAPI proxies OCR to this isolated Linux container.
    # Windows native libpaddle.pyd is blocked by Smart App Control, so Paddle
    # runs only inside Docker. Keep localhost-only binding 127.0.0.1:8010.
    ocr_worker_url: str = "http://127.0.0.1:8010"
    # Timeout in seconds for worker HTTP call (cold init may need ~60s on first request)
    ocr_worker_timeout: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        roi_values = {
            "signature_roi_x_start": self.signature_roi_x_start,
            "signature_roi_y_start": self.signature_roi_y_start,
            "signature_roi_x_end": self.signature_roi_x_end,
            "signature_roi_y_end": self.signature_roi_y_end,
        }
        for name, value in roi_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} doit être compris entre 0 et 1 (reçu : {value}).")

        if not (
            self.signature_roi_x_start < self.signature_roi_x_end
            and self.signature_roi_y_start < self.signature_roi_y_end
        ):
            raise ValueError(
                "Les bornes de la ROI doivent vérifier start < end pour chaque axe."
            )

        if not 0.0 <= self.signature_bottom_exclusion_ratio < 0.5:
            raise ValueError(
                "signature_bottom_exclusion_ratio (zone de risque MICR) doit être "
                f"dans [0, 0.5) (reçu : {self.signature_bottom_exclusion_ratio})."
            )

        ratio_fields = {
            "signature_min_component_area_ratio": self.signature_min_component_area_ratio,
            "signature_max_component_area_ratio": self.signature_max_component_area_ratio,
            "signature_min_component_width_ratio": self.signature_min_component_width_ratio,
            "signature_min_component_height_ratio": self.signature_min_component_height_ratio,
        }
        for name, value in ratio_fields.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} doit être compris entre 0 (exclu) et 1 (reçu : {value}).")

        if (
            self.signature_max_component_area_ratio
            <= self.signature_min_component_area_ratio
        ):
            raise ValueError(
                "signature_max_component_area_ratio doit être supérieur à "
                "signature_min_component_area_ratio."
            )

        if not 0.0 < self.signature_micr_max_height_ratio <= 1.0:
            raise ValueError(
                "signature_micr_max_height_ratio doit être compris entre 0 (exclu) et 1 "
                f"(reçu : {self.signature_micr_max_height_ratio})."
            )

        if not self.signature_micr_min_aspect_ratio > 1.0:
            raise ValueError(
                "signature_micr_min_aspect_ratio doit être strictement supérieur à 1 "
                f"(reçu : {self.signature_micr_min_aspect_ratio})."
            )

        if not 0.0 < self.signature_micr_min_width_ratio <= 1.0:
            raise ValueError(
                "signature_micr_min_width_ratio doit être compris entre 0 (exclu) et 1 "
                f"(reçu : {self.signature_micr_min_width_ratio})."
            )

        if not 0.0 < self.signature_component_merge_distance_ratio <= 1.0:
            raise ValueError(
                "signature_component_merge_distance_ratio doit être compris entre 0 "
                f"(exclu) et 1 (reçu : {self.signature_component_merge_distance_ratio})."
            )

        if self.signature_group_min_components < 1:
            raise ValueError(
                "signature_group_min_components doit être >= 1 "
                f"(reçu : {self.signature_group_min_components})."
            )

        if (
            self.signature_morph_kernel_size < 1
            or self.signature_morph_kernel_size > 9
            or self.signature_morph_kernel_size % 2 == 0
        ):
            raise ValueError(
                "signature_morph_kernel_size doit être un entier impair entre 1 et 9 "
                f"(reçu : {self.signature_morph_kernel_size})."
            )

        for name, value in {
            "signature_core_ink_distance_ratio": self.signature_core_ink_distance_ratio,
            "signature_core_max_h_gap_ratio": self.signature_core_max_h_gap_ratio,
            "signature_core_max_v_gap_ratio": self.signature_core_max_v_gap_ratio,
            "signature_core_directional_h_gap_ratio": self.signature_core_directional_h_gap_ratio,
            "signature_core_directional_ink_distance_ratio": self.signature_core_directional_ink_distance_ratio,
        }.items():
            if not 0.0 < value <= 0.5:
                raise ValueError(
                    f"{name} doit être compris entre 0 (exclu) et 0.5 "
                    f"(reçu : {value})."
                )

        if not 0.0 < self.signature_core_min_height_ratio <= 1.0:
            raise ValueError(
                "signature_core_min_height_ratio doit être compris entre 0 (exclu) et 1 "
                f"(reçu : {self.signature_core_min_height_ratio})."
            )

        if self.ocr_worker_timeout < 5 or self.ocr_worker_timeout > 300:
            raise ValueError(
                "ocr_worker_timeout doit être compris entre 5 et 300 secondes "
                f"(reçu : {self.ocr_worker_timeout})."
            )

        if not 0.0 <= self.signature_completeness_min_factor <= 1.0:
            raise ValueError(
                "signature_completeness_min_factor doit être compris entre 0 et 1 "
                f"(reçu : {self.signature_completeness_min_factor})."
            )

        if not 0.0 <= self.signature_hybrid_min_credible_quality <= 1.0:
            raise ValueError(
                "signature_hybrid_min_credible_quality doit être compris entre 0 et 1 "
                f"(reçu : {self.signature_hybrid_min_credible_quality})."
            )

        for name, value in {
            "signature_comparison_canvas_width": self.signature_comparison_canvas_width,
            "signature_comparison_canvas_height": self.signature_comparison_canvas_height,
        }.items():
            if not 32 <= value <= 1024:
                raise ValueError(
                    f"{name} doit être compris entre 32 et 1024 (reçu : {value})."
                )

        if self.signature_ai_device not in ("auto", "cuda", "cpu"):
            raise ValueError(
                "signature_ai_device doit être 'auto', 'cuda' ou 'cpu' "
                f"(reçu : {self.signature_ai_device})."
            )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()