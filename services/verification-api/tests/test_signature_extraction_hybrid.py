"""Focused tests for the HYBRID LOCALIZATION strategy (ROI first, full-document fallback).

Direct function-level tests (no HTTP server needed):

A. signature inside the normal ROI => ROI behavior unchanged (mode "roi",
   ROI-local bbox, no fallback evidence).
B. no valid ROI candidate + signature elsewhere => full-document fallback
   selects it (mode "global_fallback", absolute bbox).
C. multiple full-document candidates => deterministic best candidate selected
   (highest existing signature-likeness score, stable across runs).
D. no credible signature anywhere => safe extraction failure
   (SignatureExtractionError, no forced crop).
E. the extraction module never calls V5-A / AI verification (source-level
   guard: no AI imports, comparison stays a separate service).
F. see the untouched test_signature_extraction_v24.py suite (run separately).

Run with the venv activated:
    python tests/test_signature_extraction_hybrid.py
or:
    pytest tests/test_signature_extraction_hybrid.py -q
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.signature_debug_service import build_signature_debug
from app.services.signature_extraction_service import (
    DEFAULT_MIN_CREDIBLE_QUALITY,
    LOCALIZATION_MODE_GLOBAL_FALLBACK,
    LOCALIZATION_MODE_ROI,
    SignatureExtractionError,
    _analyze_ink,
    _compute_roi,
    extract_signature,
    run_hybrid_localization,
)

IMAGE_W = 800
IMAGE_H = 355
ROI = (0.40, 0.40, 0.98, 0.98)
ROI_RECT = _compute_roi(IMAGE_W, IMAGE_H, *ROI)
assert (ROI_RECT.x, ROI_RECT.y, ROI_RECT.width, ROI_RECT.height) == (
    320, 142, 464, 206,
), ROI_RECT


def make_image():
    return np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)


def add_blob(image, cx, cy, ax=40, ay=24):
    cv2.ellipse(image, (cx, cy), (ax, ay), 8, 0, 360, (20, 20, 20), -1)


def add_solid_banner(image, x0=0, y0=0, x1=800, y1=60):
    """Large solid horizontal graphic (header-banner class of false positive).

    Generic geometry: no color/texture assumption beyond "dark solid fill".
    """
    cv2.rectangle(image, (x0, y0), (x1, y1), (30, 30, 180), -1)


def add_thin_rule(image, x0=50, y0=100, x1=650, y1=108):
    """Long thin horizontal rule (border-line class of false positive)."""
    cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), -1)


def add_printed_glyphs(image, x, y, n_glyphs, gw=7, gh=13, gap=2):
    """One printed-text line made of separated dark glyphs (generic geometry,
    no font rendering, no OCR content). Small gaps merge under morphology
    like real print; wide pitch keeps lines in separate groups."""
    for i in range(n_glyphs):
        gx = x + i * (gw + gap)
        cv2.rectangle(image, (gx, y), (gx + gw, y + gh), (20, 20, 20), -1)


def add_scribble(image, x, y, scale=1.0, thickness=3):
    """Handwriting-like signature proxy: zigzag strokes + underline flourish
    (thin strokes with whitespace, never a solid blob)."""
    pts1 = (np.array(
        [[0, 30], [20, 5], [45, 28], [70, 8], [95, 30], [120, 12]], np.int32,
    ) * scale).astype(np.int32) + np.array([x, y])
    pts2 = (np.array(
        [[10, 35], [60, 38], [130, 30]], np.int32,
    ) * scale).astype(np.int32) + np.array([x, y])
    cv2.polylines(image, [pts1], False, (20, 20, 20), thickness, cv2.LINE_AA)
    cv2.polylines(image, [pts2], False, (20, 20, 20), thickness, cv2.LINE_AA)


def make_text_table_doc():
    """Faithful failure-class document: logo + printed paragraphs + table
    with header text inside the ROI + genuine scribble signature outside
    the ROI (bottom-left, clear of the table so frozen fallback grouping
    keeps it its own candidate)."""
    img = make_image()
    cv2.rectangle(img, (20, 10), (120, 40), (20, 20, 20), 2)  # logo box
    add_printed_glyphs(img, 20, 70, 14, gap=3)    # paragraph line 1
    add_printed_glyphs(img, 20, 122, 14, gap=3)   # paragraph line 2
    cv2.line(img, (340, 178), (760, 178), (20, 20, 20), 1)  # table rule
    add_printed_glyphs(img, 350, 160, 13, gh=15)  # table header in ROI
    add_printed_glyphs(img, 350, 215, 10, gap=3)  # table row in ROI
    add_scribble(img, 40, 240, scale=1.5, thickness=5)  # signature, out ROI
    return img


def make_no_signature_doc():
    """Rich business document with NO handwriting: logo, heading,
    paragraphs, left-side table, horizontal + vertical rules, footer in the
    MICR band, light watermark blocks. The ROI holds a single horizontal
    form rule (no signature line content)."""
    img = make_image()
    cv2.rectangle(img, (20, 10), (120, 40), (20, 20, 20), 2)  # logo box
    add_printed_glyphs(img, 200, 18, 10)                      # heading
    add_printed_glyphs(img, 20, 70, 16)      # paragraph lines
    add_printed_glyphs(img, 20, 94, 16)
    add_printed_glyphs(img, 20, 118, 14)
    add_printed_glyphs(img, 40, 170, 12)     # table, left of ROI
    add_printed_glyphs(img, 40, 194, 10)
    cv2.line(img, (40, 216), (300, 216), (20, 20, 20), 1)
    cv2.rectangle(img, (50, 300), (650, 308), (20, 20, 20), -1)  # h rule (ROI)
    cv2.rectangle(img, (790, 60), (798, 290), (20, 20, 20), -1)  # v rule
    add_printed_glyphs(img, 60, 330, 12)     # footer (MICR band)
    for i in range(5):                       # light watermark blocks (off ROI)
        cv2.rectangle(img, (500 + i * 40, 100), (530 + i * 40, 112),
                      (210, 210, 210), -1)
    return img


def scribble_ink_bounds(x, y, scale=1.0, thickness=3):
    """True ink bounding box of the scribble proxy alone (for integrity
    assertions): redraw on blank, threshold, boundingRect."""
    probe = np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)
    add_scribble(probe, x, y, scale=scale, thickness=thickness)
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def test_a_roi_signature_unchanged():
    img = make_image()
    add_blob(img, 540, 250)  # inside the ROI (x 320..784, y 142..348)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    assert result.fallback_candidate_count == 0
    assert result.fallback_selected_index == -1
    assert result.extraction_quality >= DEFAULT_MIN_CREDIBLE_QUALITY
    # ROI-local bbox: must lie inside the ROI rectangle dimensions.
    bb = result.signature_bbox
    assert 0 <= bb.x and 0 <= bb.y
    assert bb.x + bb.width <= ROI_RECT.width
    assert bb.y + bb.height <= ROI_RECT.height
    assert result.candidate_roi == ROI_RECT
    # Same numbers as a direct V2.4 ROI analysis (pipeline untouched).
    gray = cv2.GaussianBlur(
        cv2.cvtColor(
            img[
                ROI_RECT.y : ROI_RECT.y + ROI_RECT.height,
                ROI_RECT.x : ROI_RECT.x + ROI_RECT.width,
            ],
            cv2.COLOR_BGR2GRAY,
        ),
        (3, 3),
        0,
    )
    direct = _analyze_ink(
        gray,
        ROI_RECT,
        micr_zone_ratio=0.10,
        min_component_area_ratio=0.0005,
        max_component_area_ratio=0.60,
        min_component_width_ratio=0.01,
        min_component_height_ratio=0.01,
        micr_max_height_ratio=0.06,
        micr_min_aspect_ratio=6.0,
        micr_min_width_ratio=0.03,
        component_merge_distance_ratio=0.10,
        group_min_components=1,
        morph_kernel_size=3,
        core_ink_distance_ratio=0.018,
        core_max_h_gap_ratio=0.15,
        core_max_v_gap_ratio=0.15,
        core_min_height_ratio=0.08,
        core_directional_h_gap_ratio=0.08,
        core_directional_ink_distance_ratio=0.04,
        completeness_min_factor=0.5,
    )
    assert direct.bbox == bb
    assert direct.quality == result.extraction_quality


def test_b_fallback_selects_outside_roi_signature():
    img = make_image()
    add_blob(img, 100, 60)  # top-left: outside the ROI
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    assert result.fallback_candidate_count >= 1
    assert result.fallback_selected_index >= 0
    assert result.fallback_selected_score > 0.0
    assert result.fallback_selection_reason != ""
    assert result.final_crop_width > 0 and result.final_crop_height > 0
    # Absolute bbox near the drawn blob (full-document coordinates).
    bb = result.signature_bbox
    assert bb.x <= 100 <= bb.x + bb.width
    assert bb.y <= 60 <= bb.y + bb.height
    # Candidate ROI is the full document in fallback mode.
    assert (result.candidate_roi.x, result.candidate_roi.y) == (0, 0)
    assert (result.candidate_roi.width, result.candidate_roi.height) == (
        IMAGE_W, IMAGE_H,
    )
    # Debug mirrors the same localization evidence + candidates image.
    dbg = build_signature_debug(img, *ROI)
    assert dbg.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    assert dbg.localization_label == "Recherche globale de secours"
    assert dbg.fallback_candidate_count >= 1
    assert dbg.fallback_selected_index == result.fallback_selected_index
    assert dbg.signature_bbox == bb
    assert dbg.fallback_candidates_png_bytes is not None
    assert dbg.signature_png_bytes is not None


def test_c_multiple_fallback_candidates_deterministic_best():
    img = make_image()
    add_blob(img, 120, 70, ax=48, ay=28)  # large signature-like blob
    add_blob(img, 650, 60, ax=22, ay=14)  # smaller blob, also outside ROI
    first = extract_signature(img, *ROI)
    assert first.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    for _ in range(3):
        again = extract_signature(img, *ROI)
        assert again.signature_bbox == first.signature_bbox
        assert again.fallback_selected_index == first.fallback_selected_index
        assert again.fallback_selected_score == first.fallback_selected_score
    # Best = highest existing signature-likeness group score.
    outcome = run_hybrid_localization(img, *ROI)
    assert outcome.fallback_analysis is not None
    scores = outcome.fallback_analysis.group_scores
    assert len(scores) >= 2
    expected = max(range(len(scores)), key=lambda i: scores[i])
    assert first.fallback_selected_index == expected
    # The large blob wins over the small one.
    assert first.signature_bbox.x < 300


def test_b2_fallback_prefers_signature_over_micr_row():
    img = make_image()
    add_blob(img, 100, 60)  # genuine signature outside the ROI
    for i in range(6):  # MICR-like row at the bottom (as in the V2.4 suite)
        cv2.rectangle(
            img,
            (460 + i * 28, 330),
            (460 + i * 28 + 20, 338),
            (15, 15, 15),
            -1,
        )
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 100 <= bb.x + bb.width
    assert bb.y <= 60 <= bb.y + bb.height


def test_d_no_signature_anywhere_fails_safely():
    try:
        extract_signature(make_image(), *ROI)
    except SignatureExtractionError as exc:
        assert "Aucune signature" in str(exc)
    else:
        raise AssertionError("blank document must raise SignatureExtractionError")
    # MICR-looking row only: still a safe failure, no forced crop.
    img = make_image()
    for i in range(6):
        cv2.rectangle(img, (460 + i * 28, 330), (460 + i * 28 + 20, 338), (15, 15, 15), -1)
    try:
        extract_signature(img, *ROI)
    except SignatureExtractionError:
        pass
    else:
        raise AssertionError("MICR-only document must raise SignatureExtractionError")


def test_e_extraction_never_touches_v5a():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app",
        "services",
        "signature_extraction_service.py",
    )
    with open(path, encoding="utf-8") as fh:
        source = fh.read().lower()
    for forbidden in (
        "ai_signature_verification",
        "signature_comparison_service",
        "resnet",
        "cosine",
        "torch",
        "embedding",
    ):
        assert forbidden not in source, forbidden


def test_f_banner_rejected_signature_selected():
    # Regression: solid header banner + handwritten blob elsewhere.
    img = make_image()
    add_solid_banner(img)
    add_blob(img, 150, 250)  # outside the ROI (x < 320)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 150 <= bb.x + bb.width
    assert bb.y <= 250 <= bb.y + bb.height
    # Debug explains the banner rejection truthfully + draws candidates.
    dbg = build_signature_debug(img, *ROI)
    assert dbg.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    assert len(dbg.fallback_rejected_candidates) >= 1
    reasons = " | ".join(c.reason for c in dbg.fallback_rejected_candidates)
    assert "largeur excessive" in reasons or "bloc trop dense" in reasons
    assert dbg.fallback_candidates_png_bytes is not None
    assert dbg.signature_png_bytes is not None


def test_f2_non_full_bleed_solid_block_rejected():
    # Solid graphic block that does NOT span the page: density gate catches it.
    img = make_image()
    cv2.rectangle(img, (50, 10), (450, 90), (30, 30, 180), -1)
    add_blob(img, 150, 250)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 150 <= bb.x + bb.width
    dbg = build_signature_debug(img, *ROI)
    reasons = " | ".join(c.reason for c in dbg.fallback_rejected_candidates)
    assert "bloc trop dense" in reasons


def test_f3_thin_rule_rejected():
    # Long thin hollow rule (low true density, rule geometry) + handwritten
    # blob: rule rejected by the rule gate, blob selected.
    img = make_image()
    cv2.rectangle(img, (50, 100), (650, 108), (20, 20, 20), 1)
    add_blob(img, 150, 250)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 150 <= bb.x + bb.width
    dbg = build_signature_debug(img, *ROI)
    reasons = " | ".join(c.reason for c in dbg.fallback_rejected_candidates)
    assert "gle horizontale" in reasons


def test_f4_banner_only_fails_safely():
    # A (solid banner only): no credible signature anywhere => safe failure,
    # never the "least bad" graphic crop.
    img = make_image()
    add_solid_banner(img)
    try:
        extract_signature(img, *ROI)
    except SignatureExtractionError as exc:
        assert "Aucune signature" in str(exc)
    else:
        raise AssertionError("banner-only document must raise SignatureExtractionError")
    dbg = build_signature_debug(img, *ROI)
    assert dbg.signature_png_bytes is None
    assert dbg.signature_bbox is None
    assert len(dbg.fallback_rejected_candidates) >= 1
    assert dbg.fallback_candidates_png_bytes is not None


def test_g_roi_text_slab_rejected_fallback_selects_signature():
    # Regression for the observed failure class: the ROI catches a printed
    # table header (~116x19 px slab at ~80% quality). The plausibility gate
    # must reject it so the fallback selects the handwritten signature.
    result = extract_signature(make_text_table_doc(), *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    # The extracted crop holds the scribble (bottom-left), not ROI text:
    # absolute bbox well clear of the ROI x-range (ROI starts at x=320).
    bb = result.signature_bbox
    assert bb.x <= 150 <= bb.x + bb.width
    assert bb.y <= 275 <= bb.y + bb.height
    assert bb.x + bb.width < ROI_RECT.x
    # Debug shows: ROI candidate found, rejected, fallback ran and selected.
    dbg = build_signature_debug(make_text_table_doc(), *ROI)
    assert dbg.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    assert "textuel" in dbg.roi_failure_reason
    assert dbg.fallback_candidate_count >= 1
    assert dbg.fallback_selected_index >= 0
    assert dbg.signature_png_bytes is not None


def test_g2_compact_signature_in_roi_accepted():
    # B: a valid compact scribble inside the ROI must NOT be rejected.
    img = make_image()
    add_scribble(img, 480, 220, scale=0.7, thickness=3)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    bb = result.signature_bbox
    assert bb.x + bb.width <= ROI_RECT.width
    assert bb.y + bb.height <= ROI_RECT.height
    dbg = build_signature_debug(img, *ROI)
    assert dbg.roi_failure_reason == ""


def test_g3_lone_glyph_rejected_fallback_selects_signature():
    # P0 path: an isolated printed glyph wins the ROI at ~80% quality;
    # geometry (tiny + solid) rejects it, fallback takes the scribble.
    img = make_image()
    add_printed_glyphs(img, 350, 160, 13, gh=15, gap=5)
    add_printed_glyphs(img, 350, 205, 10, gap=5)
    add_scribble(img, 60, 250, scale=1.0, thickness=3)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 120 <= bb.x + bb.width
    dbg = build_signature_debug(img, *ROI)
    assert "peu compatible" in dbg.roi_failure_reason


def test_h_leftward_continuation_rescues_left_stroke():
    # Crop integrity: core blob + tall detached LEFT stroke. The ink gap
    # (~27px) exceeds the V2.3 limit (0.018*800=14.4) but fits the
    # directional-only band (0.04*800=32): fallback must rescue it.
    img = make_image()
    cv2.ellipse(img, (200, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (118, 228), (131, 272), (20, 20, 20), -1)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x <= 118 and bb.x + bb.width >= 240  # left stroke kept
    outcome = run_hybrid_localization(img, *ROI)
    assert outcome.fallback_analysis is not None
    assert outcome.fallback_analysis.directional_accepted_indices != []
    # The accepted component lies strictly LEFT of its anchor (mirror rule).
    fb = outcome.fallback_analysis
    primary = fb.primary_group
    for d in fb.directional_accept_details:
        cand, anchor = primary[d.component_index], primary[d.anchor_index]
        assert cand.x + cand.width <= anchor.x
        assert d.v_gap == 0


def test_h2_roi_path_still_rejects_left_stroke():
    # V2.4 contract: the ROI directional pass NEVER accepts left components.
    img = make_image()
    cv2.ellipse(img, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (470, 228), (483, 272), (20, 20, 20), -1)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    # ROI-local coords: stroke spans x 150..164, core starts at 180.
    assert result.signature_bbox.x >= 170  # left stroke excluded, as in V2.4
    dbg = build_signature_debug(img, *ROI)
    assert dbg.directional_accepted_indices == []


def test_h3_leftward_mirror_rejections():
    # Mirror of the rightward guards: short / misaligned / far left clusters
    # stay rejected even in fallback.
    img = make_image()
    cv2.ellipse(img, (200, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (118, 240), (131, 258), (20, 20, 20), -1)  # too short
    cv2.rectangle(img, (118, 100), (131, 140), (20, 20, 20), -1)  # no overlap
    cv2.rectangle(img, (40, 228), (53, 272), (20, 20, 20), -1)    # too far
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    assert bb.x >= 155  # none of the left clusters absorbed
    outcome = run_hybrid_localization(img, *ROI)
    assert outcome.fallback_analysis is not None
    assert outcome.fallback_analysis.directional_accepted_indices == []


def test_i_no_signature_document_fails_safely():
    # §7: rich business document, NO handwriting. ROI holds a single form
    # rule (rejected), fallback filters graphics/MICR/text, final gate
    # rejects the text winner: clean failure, V5-A never involved.
    try:
        extract_signature(make_no_signature_doc(), *ROI)
    except SignatureExtractionError as exc:
        assert "Aucune signature" in str(exc)
    else:
        raise AssertionError("signless document must raise SignatureExtractionError")
    dbg = build_signature_debug(make_no_signature_doc(), *ROI)
    assert dbg.signature_png_bytes is None
    assert dbg.signature_bbox is None
    assert dbg.crop_completeness is None
    assert dbg.fallback_executed is True
    assert "textuel" in dbg.roi_failure_reason or "trait incompatible" in dbg.roi_failure_reason
    # Explainability table: rows exist, none selected (final gate écarté).
    assert len(dbg.fallback_candidates) >= 1
    assert not any(c.selected for c in dbg.fallback_candidates)
    assert any("contrôle final" in c.status or "Rejeté" in c.status
               for c in dbg.fallback_candidates)


def test_j_nexora_style_signature_selected_with_integrity():
    # §8: table text in ROI + scribble with detached LEFT stroke elsewhere.
    # ROI rejects text; fallback selects the scribble; the final crop keeps
    # the COMPLETE scribble ink (left stroke included).
    img = make_text_table_doc()
    # Detached left stroke: ink gap in the directional-only band, tall,
    # vertically overlapping the scribble core.
    cv2.rectangle(img, (8, 262), (20, 306), (20, 20, 20), -1)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_GLOBAL_FALLBACK
    bb = result.signature_bbox
    # Crop integrity: every scribble ink pixel (core + left stroke) inside.
    probe = np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)
    add_scribble(probe, 40, 240, scale=1.5, thickness=5)
    cv2.rectangle(probe, (8, 262), (20, 306), (20, 20, 20), -1)
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    assert bb.x <= int(xs.min()) and int(xs.max()) <= bb.x + bb.width
    assert bb.y <= int(ys.min()) and int(ys.max()) <= bb.y + bb.height
    # Table header NOT selected: bbox stays left of the ROI x-range.
    assert bb.x + bb.width < ROI_RECT.x
    dbg = build_signature_debug(img, *ROI)
    assert "textuel" in dbg.roi_failure_reason
    assert dbg.signature_png_bytes is not None
    assert dbg.crop_completeness is not None
    assert dbg.crop_completeness.expansion_ratio <= 2.5


def test_k_fallback_candidate_table_matches_algorithm():
    # §2: the table exposes the ACTUAL ranking inputs: every pre-filter
    # group, true scores, truthful statuses, selected highlighted.
    img = make_image()
    add_solid_banner(img)
    add_blob(img, 150, 250)
    dbg = build_signature_debug(img, *ROI)
    assert dbg.fallback_executed is True
    rows = dbg.fallback_candidates
    assert len(rows) == 2
    assert [r.index for r in rows] == [0, 1]  # pre-filter order, stable
    banner, blob = rows
    assert "bloc graphique" in banner.status
    assert blob.status == "Sélectionné"
    assert blob.selected is True
    assert banner.selected is False
    # Selected row carries the exact ranking score surfaced by extraction.
    result = extract_signature(img, *ROI)
    assert blob.score == result.fallback_selected_score
    # Deterministic across runs.
    dbg2 = build_signature_debug(img, *ROI)
    assert [(r.index, r.score, r.status) for r in dbg2.fallback_candidates] == [
        (r.index, r.score, r.status) for r in rows
    ]


def test_l_final_gate_rejects_text_row_winner():
    # §6: headline-size glyph row outside ROI survives the graphic guards
    # (density below the solid-block limit, width below full-bleed) and the
    # V2.3 height gate (tall glyphs), but the final credibility check
    # rejects the uniform-glyph winner: clean failure, Écarté row in the
    # table.
    img = make_image()
    add_printed_glyphs(img, 100, 60, 8, gw=20, gh=32, gap=8)
    try:
        extract_signature(img, *ROI)
    except SignatureExtractionError as exc:
        assert "Aucune signature" in str(exc)
    else:
        raise AssertionError("glyph-row-only document must raise")
    dbg = build_signature_debug(img, *ROI)
    assert dbg.signature_png_bytes is None
    assert dbg.fallback_executed is True
    ecarte = [c for c in dbg.fallback_candidates if "contrôle final" in c.status]
    assert len(ecarte) == 1
    assert ecarte[0].selected is False
    assert not any(c.selected for c in dbg.fallback_candidates)


def add_left_flourish(image, pts, thickness=3):
    """Thin detached leading stroke proxy (never a solid rectangle: real
    flourishes are thin ink with whitespace, unlike graphic blocks)."""
    cv2.polylines(image, [np.array(pts, np.int32)], False, (20, 20, 20),
                  thickness, cv2.LINE_AA)


def test_m_roi_left_flourish_recovered():
    # A: ROI signature with detached LEFT flourish. Ink gap (~10px) exceeds
    # the V2.3 ROI gate (8.35px) so refinement strands it, but the recovery
    # band (0.20 x union width) re-attaches it. ROI mode is kept.
    img = make_image()
    cv2.ellipse(img, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    add_left_flourish(img, [[488, 235], [482, 250], [486, 265]])
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    bb = result.signature_bbox  # ROI-local coordinates
    assert bb.x <= 162 and bb.x + bb.width >= 261  # flourish + core covered
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None
    assert cc.recovered_component_count == 1
    assert cc.recovered_left == 1
    assert cc.recovered_right == 0
    assert cc.iterations >= 1
    assert cc.expansion_ratio > 1.0
    assert "étendu" in cc.status
    assert len(cc.recovered_strokes) == 1
    assert cc.recovered_strokes[0].direction == "gauche"


def test_n_roi_right_flourish_recovered():
    # B: big ROI core + short RIGHT flourish. The flourish is too short for
    # the V2.4 directional pass yet tall enough (vs the union) for recovery.
    img = make_image()
    cv2.ellipse(img, (500, 250), (70, 30), 8, 0, 360, (20, 20, 20), -1)
    add_left_flourish(img, [[592, 238], [607, 250], [592, 262]])
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None and cc.recovered_component_count >= 1
    assert cc.recovered_right >= 1
    bb = result.signature_bbox
    assert bb.x + bb.width >= 285  # flourish (ROI-local ~270..290) covered


def test_o_both_sides_bounded_expansion():
    # C: detached strokes on BOTH sides recovered, expansion strictly bounded
    # and deterministic.
    img = make_image()
    cv2.ellipse(img, (500, 250), (70, 30), 8, 0, 360, (20, 20, 20), -1)
    add_left_flourish(img, [[408, 235], [400, 250], [406, 265]])
    add_left_flourish(img, [[592, 238], [607, 250], [592, 262]])
    for _ in range(2):
        result = extract_signature(img, *ROI)
        outcome = run_hybrid_localization(img, *ROI)
        cc = outcome.roi_analysis.crop_completeness
        assert cc is not None
        assert cc.recovered_left >= 1 and cc.recovered_right >= 1
        assert cc.iterations <= 4
        assert cc.expansion_ratio <= 2.5
        bb = result.signature_bbox
        assert bb.x <= 90 and bb.x + bb.width >= 285
    assert result.localization_mode == LOCALIZATION_MODE_ROI


def test_p_printed_text_next_to_signature_not_absorbed():
    # D: body-text glyphs adjacent to the signature stay outside: too short
    # vs the union to qualify as a side stroke.
    img = make_image()
    cv2.ellipse(img, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    add_printed_glyphs(img, 595, 244, 8, gw=7, gh=13, gap=2)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    bb = result.signature_bbox  # ROI-local; glyphs span x 275..343
    assert bb.x + bb.width <= 270
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None and cc.recovered_component_count == 0


def test_q_rule_below_signature_rejected():
    # E: long horizontal rule right below the signature: too wide for the
    # union and a graphic-guard hit — never attached.
    img = make_image()
    cv2.ellipse(img, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (400, 300), (700, 308), (20, 20, 20), -1)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None and cc.recovered_component_count == 0
    assert "aucun trait voisin" in cc.status


def test_r_table_border_not_attached():
    # F: tall vertical table border beside the signature (own group, lower
    # score): oversized vs the union, so never attached.
    img = make_image()
    cv2.ellipse(img, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)
    cv2.rectangle(img, (640, 200), (648, 300), (20, 20, 20), -1)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None and cc.recovered_component_count == 0


def test_s_complete_signature_unchanged():
    # G: already-complete ROI signature: zero recovery, V2.4 numbers locked.
    img = make_image()
    add_blob(img, 540, 250)
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    bb = result.signature_bbox
    assert (bb.x, bb.y, bb.width, bb.height) == (180, 84, 81, 49)
    assert result.extraction_quality == 0.7875
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None
    assert cc.recovered_component_count == 0
    assert cc.iterations == 0
    assert cc.expansion_ratio == 1.0
    assert cc.initial_bbox == cc.final_bbox == bb
    assert cc.status == "Recadrage complet — aucun trait voisin crédible à rattacher."


def make_invoice_doc():
    """CHQ-0019-style fixture: printed invoice/table content + handwritten
    signature inside the ROI with a detached LEADING stroke (ink gap beyond
    the V2.3 gate, inside the recovery band)."""
    img = make_image()
    add_printed_glyphs(img, 20, 30, 16)                       # invoice header
    add_printed_glyphs(img, 20, 60, 14)
    add_printed_glyphs(img, 350, 150, 12, gh=14)              # table labels
    cv2.line(img, (340, 172), (760, 172), (20, 20, 20), 1)    # table rule
    add_printed_glyphs(img, 350, 185, 9, gap=3)
    add_scribble(img, 480, 230, scale=1.2, thickness=4)        # signature core
    add_left_flourish(img, [[448, 248], [440, 265], [446, 282]], thickness=4)
    return img


def test_j2_invoice_leading_stroke_recovered():
    # J: invoice doc, ROI signature with detached leading stroke: correct
    # localization, COMPLETE final crop, no printed labels absorbed.
    img = make_invoice_doc()
    result = extract_signature(img, *ROI)
    assert result.localization_mode == LOCALIZATION_MODE_ROI
    bb = result.signature_bbox  # ROI-local
    probe = np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)
    add_scribble(probe, 480, 230, scale=1.2, thickness=4)
    add_left_flourish(probe, [[448, 248], [440, 265], [446, 282]], thickness=4)
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    margin_x = int(round(bb.width * 0.08))
    margin_y = int(round(bb.height * 0.08))
    crop_x0 = max(0, bb.x - margin_x)
    crop_y0 = max(0, bb.y - margin_y)
    crop_x1 = min(ROI_RECT.width, bb.x + bb.width + margin_x)
    crop_y1 = min(ROI_RECT.height, bb.y + bb.height + margin_y)
    assert crop_x0 <= int(xs.min()) - ROI_RECT.x
    assert int(xs.max()) - ROI_RECT.x < crop_x1
    assert crop_y0 <= int(ys.min()) - ROI_RECT.y
    assert int(ys.max()) - ROI_RECT.y < crop_y1
    crop_png = cv2.imdecode(
        np.frombuffer(result.signature_png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert crop_png.shape[1] == crop_x1 - crop_x0
    assert crop_png.shape[0] == crop_y1 - crop_y0
    # Printed table labels stay outside the actual final crop, including margin.
    table_probe = np.full((IMAGE_H, IMAGE_W, 3), 245, dtype=np.uint8)
    add_printed_glyphs(table_probe, 350, 150, 12, gh=14)
    cv2.line(table_probe, (340, 172), (760, 172), (20, 20, 20), 1)
    add_printed_glyphs(table_probe, 350, 185, 9, gap=3)
    gray = cv2.cvtColor(table_probe, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tys, txs = np.where(binary > 0)
    inside = (
        (crop_x0 <= txs - ROI_RECT.x) & (txs - ROI_RECT.x < crop_x1)
        & (crop_y0 <= tys - ROI_RECT.y) & (tys - ROI_RECT.y < crop_y1)
    )
    assert not bool(np.any(inside))
    outcome = run_hybrid_localization(img, *ROI)
    cc = outcome.roi_analysis.crop_completeness
    assert cc is not None and cc.recovered_left >= 1
    assert cc.initial_bbox != cc.final_bbox == bb
    assert cc.expansion_ratio <= 2.5
    dbg = build_signature_debug(img, *ROI)
    assert dbg.crop_completeness is not None
    assert dbg.crop_completeness.recovered_left >= 1


def test_debug_roi_mode_has_no_fallback_image():
    img = make_image()
    add_blob(img, 540, 250)
    dbg = build_signature_debug(img, *ROI)
    assert dbg.localization_mode == LOCALIZATION_MODE_ROI
    assert dbg.localization_label == "ROI principale"
    assert dbg.fallback_candidates_png_bytes is None
    assert dbg.fallback_candidate_count == 0
    assert dbg.roi_failure_reason == ""


if __name__ == "__main__":
    test_a_roi_signature_unchanged()
    print("PASS A roi behavior unchanged")
    test_b_fallback_selects_outside_roi_signature()
    print("PASS B full-document fallback")
    test_b2_fallback_prefers_signature_over_micr_row()
    print("PASS B2 signature preferred over MICR row")
    test_c_multiple_fallback_candidates_deterministic_best()
    print("PASS C deterministic best candidate")
    test_d_no_signature_anywhere_fails_safely()
    print("PASS D safe failure")
    test_e_extraction_never_touches_v5a()
    print("PASS E V5-A untouched")
    test_debug_roi_mode_has_no_fallback_image()
    print("PASS debug ROI mode (no fallback image)")
    test_f_banner_rejected_signature_selected()
    print("PASS F banner rejected, signature selected")
    test_f2_non_full_bleed_solid_block_rejected()
    print("PASS F2 solid block rejected by density gate")
    test_f3_thin_rule_rejected()
    print("PASS F3 thin rule rejected")
    test_f4_banner_only_fails_safely()
    print("PASS F4 banner-only safe failure")
    test_g_roi_text_slab_rejected_fallback_selects_signature()
    print("PASS G ROI text slab rejected, fallback selects signature")
    test_g2_compact_signature_in_roi_accepted()
    print("PASS G2 compact ROI signature accepted")
    test_g3_lone_glyph_rejected_fallback_selects_signature()
    print("PASS G3 lone glyph rejected, fallback selects signature")
    test_h_leftward_continuation_rescues_left_stroke()
    print("PASS H leftward continuation rescues left stroke")
    test_h2_roi_path_still_rejects_left_stroke()
    print("PASS H2 ROI path still rejects left (V2.4 contract)")
    test_h3_leftward_mirror_rejections()
    print("PASS H3 leftward mirror rejections")
    test_i_no_signature_document_fails_safely()
    print("PASS I no-signature document clean failure")
    test_j_nexora_style_signature_selected_with_integrity()
    print("PASS J Nexora-style signature with crop integrity")
    test_k_fallback_candidate_table_matches_algorithm()
    print("PASS K candidate table matches algorithm")
    test_l_final_gate_rejects_text_row_winner()
    print("PASS L final gate rejects text-row winner")
    test_m_roi_left_flourish_recovered()
    print("PASS M ROI left flourish recovered")
    test_n_roi_right_flourish_recovered()
    print("PASS N ROI right flourish recovered")
    test_o_both_sides_bounded_expansion()
    print("PASS O both sides bounded expansion")
    test_p_printed_text_next_to_signature_not_absorbed()
    print("PASS P printed text not absorbed")
    test_q_rule_below_signature_rejected()
    print("PASS Q rule below signature rejected")
    test_r_table_border_not_attached()
    print("PASS R table border not attached")
    test_s_complete_signature_unchanged()
    print("PASS S complete signature unchanged")
    test_j2_invoice_leading_stroke_recovered()
    print("PASS J2 invoice leading stroke recovered")
    print("All hybrid localization tests passed.")
