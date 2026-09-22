"""Pre-validation of DEMO-V5-CONFORME-001 with Extraction V2.4 READ-ONLY."""

import sys
from pathlib import Path
import cv2
import numpy as np

SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))

from app.services.signature_extraction_service import extract_signature
from app.services.signature_extraction_service import EXTRACTION_PIPELINE_VERSION

CHEQUE_PATH = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-CONFORME-001.png")
# Paste rect from GOOD generation (seed DEMO-V5-CONFORME-001-attempt-80): (1099, 315, 406, 144)
PASTE_RECT = (1099, 315, 406, 144)

def rect_iou(a, b) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2-ix1)
    ih = max(0, iy2-iy1)
    inter = iw*ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union>0 else 0.0

def bbox_of(rect, roi=None):
    x, y = int(rect.x), int(rect.y)
    if roi is not None:
        x += int(roi.x)
        y += int(roi.y)
    return (x, y, int(rect.width), int(rect.height))

def main():
    print(f"Pipeline version constant: {EXTRACTION_PIPELINE_VERSION}")
    assert CHEQUE_PATH.exists(), f"Missing {CHEQUE_PATH}"
    img = cv2.imread(str(CHEQUE_PATH))
    assert img is not None, "imread failed"
    print(f"Image opened: {img.shape[1]}x{img.shape[0]}, pasted rect {PASTE_RECT}")
    # Verify paste inside validated region
    rx0, rx1 = 660, 1520
    ry0, ry1 = 290, 590
    px, py, pw, ph = PASTE_RECT
    assert rx0 <= px and px+pw <= rx1, "paste x outside region"
    assert ry0 <= py and py+ph <= ry1, "paste y outside region"
    print(f"Signature placement region check: PASS (inside {rx0}-{rx1} x {ry0}-{ry1})")

    # Extraction V2.4 READ-ONLY
    res = extract_signature(img, 0.40, 0.40, 0.98, 0.98)
    print(f"\nExtraction successful: {res is not None}")
    print(f"Pipeline version: {EXTRACTION_PIPELINE_VERSION}")
    print(f"Original: {res.original_width}x{res.original_height}")
    print(f"Candidate ROI: x={res.candidate_roi.x} y={res.candidate_roi.y} w={res.candidate_roi.width} h={res.candidate_roi.height}")
    print(f"Signature bbox (ROI-local): x={res.signature_bbox.x} y={res.signature_bbox.y} w={res.signature_bbox.width} h={res.signature_bbox.height}")
    full_bbox = bbox_of(res.signature_bbox, res.candidate_roi)
    print(f"Signature bbox (full-image): {full_bbox}")
    print(f"Paste rect (full): {PASTE_RECT}")
    iou = rect_iou(full_bbox, PASTE_RECT)
    print(f"IoU vs paste_rect: {iou:.4f}")
    print(f"Extraction quality: {res.extraction_quality:.4f}")
    # Decode crop to check dimensions
    crop = cv2.imdecode(np.frombuffer(res.signature_png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert crop is not None
    print(f"Extracted crop dimensions: {crop.shape[1]}x{crop.shape[0]} (WxH), channels={crop.shape[2] if crop.ndim==3 else 1}")
    # Quality heuristics
    # Completeness / contamination via detailed analysis? Call _analyze_ink for diagnostics
    from app.services.signature_extraction_service import _analyze_ink, _compute_roi
    import cv2 as cv2b
    roi = _compute_roi(img.shape[1], img.shape[0], 0.40, 0.40, 0.98, 0.98)
    roi_crop = img[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
    gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    analysis = _analyze_ink(gray, roi, micr_zone_ratio=0.10, min_component_area_ratio=0.0005, max_component_area_ratio=0.60,
                            min_component_width_ratio=0.01, min_component_height_ratio=0.01,
                            micr_max_height_ratio=0.06, micr_min_aspect_ratio=6.0, micr_min_width_ratio=0.03,
                            component_merge_distance_ratio=0.10, group_min_components=1, morph_kernel_size=3,
                            core_ink_distance_ratio=0.018, core_max_h_gap_ratio=0.15, core_max_v_gap_ratio=0.15,
                            core_min_height_ratio=0.08, core_directional_h_gap_ratio=0.08,
                            core_directional_ink_distance_ratio=0.04, completeness_min_factor=0.5)
    print(f"\nAnalysis diagnostics:")
    print(f"  micr_band_top (ROI-local y): {analysis.micr_band_top} -> full y={roi.y+analysis.micr_band_top}")
    print(f"  all_components: {len(analysis.all_components)}")
    print(f"  kept: {len(analysis.kept_components)} rejected: {len(analysis.rejected_components)} micr_rejected: {len(analysis.micr_rejected_components)}")
    print(f"  groups: {len(analysis.groups)} selected_group_index: {analysis.selected_group_index}")
    print(f"  refined_components: {len(analysis.refined_components)} discarded_from_selected: {len(analysis.discarded_from_selected_group_indices)}")
    print(f"  completeness_score: {analysis.completeness_score:.4f} ({analysis.completeness_refined_ink}/{analysis.completeness_reference_ink})")
    print(f"  quality_base: {analysis.quality_base:.4f} factor: {analysis.quality_completeness_factor:.4f} final: {analysis.quality:.4f}")
    print(f"  dominant core idx: {analysis.dominant_core_index_in_group} ink:{analysis.dominant_component_ink} score:{analysis.dominant_component_score:.5f}")
    print(f"  selection_reason: {analysis.selection_reason[:160]}...")
    print(f"  refinement_reason: {analysis.refinement_reason[:220]}...")

    # Checks
    # Must capture paste_rect significantly
    # For this controlled demo, IoU >0.30 is success threshold used in end_to_end benchmark
    # But for demo we expect high IoU since no competing strong strokes
    iou_ok = iou > 0.50
    print(f"\nCompleteness checks:")
    print(f"  IoU >0.50 : {iou_ok} (actual {iou:.4f})")
    # Check crop not empty, quality not zero
    quality_ok = res.extraction_quality > 0.35
    print(f"  Quality >0.35 : {quality_ok} ({res.extraction_quality})")
    completeness_ok = analysis.completeness_score >= 0.95
    print(f"  Completeness >=0.95 : {completeness_ok} ({analysis.completeness_score})")
    contamination_ok = len(analysis.discarded_from_selected_group_indices) <= 2  # allow small discard but not massive
    print(f"  Discarded from selected <=2 : {contamination_ok} ({len(analysis.discarded_from_selected_group_indices)})")
    candidate_captured = iou_ok and quality_ok and completeness_ok
    print(f"  Candidate correctly captured: {'YES' if candidate_captured else 'NO'}")

    # Save extracted crop for visual inspection
    out_crop = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-CONFORME-001_extracted.png")
    out_crop.write_bytes(res.signature_png_bytes)
    print(f"\nExtracted crop saved to: {out_crop} ({out_crop.stat().st_size} bytes)")

    # Save diagnostic overlay
    # Draw bbox on cheque for visual check
    vis = img.copy()
    # paste rect green
    cv2.rectangle(vis, (PASTE_RECT[0], PASTE_RECT[1]), (PASTE_RECT[0]+PASTE_RECT[2], PASTE_RECT[1]+PASTE_RECT[3]), (0,255,0), 2)
    cv2.putText(vis, "paste", (PASTE_RECT[0], PASTE_RECT[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1, cv2.LINE_AA)
    # extracted bbox red
    cv2.rectangle(vis, (full_bbox[0], full_bbox[1]), (full_bbox[0]+full_bbox[2], full_bbox[1]+full_bbox[3]), (0,0,255), 2)
    cv2.putText(vis, f"extracted IoU={iou:.2f}", (full_bbox[0], full_bbox[1]+full_bbox[3]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1, cv2.LINE_AA)
    # ROI blue
    cv2.rectangle(vis, (roi.x, roi.y), (roi.x+roi.width, roi.y+roi.height), (255,0,0), 1)
    diag_path = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-CONFORME-001_diagnostic.png")
    cv2.imwrite(str(diag_path), vis)
    print(f"Diagnostic overlay saved to: {diag_path}")

    print("\n--- FINAL PRECHECK ---")
    print(f"Extraction successful: YES")
    print(f"Pipeline version: {EXTRACTION_PIPELINE_VERSION}")
    print(f"Extracted dimensions: {crop.shape[1]}x{crop.shape[0]}")
    print(f"Quality: {res.extraction_quality:.4f}")
    print(f"Completeness: {analysis.completeness_score:.4f}")
    print(f"Contamination: {'LOW' if contamination_ok else 'CHECK'}")
    print(f"Candidate correctly captured: {'YES' if candidate_captured else 'NO'}")

if __name__ == "__main__":
    main()
