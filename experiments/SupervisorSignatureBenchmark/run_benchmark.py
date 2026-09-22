"""Isolated signature-consistency benchmark (EXPERIMENT ONLY).

Uses the 7 supervisor front-side cheque images and measures how mutually
similar their handwritten signatures are according to the CURRENT production
engine:

  - extraction: current hybrid OpenCV pipeline
    (app.services.signature_extraction_service.extract_signature),
    called with the exact same settings wiring as the production
    POST /api/signatures/extract route;
  - comparison: current frozen V5-A model, as-is
    (app.services.ai_signature_verification_service.get_ai_service),
    raw cosine similarity between L2-normalized 128-D embeddings.

READ-ONLY reuse: this script never writes to production code, the database,
reference signatures, thresholds, or checkpoints. It only reads the 7 JPEG
inputs and writes artifacts under experiments/SupervisorSignatureBenchmark/.

No authenticity classification is produced here (no genuine/forged/fraud
language): only mutual-similarity statistics.

Usage (from the service root so `app.*` / `ai.*` import):
    .venv\\Scripts\\python.exe ..\\..\\experiments\\SupervisorSignatureBenchmark\\run_benchmark.py
Run with CWD = services/verification-api.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parents[2] / "services" / "verification-api"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.ai_signature_verification_service import get_ai_service  # noqa: E402
from app.services.image_processing_service import decode_image  # noqa: E402
from app.services.signature_extraction_service import (  # noqa: E402
    SignatureExtractionError,
    extract_signature,
)

HERE = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\Dev\fwchq")
EXTRACTED_DIR = HERE / "extracted"
RESULTS_DIR = HERE / "results"

# Policy thresholds shown ONLY as contextual reference (cheque-to-reference
# policy, not validated for signature-to-signature clustering). Never applied
# as decisions in this experiment.
THRESHOLD_L = 0.6585
THRESHOLD_U = 0.9150

FILES = [
    ("20260226141441_0001+.jpg", "sig_0001"),
    ("20260226141441_0003+.jpg", "sig_0003"),
    ("20260226141441_0005+.jpg", "sig_0005"),
    ("20260226141441_0007+.jpg", "sig_0007"),
    ("20260226141441_0009+.jpg", "sig_0009"),
    ("20260226141441_0011+.jpg", "sig_0011"),
    ("20260226141441_0013+.jpg", "sig_0013"),
]


def rect_dict(r) -> dict:
    return {"x": int(r.x), "y": int(r.y), "width": int(r.width), "height": int(r.height)}


def main() -> int:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    print("=== Phase 1: hybrid OpenCV extraction (current production pipeline) ===")
    records: list[dict] = []
    for fname, sid in FILES:
        path = DATASET_DIR / fname
        rec: dict = {"sig_id": sid, "file": fname}
        try:
            raw = path.read_bytes()
            image = decode_image(raw)  # same decode as production route
            # Same settings wiring as POST /api/signatures/extract route.
            result = extract_signature(
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
            crop_path = EXTRACTED_DIR / f"{sid}.png"
            crop_path.write_bytes(result.signature_png_bytes)
            rec.update(
                extraction_success=True,
                extraction_quality=round(float(result.extraction_quality), 6),
                localization_mode=result.localization_mode,
                bbox=rect_dict(result.signature_bbox),
                candidate_roi=rect_dict(result.candidate_roi),
                crop_file=str(crop_path),
                error="",
            )
            print(
                f"  {sid} ({fname}): OK quality={rec['extraction_quality']:.4f} "
                f"mode={rec['localization_mode']} bbox={rec['bbox']} -> {crop_path.name}"
            )
        except SignatureExtractionError as exc:
            rec.update(
                extraction_success=False,
                extraction_quality=None,
                localization_mode="",
                bbox=None,
                candidate_roi=None,
                crop_file="",
                error=f"SignatureExtractionError: {exc}",
            )
            print(f"  {sid} ({fname}): EXTRACTION FAILED: {exc}")
        except Exception as exc:  # unexpected: record, do not fabricate a crop
            rec.update(
                extraction_success=False,
                extraction_quality=None,
                localization_mode="",
                bbox=None,
                candidate_roi=None,
                crop_file="",
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"  {sid} ({fname}): UNEXPECTED FAILURE: {type(exc).__name__}: {exc}")
        records.append(rec)

    ok = [r for r in records if r["extraction_success"]]
    failed = [r for r in records if not r["extraction_success"]]
    print(f"Extraction: {len(ok)}/7 succeeded, {len(failed)}/7 failed.")
    if len(ok) < 2:
        print("Fewer than 2 successful extractions: AI comparison impossible. STOP.")
        (RESULTS_DIR / "summary.json").write_text(
            json.dumps({"extraction": records, "error": "insufficient extractions"},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 2

    print()
    print("=== Phase 2: V5-A embeddings (frozen checkpoint, as-is) + pairwise cosine ===")
    service = get_ai_service()
    print(f"Model: {service.model_name} version={service.version} "
          f"dim={service.embedding_dim} device={service.device_str} "
          f"checkpoint={service.checkpoint_path.name}")

    embeddings: dict[str, np.ndarray] = {}
    for r in ok:
        png = Path(r["crop_file"]).read_bytes()
        embeddings[r["sig_id"]] = service.encode_signature(png)

    ids = [r["sig_id"] for r in ok]
    sim: dict[tuple[str, str], float] = {}
    for a, b in combinations(ids, 2):
        s = float(np.clip(float(np.dot(embeddings[a], embeddings[b])), -1.0, 1.0))
        sim[(a, b)] = round(s, 6)

    # pairwise_scores.csv
    with open(RESULTS_DIR / "pairwise_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sig_a", "sig_b", "score"])
        for (a, b), s in sorted(sim.items()):
            w.writerow([a, b, f"{s:.6f}"])

    # matrix.csv (full symmetric 7x7 over successful ids; diagonal 1.0)
    with open(RESULTS_DIR / "matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + ids)
        for a in ids:
            row = [a]
            for b in ids:
                if a == b:
                    row.append("1.000000")
                else:
                    key = (a, b) if (a, b) in sim else (b, a)
                    row.append(f"{sim[key]:.6f}")
            w.writerow(row)

    scores = list(sim.values())
    per_sig: dict[str, dict] = {}
    for sid in ids:
        others = [s for (a, b), s in sim.items() if a == sid or b == sid]
        per_sig[sid] = {
            "mean_similarity": round(float(np.mean(others)), 6),
            "min_similarity": round(float(np.min(others)), 6),
            "max_similarity": round(float(np.max(others)), 6),
        }

    strongest = max(sim.items(), key=lambda kv: kv[1])
    weakest = min(sim.items(), key=lambda kv: kv[1])
    best_sig = max(ids, key=lambda s: per_sig[s]["mean_similarity"])
    worst_sig = min(ids, key=lambda s: per_sig[s]["mean_similarity"])

    summary = {
        "signatures": [
            {
                "sig_id": r["sig_id"],
                "file": r["file"],
                "extraction_success": r["extraction_success"],
                "extraction_quality": r["extraction_quality"],
                "localization_mode": r["localization_mode"],
                "bbox": r["bbox"],
                "mean_similarity": per_sig[r["sig_id"]]["mean_similarity"] if r["extraction_success"] else None,
                "min_similarity": per_sig[r["sig_id"]]["min_similarity"] if r["extraction_success"] else None,
                "max_similarity": per_sig[r["sig_id"]]["max_similarity"] if r["extraction_success"] else None,
                "error": r["error"],
            }
            for r in records
        ],
        "global": {
            "n_compared": len(ids),
            "n_pairs": len(scores),
            "mean_pairwise_similarity": round(float(np.mean(scores)), 6),
            "median_pairwise_similarity": round(float(statistics.median(scores)), 6),
            "min_pairwise_similarity": round(float(np.min(scores)), 6),
            "max_pairwise_similarity": round(float(np.max(scores)), 6),
        },
        "strongest_pair": {"a": strongest[0][0], "b": strongest[0][1], "score": strongest[1]},
        "weakest_pair": {"a": weakest[0][0], "b": weakest[0][1], "score": weakest[1]},
        "highest_consistency_signature": best_sig,
        "lowest_consistency_signature": worst_sig,
        "threshold_context_only": {"L": THRESHOLD_L, "U": THRESHOLD_U,
            "note": "Cheque-to-reference policy values shown for context only; "
                    "not applied as decisions in this signature-to-signature experiment."},
        "model": {"name": service.model_name, "version": service.version,
                  "embedding_dim": service.embedding_dim, "device": service.device_str,
                  "checkpoint": service.checkpoint_path.name},
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Pairwise scores (21 max):")
    for (a, b), s in sorted(sim.items()):
        print(f"  {a}-{b}: {s:.6f}")
    g = summary["global"]
    print(f"Global: mean={g['mean_pairwise_similarity']:.6f} "
          f"median={g['median_pairwise_similarity']:.6f} "
          f"min={g['min_pairwise_similarity']:.6f} max={g['max_pairwise_similarity']:.6f}")
    print(f"Strongest pair: {strongest[0][0]}-{strongest[0][1]} = {strongest[1]:.6f}")
    print(f"Weakest pair: {weakest[0][0]}-{weakest[0][1]} = {weakest[1]:.6f}")
    print(f"Highest-consistency: {best_sig} "
          f"(mean={per_sig[best_sig]['mean_similarity']:.6f}); "
          f"lowest-consistency: {worst_sig} "
          f"(mean={per_sig[worst_sig]['mean_similarity']:.6f})")
    print(f"Threshold context only (NOT applied): L={THRESHOLD_L} U={THRESHOLD_U}")
    print(f"Artifacts: {EXTRACTED_DIR} + {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
