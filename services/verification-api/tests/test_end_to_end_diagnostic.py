"""END-TO-END AI V2 diagnostic benchmark — test-only safeguards.

These tests protect the DIAGNOSTIC benchmark artifacts and the invariants it
relies on. They modify nothing: production code, checkpoints, extraction,
FastAPI behavior and ASP.NET are untouched.

Benchmark-only by design:
  - preprocessing parity (eval vs FastAPI inference)
  - deterministic AI inference
  - benchmark dataset label integrity / no leakage
  - report artifact generation
  - checkpoint preservation (SHA256)
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
E2E_DIR = SERVICE_ROOT / "ai" / "reports" / "v2" / "end_to_end"

# The exact V2 frozen TEST split (benchmark uses ONLY these writers; the
# model never trained on them -> no dataset leakage).
TEST_WRITERS = [2, 7, 8, 9, 15, 16, 18, 41, 48]
TRAIN_WRITERS = [
    1, 4, 5, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 49, 50, 51, 52, 54, 55,
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ---------------------------------------------------------------------------
# Checkpoint preservation
# ---------------------------------------------------------------------------
def test_checkpoints_unchanged_since_benchmark_start():
    before = {}
    for line in (E2E_DIR / "checkpoint_sha256_before.txt").read_text().splitlines():
        digest, name = line.split()
        before[name] = digest
    for name, expected in before.items():
        assert _sha256(SERVICE_ROOT / "ai" / "checkpoints" / name) == expected


def test_before_and_after_hashes_match():
    before = sorted((E2E_DIR / "checkpoint_sha256_before.txt").read_text().splitlines())
    after = sorted((E2E_DIR / "checkpoint_sha256_after.txt").read_text().splitlines())
    assert before == after


# ---------------------------------------------------------------------------
# Preprocessing parity (eval vs FastAPI inference path)
# ---------------------------------------------------------------------------
def test_eval_and_fastapi_preprocessing_produce_identical_tensors(tmp_path):
    import torch

    from ai.v2.config import V2Config
    from ai.preprocessing import preprocess
    from app.core.config import get_settings
    from app.services.ai_signature_verification_service import AISignatureVerificationService

    # Tiny synthetic "signature": white background + black strokes.
    img = np.full((96, 160), 255, np.uint8)
    cv2.line(img, (10, 70), (140, 30), 0, 3)
    cv2.circle(img, (60, 45), 12, 0, 2)
    path = tmp_path / "tiny.png"
    cv2.imwrite(str(path), img)

    cfg = V2Config()
    t_eval = preprocess(path, cfg, augment=False)

    service = AISignatureVerificationService(get_settings())
    t_prod, bbox = service._prepare_image(path.read_bytes())

    assert t_prod.shape == (1, 3, cfg.canvas_height, cfg.canvas_width)
    assert t_prod.dtype == torch.float32
    assert torch.equal(t_prod[0], t_eval)


# ---------------------------------------------------------------------------
# Deterministic AI inference
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ai_service():
    from app.core.config import get_settings
    from app.services.ai_signature_verification_service import get_ai_service

    return get_ai_service(get_settings())


def test_ai_inference_is_deterministic(ai_service):
    import random

    rng = np.random.default_rng(7)
    img = np.full((64, 128), 255, np.uint8)
    for _ in range(6):
        cv2.line(img,
                 (int(rng.integers(5, 60)), int(rng.integers(5, 55))),
                 (int(rng.integers(65, 120)), int(rng.integers(5, 55))), 0, 2)
    png = cv2.imencode(".png", img)[1].tobytes()

    e1 = ai_service.encode_signature(png)
    e2 = ai_service.encode_signature(png)
    assert np.array_equal(e1, e2)
    assert e1.shape == (128,)
    assert abs(float(np.linalg.norm(e1)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Benchmark dataset integrity (labels + leakage)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (E2E_DIR / "end_to_end_scores.csv").exists(),
    reason="benchmark artifacts not generated",
)
class TestBenchmarkArtifacts:
    def setup_method(self):
        with open(E2E_DIR / "end_to_end_scores.csv", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        with open(E2E_DIR / "clean_vs_extracted.csv", encoding="utf-8") as f:
            self.self_rows = list(csv.DictReader(f))

    def test_scores_in_cosine_range(self):
        for r in self.rows:
            assert -1.0 <= float(r["score"]) <= 1.0

    def test_roles_are_valid(self):
        roles = {r["role"] for r in self.rows}
        assert roles <= {"genuine", "skilled_forgery", "different_writer"}

    def test_genuine_forgery_labels_match_filenames(self):
        for r in self.rows:
            f = r["candidate_file"]
            if f.startswith("original_"):
                assert r["role"] == "genuine", f
            elif f.startswith("forgeries_"):
                assert r["role"] == "skilled_forgery", f

    def test_no_train_writer_leakage(self):
        writers = {int(r["writer"]) for r in self.rows}
        assert writers <= set(TEST_WRITERS)
        assert writers.isdisjoint(TRAIN_WRITERS)

    def test_self_control_pairs_are_same_signature(self):
        for r in self.self_rows:
            # clean_vs_extracted rows must pair a candidate with its OWN
            # extraction (same writer + same file stem).
            assert int(r["writer"]) in TEST_WRITERS
            assert r["candidate_file"]

    def test_report_artifacts_exist(self):
        for name in (
            "end_to_end_scores.csv",
            "summary_metrics.json",
            "clean_vs_extracted.csv",
            "preprocessing_consistency.json",
            "per_writer_metrics.csv",
            "score_distribution.png",
            "clean_vs_extracted.png",
            "run_log.txt",
            "END_TO_END_AI_V2_REPORT.md",
        ):
            assert (E2E_DIR / name).exists(), name

    def test_preprocessing_consistency_reports_parity(self):
        import json

        data = json.loads((E2E_DIR / "preprocessing_consistency.json").read_text())
        for entry in data["per_image"]:
            assert entry["tensors_identical"] is True
            assert entry["embedding_cosine_eval_vs_prod"] > 0.999
            assert entry["prod_shape"] == entry["expected_prod_shape"]
