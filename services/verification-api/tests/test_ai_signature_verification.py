"""Tests for the AI V2 signature-verification integration.

Run from the service root:
    .venv\\Scripts\\python.exe -m pytest tests\\test_ai_signature_verification.py -q

Covers: AI disabled behavior, checkpoint load + validation, CPU inference,
CUDA selection, preprocessing parity (deterministic, no augmentation),
embedding shape/norm, similarity range, pair endpoint, error paths, OpenCV
baseline unchanged, single-load caching, multi-reference service and
aggregation, and the absence of any threshold/conformity decision.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.ai_signature_verification_service import (  # noqa: E402
    AICheckpointError,
    AIDisabledError,
    AISignatureVerificationService,
    EMBEDDING_DIMENSION,
    METHOD,
    MODEL_NAME,
    VERSION,
    get_ai_service,
    reset_ai_service,
)

CEDAR_WRITER_7 = r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR\7"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_signature_bytes(variant: int = 0) -> bytes:
    import cv2

    w, h = 300, 150
    cx, cy = w // 2, h // 2
    img = np.full((h, w), 245, dtype=np.uint8)
    if variant == 0:
        cv2.ellipse(img, (cx, cy), (90, 40), 20, 0, 360, (20, 20, 20), -1)
        cv2.line(img, (cx - 90, cy - 20), (cx - 45, cy + 40), (20, 20, 20), 4)
        cv2.line(img, (cx + 30, cy - 40), (cx + 90, cy + 20), (20, 20, 20), 4)
    else:
        cv2.circle(img, (cx, cy), 55, (20, 20, 20), -1)
        cv2.line(img, (0, cy + 40), (w, cy + 40), (20, 20, 20), 6)
    ok, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def _cedar_bytes(name: str) -> bytes:
    path = os.path.join(CEDAR_WRITER_7, name)
    if not os.path.exists(path):
        pytest.skip(f"CEDAR file not found: {path}")
    with open(path, "rb") as fh:
        return fh.read()


def _write_env(monkeypatch, **overrides) -> None:
    base = {
        "SIGNATURE_AI_ENABLED": "true",
        "SIGNATURE_AI_CHECKPOINT": "ai/checkpoints/metric_resnet18_v2.pt",
        "SIGNATURE_AI_DEVICE": "cpu",
    }
    base.update(overrides)
    for key, value in base.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _cleanup():
    reset_ai_service()
    yield
    reset_ai_service()


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture()
def cpu_settings(monkeypatch):
    _write_env(monkeypatch, SIGNATURE_AI_DEVICE="cpu")
    return get_settings()


# ---------------------------------------------------------------------------
# 1. AI disabled behavior
# ---------------------------------------------------------------------------
def test_ai_disabled_service_raises(monkeypatch):
    _write_env(monkeypatch, SIGNATURE_AI_ENABLED="false")
    with pytest.raises(AIDisabledError):
        get_ai_service()


def test_ai_disabled_endpoint_returns_controlled_unavailable(client, monkeypatch):
    _write_env(monkeypatch, SIGNATURE_AI_ENABLED="false")
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(1), "image/png"),
        },
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["success"] is False
    assert body["similarity_score"] is None


def test_ai_disabled_status_and_health_still_ok(client, monkeypatch):
    _write_env(monkeypatch, SIGNATURE_AI_ENABLED="false")
    status = client.get("/api/signatures/ai-status")
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert status.json()["loaded"] is False
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_app_starts_without_torch_and_opencv_still_works(tmp_path):
    """The app must start when torch is NOT importable (import-safe module),
    the AI endpoint must return a controlled 503 and the OpenCV baseline must
    keep working. Verified in a fresh subprocess that blocks torch."""
    import subprocess
    import textwrap

    cedar_a = os.path.join(CEDAR_WRITER_7, "original_7_5.png")
    cedar_b = os.path.join(CEDAR_WRITER_7, "original_7_6.png")
    if not os.path.exists(cedar_a) or not os.path.exists(cedar_b):
        pytest.skip("CEDAR files not found")
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules["torch"] = None
        from fastapi.testclient import TestClient
        import app.main as m
        c = TestClient(m.create_app())
        s = c.get("/api/signatures/ai-status")
        assert s.status_code == 200, s.text
        assert s.json()["loaded"] is False
        a = open(r"{cedar_a}", "rb").read()
        b = open(r"{cedar_b}", "rb").read()
        ocv = c.post("/api/signatures/compare", files={{
            "extracted_file": ("a.png", a, "image/png"),
            "reference_file": ("b.png", b, "image/png"),
        }})
        assert ocv.status_code == 200, ocv.text
        assert ocv.json()["method"] == "opencv_baseline"
        ai = c.post("/api/signatures/compare-ai", files={{
            "extracted_file": ("a.png", a, "image/png"),
            "reference_file": ("b.png", b, "image/png"),
        }})
        assert ai.status_code == 503, ai.text
        assert ai.json()["success"] is False
        assert ai.json()["device"] == "unavailable"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# 2 + 3 + 4. Checkpoint load / invalid / metadata validation
# ---------------------------------------------------------------------------
def test_checkpoint_loads_successfully(cpu_settings):
    service = get_ai_service(cpu_settings)
    assert service.model is not None
    assert service.device_str == "cpu"
    assert service.model_name == MODEL_NAME
    assert service.version == VERSION


def test_invalid_checkpoint_missing_file(tmp_path, monkeypatch):
    _write_env(monkeypatch, SIGNATURE_AI_CHECKPOINT=str(tmp_path / "nope.pt"))
    with pytest.raises(AICheckpointError, match="introuvable"):
        get_ai_service()


def test_invalid_checkpoint_corrupt_file(tmp_path, monkeypatch):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a torch file")
    _write_env(monkeypatch, SIGNATURE_AI_CHECKPOINT=str(bad))
    with pytest.raises(AICheckpointError, match="charger le checkpoint"):
        get_ai_service()


def test_checkpoint_metadata_validation(tmp_path):
    import torch

    from app.core.config import Settings

    def _payload(**overrides):
        payload = {
            "model_version": "ai_metric_v2",
            "architecture": {"backbone": "resnet18", "embedding_dim": 128},
            "canvas_size": {"width": 256, "height": 128},
            "model_state": {},
        }
        payload.update(overrides)
        return payload

    cases = [
        (_payload(model_version="ai_metric_v3"), "model_version"),
        (_payload(architecture={"backbone": "resnet50", "embedding_dim": 128}), "backbone"),
        (_payload(architecture={"backbone": "resnet18", "embedding_dim": 64}), "embedding_dim"),
        (_payload(canvas_size={"width": 128, "height": 128}), "canvas"),
        (_payload(model_state="not-a-dict"), "model_state"),
    ]
    for payload, match in cases:
        path = tmp_path / f"ck_{match}.pt"
        torch.save(payload, path)
        settings = Settings(
            signature_ai_enabled=True,
            signature_ai_checkpoint=str(path),
            signature_ai_device="cpu",
        )
        service = AISignatureVerificationService(settings)
        with pytest.raises(AICheckpointError, match=match):
            service.load()


def test_invalid_checkpoint_endpoint_returns_503(client, tmp_path, monkeypatch):
    import torch

    bad = tmp_path / "wrong_version.pt"
    torch.save(
        {
            "model_version": "ai_metric_v3",
            "architecture": {"backbone": "resnet18", "embedding_dim": 128},
            "canvas_size": {"width": 256, "height": 128},
            "model_state": {},
        },
        bad,
    )
    _write_env(monkeypatch, SIGNATURE_AI_CHECKPOINT=str(bad), SIGNATURE_AI_DEVICE="cpu")
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(1), "image/png"),
        },
    )
    assert resp.status_code == 503
    assert resp.json()["success"] is False
    # OpenCV baseline remains functional even with a broken AI checkpoint.
    ok = client.post(
        "/api/signatures/compare",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    assert ok.status_code == 200
    assert ok.json()["method"] == "opencv_baseline"


# ---------------------------------------------------------------------------
# 5 + 6. CPU inference / CUDA selection
# ---------------------------------------------------------------------------
def test_cpu_inference_works(cpu_settings):
    service = get_ai_service(cpu_settings)
    emb = service.encode_signature(_make_signature_bytes())
    assert emb.shape == (128,)


def test_device_resolution_auto_and_cuda():
    import torch

    from app.services.ai_signature_verification_service import AIError, _resolve_device

    dev, name = _resolve_device("auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert name == expected
    dev, name = _resolve_device("cpu")
    assert name == "cpu"
    if torch.cuda.is_available():
        dev, name = _resolve_device("cuda")
        assert name == "cuda"
    else:
        with pytest.raises(AIError):
            _resolve_device("cuda")


# ---------------------------------------------------------------------------
# 7 + 8 + 9. Preprocessing shape / determinism / parity (no augmentation)
# ---------------------------------------------------------------------------
def test_preprocessing_output_shape(cpu_settings):
    service = get_ai_service(cpu_settings)
    tensor, bbox = service._prepare_image(_make_signature_bytes())
    assert tuple(tensor.shape) == (1, 3, 128, 256)
    assert bbox["width"] > 0 and bbox["height"] > 0
    assert 0 <= bbox["x"] < 256 and 0 <= bbox["y"] < 128


def test_preprocessing_deterministic(cpu_settings):
    service = get_ai_service(cpu_settings)
    bytes_ = _make_signature_bytes()
    t1, _ = service._prepare_image(bytes_)
    t2, _ = service._prepare_image(bytes_)
    assert np.array_equal(t1.numpy(), t2.numpy())


def test_preprocessing_parity_with_ai_evaluation(cpu_settings, tmp_path):
    import cv2

    from ai.config import ExperimentConfig
    from ai.preprocessing import preprocess

    service = get_ai_service(cpu_settings)
    bytes_ = _make_signature_bytes()
    path = tmp_path / "sig.png"
    with open(path, "wb") as fh:
        fh.write(bytes_)

    expected = preprocess(path, ExperimentConfig(), augment=False)
    service_tensor, _ = service._prepare_image(bytes_)
    assert service_tensor.shape == expected.unsqueeze(0).shape
    assert np.allclose(service_tensor.numpy(), expected.unsqueeze(0).numpy(), atol=1e-6)


def test_no_augmentation_in_inference(cpu_settings, tmp_path):
    from ai.config import ExperimentConfig
    from ai.preprocessing import preprocess

    service = get_ai_service(cpu_settings)
    bytes_ = _make_signature_bytes()
    t, _ = service._prepare_image(bytes_)
    path = tmp_path / "_aug_check.png"
    with open(path, "wb") as fh:
        fh.write(bytes_)
    augmented = preprocess(path, ExperimentConfig(), augment=True, seed=7)
    # The service output is NOT affected by training augmentation (it must match
    # the deterministic eval path, not the augmented one).
    assert not np.allclose(t.numpy(), augmented.unsqueeze(0).numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# 10 + 11 + 12 + 13. Embedding / similarity
# ---------------------------------------------------------------------------
def test_embedding_dimension(cpu_settings):
    service = get_ai_service(cpu_settings)
    assert service.embedding_dim == EMBEDDING_DIMENSION
    assert service.encode_signature(_make_signature_bytes()).shape == (EMBEDDING_DIMENSION,)


def test_embedding_l2_norm_unit(cpu_settings):
    service = get_ai_service(cpu_settings)
    emb = service.encode_signature(_make_signature_bytes())
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-5


def test_similarity_in_range(cpu_settings):
    service = get_ai_service(cpu_settings)
    a = service.encode_signature(_make_signature_bytes(0))
    b = service.encode_signature(_make_signature_bytes(1))
    sim = float(np.dot(a, b))
    assert -1.0 <= sim <= 1.0


def test_identical_image_high_similarity(cpu_settings):
    service = get_ai_service(cpu_settings)
    res = service.compare_signatures_ai(_make_signature_bytes(), _make_signature_bytes())
    assert res["similarity_score"] >= 0.999
    assert res["method"] == METHOD
    assert res["version"] == VERSION
    assert res["model"] == MODEL_NAME
    assert res["embedding_dimension"] == EMBEDDING_DIMENSION


# ---------------------------------------------------------------------------
# 14 + 15 + 16. Pair endpoint / errors
# ---------------------------------------------------------------------------
def test_pair_endpoint_success(client, cpu_settings):
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["method"] == "ai_metric"
    assert body["version"] == "v2"
    assert body["model"] == "siamese_resnet18"
    assert body["embedding_dimension"] == 128
    assert body["device"] in ("cuda", "cpu")
    assert -1.0 <= body["similarity_score"] <= 1.0


def test_pair_endpoint_real_cedar_genuine(client, cpu_settings):
    _cedar_bytes("original_7_5.png")
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("e.png", _cedar_bytes("original_7_5.png"), "image/png"),
            "reference_file": ("r.png", _cedar_bytes("original_7_6.png"), "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["similarity_score"] > 0.8


def test_invalid_image_controlled_error(client, cpu_settings):
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", b"not an image at all", "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    assert resp.status_code == 400


def test_blank_image_controlled_error(client, cpu_settings):
    blank = np.full((300, 150), 255, dtype=np.uint8)
    import cv2

    ok, encoded = cv2.imencode(".png", blank)
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", encoded.tobytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 17. OpenCV baseline unchanged
# ---------------------------------------------------------------------------
def test_opencv_compare_endpoint_unchanged(client, cpu_settings):
    resp = client.post(
        "/api/signatures/compare",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "opencv_baseline"
    assert body["version"] == "v1"
    assert 0.0 <= body["similarity_score"] <= 1.0
    assert "metrics" in body


# ---------------------------------------------------------------------------
# 18. Model loaded once, not per request
# ---------------------------------------------------------------------------
def test_model_loaded_once(cpu_settings):
    import app.services.ai_signature_verification_service as svc_mod

    first = get_ai_service(cpu_settings)
    second = get_ai_service(cpu_settings)
    assert first is second
    assert svc_mod._service is first


# ---------------------------------------------------------------------------
# 19 + 20. Multi-reference service
# ---------------------------------------------------------------------------
def test_multiref_one_score_per_reference(cpu_settings):
    service = get_ai_service(cpu_settings)
    result = service.compare_against_references_ai(
        _make_signature_bytes(), [_make_signature_bytes(), _make_signature_bytes(1)]
    )
    assert result["K"] == 2
    assert len(result["similarities"]) == 2
    assert all(-1.0 <= s <= 1.0 for s in result["similarities"])


def test_multiref_aggregation(cpu_settings):
    from app.services.ai_signature_verification_service import AISignatureVerificationService as S

    sims = [0.9, 0.5, 0.7, 0.2]
    assert S.aggregate(sims, "max") == pytest.approx(0.9)
    assert S.aggregate(sims, "mean") == pytest.approx(0.575)
    assert S.aggregate(sims, "median") == pytest.approx(0.6)
    assert S.aggregate(sims, "top2_mean") == pytest.approx(0.8)


def test_multiref_k3_z_normalization_references_only(cpu_settings):
    service = get_ai_service(cpu_settings)
    refs = [_make_signature_bytes() for _ in range(3)]
    result = service.compare_against_references_ai(_make_signature_bytes(), refs)
    assert result["K"] == 3
    assert result["normalization"] in ("raw", "z")
    assert len(result["similarities"]) == 3
    assert result["aggregations"]["max"] >= result["aggregations"]["mean"]


# ---------------------------------------------------------------------------
# 21. No threshold / conformity decision returned
# ---------------------------------------------------------------------------
def test_no_threshold_or_decision_returned(client, cpu_settings):
    resp = client.post(
        "/api/signatures/compare-ai",
        files={
            "extracted_file": ("a.png", _make_signature_bytes(), "image/png"),
            "reference_file": ("b.png", _make_signature_bytes(), "image/png"),
        },
    )
    body = resp.json()
    for forbidden in ("threshold", "decision", "probability", "verdict", "conform", "authentic"):
        assert forbidden not in body, f"forbidden key/field: {forbidden}"