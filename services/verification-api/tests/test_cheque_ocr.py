"""Cheque OCR endpoint tests — isolated worker mocked.

V1: Tests do not download OCR models. Worker is mocked via run_ocr patch.
"""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services import cheque_ocr_service
import app.api.routes.cheques_ocr as routes_module


def _tiny_png_bytes() -> bytes:
    import cv2
    import numpy as np

    img = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (10, 10), (90, 30), (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _mock_run_ocr_success(image_bytes: bytes):
    return {
        "success": True,
        "message": "OCR effectuée avec succès.",
        "full_text": "BANQUE ALGERIE\nPAYEZ CONTRE CE CHEQUE\nCHQ 123456",
        "lines": [
            {"text": "BANQUE ALGERIE", "confidence": 0.98, "box": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"text": "PAYEZ CONTRE CE CHEQUE", "confidence": 0.95, "box": [[0, 12], [20, 12], [20, 22], [0, 22]]},
            {"text": "CHQ 123456", "confidence": 0.92, "box": [[0, 24], [15, 24], [15, 34], [0, 34]]},
        ],
        "fields": {"cheque_number": "123456", "date": None, "amount_text": None, "amount_numeric": None, "account_number": None},
        "processing_ms": 42,
        "lang": "fr",
        "device": "cpu",
    }


def _patch_run_ocr(fake_run, fake_status=None):
    orig_run_svc = cheque_ocr_service.run_ocr
    orig_status_svc = cheque_ocr_service.get_ocr_status
    orig_run_routes = routes_module.run_ocr
    orig_status_routes = routes_module.get_ocr_status
    cheque_ocr_service.run_ocr = fake_run  # type: ignore
    routes_module.run_ocr = fake_run  # type: ignore
    if fake_status is not None:
        cheque_ocr_service.get_ocr_status = fake_status  # type: ignore
        routes_module.get_ocr_status = fake_status  # type: ignore
    return orig_run_svc, orig_status_svc, orig_run_routes, orig_status_routes


def _restore_run_ocr(orig_run_svc, orig_status_svc, orig_run_routes, orig_status_routes):
    cheque_ocr_service.run_ocr = orig_run_svc  # type: ignore
    cheque_ocr_service.get_ocr_status = orig_status_svc  # type: ignore
    routes_module.run_ocr = orig_run_routes  # type: ignore
    routes_module.get_ocr_status = orig_status_routes  # type: ignore


def _client_with_mock():
    def fake_run(image_bytes: bytes):
        return _mock_run_ocr_success(image_bytes)

    def fake_status():
        return {"available": True, "lang": "fr", "device": "cpu", "error": None}

    origs = _patch_run_ocr(fake_run, fake_status)
    app = create_app()
    client = TestClient(app)
    return client, origs


def test_valid_image_returns_normalized_response():
    client, origs = _client_with_mock()
    try:
        png = _tiny_png_bytes()
        resp = client.post("/api/cheques/ocr", files={"file": ("cheque.png", png, "image/png")})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["success"] is True
        assert "BANQUE" in data["full_text"]
        assert len(data["lines"]) == 3
        assert data["lines"][0]["text"] == "BANQUE ALGERIE"
        assert 0.9 <= data["lines"][0]["confidence"] <= 1.0
        assert data["processing_ms"] >= 0
        assert data["lang"] in ("fr", "en")
        assert data["device"] == "cpu"
        assert "fields" in data
        assert data["fields"]["cheque_number"] is not None
    finally:
        _restore_run_ocr(*origs)


def test_invalid_bytes_rejected():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/cheques/ocr", files={"file": ("bad.png", b"not an image at all", "image/png")})
    assert resp.status_code == 400


def test_empty_file_rejected():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/api/cheques/ocr", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 400


def test_paddleocr_unavailable_returns_503():
    def failing_run(image_bytes: bytes):
        raise cheque_ocr_service.OcrUnavailableError("PaddleOCR non installé")

    def failing_status():
        return {"available": False, "lang": "fr", "device": "unavailable", "error": "PaddleOCR non installé"}

    origs = _patch_run_ocr(failing_run, failing_status)
    app = create_app()
    client = TestClient(app)
    png = _tiny_png_bytes()
    resp = client.post("/api/cheques/ocr", files={"file": ("cheque.png", png, "image/png")})
    assert resp.status_code == 503, resp.text
    assert "PaddleOCR" in resp.text or "indisponible" in resp.text.lower()
    _restore_run_ocr(*origs)


def test_response_contains_lines_text():
    client, origs = _client_with_mock()
    try:
        png = _tiny_png_bytes()
        resp = client.post("/api/cheques/ocr", files={"file": ("cheque.png", png, "image/png")})
        data = resp.json()
        assert data["lines"][1]["text"] == "PAYEZ CONTRE CE CHEQUE"
        assert data["full_text"].count("\n") == 2
    finally:
        _restore_run_ocr(*origs)


def test_no_crash_when_ocr_init_fails():
    def failing_run(image_bytes: bytes):
        raise cheque_ocr_service.OcrUnavailableError("init failed on purpose")

    def failing_status():
        return {"available": False, "lang": "fr", "device": "unavailable", "error": "init failed"}

    origs = _patch_run_ocr(failing_run, failing_status)
    app = create_app()
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200
    png = _tiny_png_bytes()
    resp = client.post("/api/cheques/ocr", files={"file": ("cheque.png", png, "image/png")})
    assert resp.status_code == 503
    _restore_run_ocr(*origs)
