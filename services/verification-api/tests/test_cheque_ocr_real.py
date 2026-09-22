"""Real PaddleOCR smoke test — isolated worker, no mock.

Runs only when OCR worker is available and RUN_REAL_OCR=1.
Does not run on every CI; requires model download (~100MB) and ~10s.

Verifies:
- PaddleOCR model initializes via worker
- Real cheque image decodes
- OCR returns non-empty text
- No shm.dll crash
"""

import os

import pytest

# Skip unless explicitly enabled
RUN_REAL = os.getenv("RUN_REAL_OCR") == "1"


@pytest.mark.skipif(not RUN_REAL, reason="Real OCR disabled (set RUN_REAL_OCR=1 to enable)")
def test_real_cheque_ocr_returns_text():
    import pathlib

    from app.services.cheque_ocr_service import run_ocr

    cheque_path = pathlib.Path(__file__).parent.parent.parent / "src" / "ChequeVerification.Web" / "wwwroot" / "uploads" / "cheques" / "3fe42d8c4d0c4bb7b145f9cd8ab8f235.png"
    if not cheque_path.exists():
        # Try alternative
        cheque_path = pathlib.Path("C:/Users/nassime khatib/Desktop/ChequeVerificationPlatform/src/ChequeVerification.Web/wwwroot/uploads/cheques/3fe42d8c4d0c4bb7b145f9cd8ab8f235.png")
    if not cheque_path.exists():
        pytest.skip("Cheque image not found")

    img_bytes = cheque_path.read_bytes()
    result = run_ocr(img_bytes)
    assert result["success"] is True
    assert len(result["full_text"]) > 10
    assert len(result["lines"]) >= 3
    # At least one line should contain BANQUE or PAYEZ or MAD/DH
    full_upper = result["full_text"].upper()
    assert any(k in full_upper for k in ("BANQUE", "PAYEZ", "MAD", "DH", "CHEQUE"))
    assert result["lang"] in ("fr", "en")
    assert result["device"] == "cpu"
    # Fields optional but cheque_number should be found
    assert result["fields"]["cheque_number"] is not None or result["fields"]["amount_numeric"] is not None
