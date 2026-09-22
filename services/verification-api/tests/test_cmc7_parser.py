"""CMC7 constrained parser tests — supervisor CMC7 dataset (Phase 1 audit).

Uses redacted OCR-style strings (account digits zeroed, format preserved).
No images required.
"""

from app.services.cheque_ocr_service import extract_cmc7

EXPECTED_ACCOUNT = "021780000000000000000085"

# Observed full-pipeline CMC7 lines (POST /api/cheques/ocr on the 7 fronts).
FULL_LINES = {
    "0001": "306±3177634A021780A0000000000000000A85d",
    "0003": "306:31776354021780400000000000000004852",
    "0005": "30683177641A021780A0000000000000000A85d",
    "0007": "306±3177643A021780A0000000000000000A85d",
    "0009": "306±3177642A021780A0000000000000000A85",
    "0011": "306±3177645A021780A00000000000000004851",
    "0013": "306±3177644A021780A0000000000000000",  # truncated: 85 lost
}

EXPECTED_CHEQUES = {
    "0001": "3177634",
    "0003": "3177635",
    "0005": "3177641",
    "0007": "3177643",
    "0009": "3177642",
    "0011": "3177645",
    "0013": "3177644",  # visual truth; OCR line is truncated (see test below)
}

BODY_TEXT = (
    "CRÉDIT DU MAROC\n021 780 0000 000 000 00000 0 85\n"
    "TEL:022473887\n253 3177634\nChèque N°\n"
)


def _lines(*texts):
    return [{"text": t, "confidence": 0.9, "box": []} for t in texts]


def test_all_valid_lines_parse_cheque_and_account():
    for tag in ("0001", "0003", "0005", "0007", "0009", "0011"):
        r = extract_cmc7(_lines("CRÉDIT DU MAROC", FULL_LINES[tag]), BODY_TEXT)
        assert r["cmc7_valid"] is True, tag
        assert r["cmc7_cheque_number"] == EXPECTED_CHEQUES[tag], tag
        assert r["cmc7_account_number"] == EXPECTED_ACCOUNT, tag


def test_all_valid_share_same_account_and_distinct_cheques():
    cheques = set()
    for tag in ("0001", "0003", "0005", "0007", "0009", "0011"):
        r = extract_cmc7(_lines(FULL_LINES[tag]), "")
        assert r["cmc7_account_number"] == EXPECTED_ACCOUNT
        cheques.add(r["cmc7_cheque_number"])
    assert len(cheques) == 6


def test_trailing_endmark_noise_accepted_with_flag():
    # 0003 ends "...4852" and 0011 "...4851": one phantom digit from the
    # MICR end-mark. Must validate but stay flagged as diagnostic.
    for tag in ("0003", "0011"):
        r = extract_cmc7(_lines(FULL_LINES[tag]), "")
        assert r["cmc7_valid"] is True
        assert r["cmc7_trailing_noise"] is True
    r = extract_cmc7(_lines(FULL_LINES["0001"]), "")  # tail "d", no digit
    assert r["cmc7_valid"] is True
    assert r["cmc7_trailing_noise"] is False


def test_truncated_0013_line_rejected_not_guessed():
    # The observed 0013 OCR line lost the final "85" (next line "50" is a
    # mangled fragment, not recoverable). The parser must fail, never invent.
    r = extract_cmc7(_lines(FULL_LINES["0013"], "50"), BODY_TEXT)
    assert r["cmc7_valid"] is False
    assert r["cmc7_cheque_number"] is None
    assert r["cmc7_account_number"] is None


def test_garbled_micr_reads_rejected():
    # Band-crop failures observed in audit: must not validate.
    for raw in (
        "1S8TSE6260T0090600007082520726922TE8908",
        "PS8757228070090800007082720V22922780908",
    ):
        r = extract_cmc7(_lines(raw), BODY_TEXT)
        assert r["cmc7_valid"] is False, raw


def test_phantom_digits_rejected():
    r = extract_cmc7(_lines("306±3177634A021780A0000000000000000A8599"), "")
    assert r["cmc7_valid"] is False


def test_missing_cmc7_rejected_despite_valid_body():
    # Business rule: body values must NEVER substitute a missing CMC7.
    r = extract_cmc7(_lines("CRÉDIT DU MAROC", "TEL:022473887"), BODY_TEXT)
    assert r["cmc7_valid"] is False
    assert r["cmc7_cheque_number"] is None
    # ...but body diagnostics are still harvested for display.
    assert r["printed_cheque_number"] == "2533177634"
    assert r["printed_account_number"] == EXPECTED_ACCOUNT


def test_cross_check_suffix_and_exact():
    r = extract_cmc7(_lines(FULL_LINES["0001"]), BODY_TEXT)
    assert "cheque-suffix-compatible" in (r["cmc7_cross_check"] or "")
    assert "account-exact" in (r["cmc7_cross_check"] or "")


def test_cross_check_mismatch_flagged():
    bad_body = BODY_TEXT.replace("253 3177634", "253 3999999")
    r = extract_cmc7(_lines(FULL_LINES["0001"]), bad_body)
    assert r["cmc7_valid"] is True
    assert "CHEQUE-MISMATCH" in (r["cmc7_cross_check"] or "")
    # CMC7 value itself is NOT replaced by the body value.
    assert r["cmc7_cheque_number"] == "3177634"


def test_lines_never_joined_stray_endmark():
    # 0009 audit: stray "5" on the next line must not corrupt the clean line.
    r = extract_cmc7(_lines(FULL_LINES["0009"], "5"), "")
    assert r["cmc7_valid"] is True
    assert r["cmc7_cheque_number"] == "3177642"


def test_verso_flagged_no_cmc7():
    verso = "AVIS IMPORTANT\nTRG3 - 1025 - 0100 - 07\nloi 15-95 Code de Commerce"
    r = extract_cmc7(_lines("AVIS IMPORTANT", "TRG3 - 1025 - 0100 - 07"), verso)
    assert r["cmc7_valid"] is False
    assert r["is_probable_verso"] is True


def test_empty_input_safe():
    r = extract_cmc7([], "")
    assert r["cmc7_valid"] is False
    assert r["cmc7_error"] == "CMC7 non détecté."
