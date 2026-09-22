"""DH-zone glued amount tests — supervisor amount audit (read-only findings).

Covers the recoverable class ("DH232100", "DH H329400#") and proves the
must-reject class ("DH34", "29000000", multi-candidate, off-zone) stays
unresolved. Existing decimal/integer behavior is pinned unchanged.
"""

from app.services.cheque_ocr_service import _extract_fields

RIB = "021 780 0000 000 000 00000 0 85"


def _lines(*texts):
    return [{"text": t, "confidence": 0.9, "box": []} for t in texts]


def _amount(texts, full_extra=""):
    lines = _lines(*texts)
    full = "\n".join(texts) + full_extra
    f = _extract_fields(full, lines)
    return f["amount_text"], f["amount_numeric"]


def test_dh_glued_plain():
    text, num = _amount(["CM", "DH232100", "CRÉDIT DU MAROC"])
    assert num == "232100.00"


def test_dh_glued_spaced():
    text, num = _amount(["CM", "DH 232100", "CRÉDIT DU MAROC"])
    assert num == "232100.00"


def test_dh_guarded_hash():
    text, num = _amount(["DH#190250#", "CRÉDIT DU MAROC"])
    assert num == "190250.00"


def test_dh_stray_h():
    text, num = _amount(["DH H329400#", "CRÉDIT DU MAROC"])
    assert num == "329400.00"


def test_dh_zone_only_off_zone_not_promoted():
    # Same token deep in the page (non-amount region) must NOT validate.
    texts = ["CRÉDIT DU MAROC", "PAYEZ CONTRE CE CHEQUE", "Somme en toutes lettres",
             "CASABLANCA", "N° de compte", RIB, "TEL:022473887", "DH232100"]
    text, num = _amount(texts)
    assert num is None


def test_multiple_dh_candidates_unresolved():
    text, num = _amount(["DH232100", "DH H329400#", "CRÉDIT DU MAROC"])
    assert num is None


def test_dh_short_fragment_unresolved():
    # 0003 audit: "DH34" carries no usable amount.
    text, num = _amount(["CM", "DH34", "CRÉDIT DU MAROC", "Cent Quarante"])
    assert num is None


def test_separator_loss_unresolved():
    # 0013 audit: "29000000" (8 digits, no DH marker) stays unresolved.
    text, num = _amount(["29000000", "CRÉDIT DU MAROC", RIB])
    assert num is None


def test_existing_decimal_priority_kept():
    # Generic winner keeps priority over the DH fallback.
    text, num = _amount(["#300,00", "DH232100", "CRÉDIT DU MAROC"])
    assert (text, num) == ("300,00", "300.00")


def test_existing_200_unchanged():
    text, num = _amount(["#200,00#", "DH", "CRÉDIT DU MAROC"])
    assert (text, num) == ("200,00", "200.00")


def test_rib_date_tel_guardrails_unchanged():
    texts = ["À Casa, le 17/02/26", "N° de compte", RIB, "TEL:022473887",
             "253 3177634", "306±3177634A021780A0000000000000000A85d"]
    text, num = _amount(texts)
    assert num is None


def test_dh_line_with_extra_text_rejected():
    text, num = _amount(["DH232100 TEL:022473887", "CRÉDIT DU MAROC"])
    assert num is None
