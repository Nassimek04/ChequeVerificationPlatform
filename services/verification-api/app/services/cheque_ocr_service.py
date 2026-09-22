"""Cheque OCR service — Docker Linux worker via HTTP.

V1 architecture:

  Main FastAPI (Windows, torch/AI) -> HTTP -> PaddleOCR Docker Linux worker (CPU)
  ASP.NET POST /api/cheques/ocr contract is PRESERVED.

- Validates image in main process (OpenCV)
- Proxies bytes to OCR_WORKER_URL via multipart/form-data
- Parses normalized worker JSON, does Moroccan field extraction in main process
- Never logs image bytes, timeout from settings.ocr_worker_timeout, localhost only

Worker endpoints:
  POST {ocr_worker_url}/ocr  (multipart file, lang=fr)
  GET  {ocr_worker_url}/health

Language: French/Latin first (lang='fr'), Arabic not pretended supported.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_OCR_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Global error cache for status
_ocr_error: Optional[BaseException] = None


class OcrUnavailableError(Exception):
    pass


class OcrImageError(Exception):
    pass


# ---------------------------------------------------------------------------
# DH-zone glued amount recovery (supervisor amount audit, read-only findings).
#
# PaddleOCR glues the handwritten amount to its "DH" label ("DH232100",
# "DH H329400#") so the generic \b-based candidate regexes never fire even
# though the digits are perfectly recognized. Conservative recovery, applied
# ONLY when the generic scoring found nothing:
#  - the line must carry the DH marker and match the full-line template
#    (marker + harmless noise + ONE intact 4-7 digit run + guards, nothing
#    else; no letter-to-digit conversion anywhere);
#  - the line must sit in the top-of-page band (first 3 non-blank OCR lines;
#    PaddleOCR emits top-to-bottom and this pipeline does not plumb boxes,
#    so line order is the available spatial context — RIB/TEL/CMC7 lines live
#    far below and are excluded by construction);
#  - exactly one distinct value across qualifying lines, else unresolved.
# "DH34" (2 digits), "29000000" (8 digits, no DH marker) and multi-candidate
# pages stay unresolved by design.
# ---------------------------------------------------------------------------
_DH_GLUED_RE = re.compile(
    r"^\s*DHS?\s*[#\s\.,;:]*H?[ #\s\.,;:]*(\d{4,7})[#\s\.,;:]*$",
    re.IGNORECASE,
)

# Max non-blank-line rank (0-based) for the DH amount zone.
_DH_ZONE_MAX_RANK = 2


def _dh_zone_digits(lines: List[Dict[str, Any]]) -> Optional[str]:
    """Return the single DH-zone digit run, or None (never guesses)."""
    found: List[str] = []
    rank = -1
    for line in lines or []:
        text = line.get("text", "") or ""
        if not text.strip():
            continue
        rank += 1
        if rank > _DH_ZONE_MAX_RANK:
            break
        m = _DH_GLUED_RE.match(text)
        if m and m.group(1) not in found:
            found.append(m.group(1))
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# CMC7 constrained validator (supervisor real-cheque dataset, Phase 1 audit).
#
# Proven structure on 7/7 rectos (CDM Recife Maroc, scan 20260226):
#     306 | <7-digit cheque> | 021780 | <16-digit account body> | 85
# e.g. 306<sep>3177634<sep>021780<sep>0000000000000000<sep>85<trail>
# Canonical: cheque = 7-digit field (NO "253" prefix fabrication);
#            account = 021780 + 16-digit body + 85 (24 digits = printed RIB).
#
# PaddleOCR is NOT CMC7-trained: MICR separators surface as A, ±, :, 8, 4, d
# and the end-mark may append/split one phantom digit. The validator therefore
# matches the anchored template per OCR line (never joins lines: audit shows
# joining corrupts via stray end-mark fragments) and rejects anything that
# does not complete the template uniquely. Printed body values are harvested
# as DIAGNOSTICS ONLY and must never substitute CMC7.
# ---------------------------------------------------------------------------

# A CMC7 field separator as misread by PaddleOCR: non-digit glyph noise plus
# at most ONE glued digit from {4, 8} (observed misreads of the MICR
# separator symbols, e.g. "3177635|4|021780", "306|8|3177641").
_CMC7_SEP = r"(?:[^0-9]*[48])?[^0-9]*"

# Full anchored template on a whitespace-stripped, uppercased OCR line.
_CMC7_TEMPLATE = re.compile(
    r"^306" + _CMC7_SEP
    + r"(?P<cheque>\d{7})" + _CMC7_SEP
    + r"021780" + _CMC7_SEP
    + r"(?P<account_body>\d{16})" + _CMC7_SEP
    + r"85(?P<tail>.*)$"
)

# Tail after the final 85: only end-mark noise is tolerated — non-digits plus
# at most ONE phantom digit (observed: "d", "2", "1", "0" from the MICR
# end-mark; anything more means phantom digits and the candidate is rejected).
_CMC7_TAIL_OK = re.compile(r"^[^0-9]*[0-9]?[^0-9]*$")

# Printed-body diagnostics (evidence-anchored to the supervisor dataset only):
# body cheque "253 3177634", body RIB "021 780 0000 000 000 00000 0 85".
_PRINTED_CHEQUE_RE = re.compile(r"253\s*(?P<num>3\d{6})")
_PRINTED_RIB_RE = re.compile(
    r"021\s*780\s*0000\s*000\s*000\s*00000\s*0\s*85"
)

# Verso markers observed on the supervisor verso images (no CMC7 band there).
_VERSO_MARKERS = ("AVIS IMPORTANT", "TRG3", "CODE DE COMMERCE", "15-95")


def _compact_cmc7(text: str) -> str:
    """Uppercase + strip ALL whitespace (incl. NBSP) for template matching."""
    return re.sub(r"\s+", "", (text or "").upper())


def extract_cmc7(lines: List[Dict[str, Any]], full_text: str) -> Dict[str, Optional[str]]:
    """Constrained CMC7 parse. Never guesses digits; rejects ambiguous reads.

    Returns keys: cmc7_raw, cmc7_cheque_number, cmc7_account_number,
    cmc7_valid (bool), cmc7_error, cmc7_trailing_noise (bool),
    printed_cheque_number, printed_account_number, cmc7_cross_check,
    is_probable_verso (bool).
    """
    result: Dict[str, Optional[str]] = {
        "cmc7_raw": None,
        "cmc7_cheque_number": None,
        "cmc7_account_number": None,
        "cmc7_valid": False,
        "cmc7_error": "CMC7 non détecté.",
        "cmc7_trailing_noise": False,
        "printed_cheque_number": None,
        "printed_account_number": None,
        "cmc7_cross_check": None,
        "is_probable_verso": False,
    }

    text_upper = (full_text or "").upper()
    if any(m in text_upper for m in _VERSO_MARKERS):
        result["is_probable_verso"] = True

    # --- printed-body diagnostics (FULL TEXT only, never authoritative) ---
    m_pc = _PRINTED_CHEQUE_RE.search(full_text or "")
    if m_pc:
        result["printed_cheque_number"] = "253" + m_pc.group("num")
    m_pa = _PRINTED_RIB_RE.search(full_text or "")
    if m_pa:
        result["printed_account_number"] = re.sub(r"\s+", "", m_pa.group(0))

    # --- CMC7 candidates: per OCR line, never joined across lines ---
    saw_candidate = False
    first_candidate_raw: Optional[str] = None
    for line in lines or []:
        compact = _compact_cmc7(line.get("text", "") or "")
        if not compact:
            continue
        # Cheap prefilter: must mention the proven anchors and carry enough
        # digits; failed MICR reads (e.g. "1S8TSE...") fall out here.
        digits_only = re.sub(r"\D", "", compact)
        if "306" not in compact and "021780" not in compact:
            continue
        if len(digits_only) < 30:
            continue
        saw_candidate = True
        if first_candidate_raw is None:
            first_candidate_raw = line.get("text", "")
        m = _CMC7_TEMPLATE.match(compact)
        if not m:
            continue
        tail = m.group("tail") or ""
        if not _CMC7_TAIL_OK.match(tail):
            # Phantom digits after the final 85: not safely recoverable.
            result["cmc7_raw"] = line.get("text", "")
            result["cmc7_error"] = "CMC7 invalide (caractères excédentaires)."
            return result
        cheque = m.group("cheque")
        account = "021780" + m.group("account_body") + "85"
        result["cmc7_raw"] = line.get("text", "")
        result["cmc7_cheque_number"] = cheque
        result["cmc7_account_number"] = account
        result["cmc7_valid"] = True
        result["cmc7_error"] = None
        result["cmc7_trailing_noise"] = bool(re.search(r"[0-9]", tail))
        break

    if result["cmc7_valid"]:
        # --- cross-check vs printed body (diagnostic only) ---
        notes = []
        if result["printed_cheque_number"]:
            if result["printed_cheque_number"].endswith(result["cmc7_cheque_number"] or ""):
                notes.append("cheque-suffix-compatible")
            else:
                notes.append("CHEQUE-MISMATCH")
        else:
            notes.append("cheque-body-absent")
        if result["printed_account_number"]:
            if result["printed_account_number"] == result["cmc7_account_number"]:
                notes.append("account-exact")
            else:
                notes.append("ACCOUNT-MISMATCH")
        else:
            notes.append("account-body-absent")
        if result["cmc7_trailing_noise"]:
            notes.append("trailing-mark-noise")
        result["cmc7_cross_check"] = ";".join(notes)
    elif saw_candidate:
        if result["cmc7_error"] == "CMC7 non détecté.":
            result["cmc7_error"] = "CMC7 invalide (structure non conforme)."
        # Surface the failed candidate line as diagnostic evidence only.
        result["cmc7_raw"] = result["cmc7_raw"] or first_candidate_raw

    return result


def get_ocr_status() -> Dict[str, Any]:
    """Never raises. Probes Docker worker health without heavy init."""
    settings = get_settings()
    url = settings.ocr_worker_url.rstrip("/") + "/health"
    try:
        # short timeout for health probe
        resp = httpx.get(url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            # worker returns {"status":"ok","ocr_available":true,"device":"cpu"}
            available = bool(data.get("ocr_available", True))
            device = data.get("device", "cpu")
            if available:
                return {"available": True, "lang": "fr", "device": device, "error": None}
            return {"available": False, "lang": "fr", "device": device, "error": data.get("error")}
        return {"available": False, "lang": "fr", "device": "unavailable", "error": f"worker health {resp.status_code}"}
    except Exception as exc:
        return {"available": False, "lang": "fr", "device": "unavailable", "error": str(exc)[:500]}


def reset_ocr_engine() -> None:
    global _ocr_error
    _ocr_error = None


def validate_and_decode(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise OcrImageError("Le fichier est vide.")
    if len(image_bytes) > MAX_OCR_FILE_SIZE_BYTES:
        raise OcrImageError("Le fichier dépasse la taille maximale autorisée (10 Mo).")
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise OcrImageError("L'image ne peut pas être décodée.")
    if img.size == 0:
        raise OcrImageError("Image vide après décodage.")
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        raise OcrImageError("Image trop petite pour l'OCR.")
    if h > 8000 or w > 8000:
        raise OcrImageError("Image trop grande pour l'OCR.")
    return img


def _extract_fields(full_text: str, lines: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Moroccan field parsing — MAD/DH/DHS/DIRHAM, no DA/DZD unless intentionally."""
    fields: Dict[str, Optional[str]] = {
        "cheque_number": None,
        "date": None,
        "amount_text": None,
        "amount_numeric": None,
        "account_number": None,
    }
    if not full_text:
        return fields
    text_upper = full_text.upper()
    # Phase 1 batch-import references (conservative, additive only):
    # alphanumeric test accounts like BLIND-0001 and cheque refs like
    # CHQ-BATCH-001 are exposed as-is. Generic numeric logic below is untouched.
    m_blind = re.search(r"\bBLIND\s*-\s*(\d{1,10})\b", text_upper)
    if m_blind:
        fields["account_number"] = "BLIND-" + m_blind.group(1)
    m_chq = re.search(r"\bCHQ\s*-\s*([A-Z0-9]+(?:\s*-\s*[A-Z0-9]+)*)", text_upper)
    if m_chq:
        fields["cheque_number"] = "CHQ-" + re.sub(r"\s+", "", m_chq.group(1))
    m = re.search(r"(?:CH[EÈ]QUE|CHEQUE|N[°º]|NUM[ÉE]RO)[\s:]*([A-Z0-9\-]{5,20})", text_upper)
    if m and fields["cheque_number"] is None:
        candidate = m.group(1).strip()
        if re.search(r"\d", candidate):
            fields["cheque_number"] = candidate
    if fields["cheque_number"] is None:
        nums = re.findall(r"\b\d{6,10}\b", full_text)
        if nums:
            fields["cheque_number"] = nums[0]
    # Date with flexible spaces: 18 /05 / 2026 -> 18/05/2026
    m_date = re.search(r"\b(\d{1,2}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{2,4})\b", full_text)
    if m_date:
        raw_date = m_date.group(1)
        # Normalize: remove spaces around separators
        fields["date"] = re.sub(r"\s*([\/\-\.])\s*", r"\1", raw_date)
    # --- Moroccan amount extraction V1 (context-aware) ---
    # Previous simple "first currency line" picked CAPITAL 1.000.000.000 MAD
    # instead of DH 40.000,00. New strategy: candidate scoring with positive/
    # negative context and line proximity (±3), then normalization.
    moroccan_currency = ("MAD", "DH", "DHS", "DIRHAM", "DIRHAMS")
    positive_terms = ("LA SOMME DE", "SOMME", "MONTANT", "PAYEZ CONTRE", "À L'ORDRE", "A L'ORDRE", "MILLE", "CENT", "DIRHAM", "DIRHAMS")
    negative_terms = ("CAPITAL", "CAPITAL SOCIAL")

    def _normalize_amount(raw: str) -> Optional[str]:
        """Normalize Moroccan/French formats to invariant 1234.56."""
        if not raw:
            return None
        s = raw.strip().replace("\u00a0", " ").replace(" ", "")
        # Remove stray leading/trailing punctuation
        s = s.strip(".,")
        if not s:
            return None
        # Both separators present: rightmost is decimal
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                # French: 40.000,00 -> 40000.00
                s = s.replace(".", "").replace(",", ".")
            else:
                # US: 40,000.00 -> 40000.00
                s = s.replace(",", "")
        elif "," in s:
            # Only comma
            if re.search(r",\d{2}$", s):
                # comma is decimal e.g., 40000,00 -> 40000.00
                # Remove any dot thousand first (unlikely) then replace
                s = s.replace(".", "").replace(",", ".")
            else:
                # comma as thousand (e.g., 40,000)
                s = s.replace(",", "")
        elif "." in s:
            if re.search(r"\.\d{2}$", s):
                # dot is decimal; may have multiple dots as thousand
                parts = s.split(".")
                if len(parts) > 2:
                    # e.g., 1.000.000.00 -> 1000000.00
                    s = "".join(parts[:-1]) + "." + parts[-1]
                # else single dot decimal keeps as is
            else:
                # dot as thousand, no decimal -> integer
                s = s.replace(".", "")
        # Now s should be digits + optional dot
        if "." not in s:
            # integer amount -> add .00
            if not s.isdigit():
                # may have remaining separators
                s = re.sub(r"[^\d]", "", s)
            if not s.isdigit() or len(s) == 0:
                return None
            # Reject implausibly small integer without context? handled by scoring
            s = s + ".00"
        else:
            integer, dec = s.split(".", 1)
            integer = re.sub(r"[^\d]", "", integer)
            dec = re.sub(r"[^\d]", "", dec)
            if not integer:
                integer = "0"
            dec = (dec + "00")[:2]
            s = integer + "." + dec
        try:
            v = float(s)
            if v < 0 or v > 99999999:
                # allow up to ~100M, but scoring will penalize huge capital-like
                pass
            return f"{float(s):.2f}" if "." in s else s
        except ValueError:
            return None

    # Collect candidates
    candidates = []  # list of {raw, line_idx, type}
    for idx, line in enumerate(lines):
        txt = line.get("text", "") or ""
        # decimal candidates: at least decimal 2 digits
        for m in re.finditer(r"\b\d[\d\s\.,]*\d[,\.]\d{2}\b", txt):
            raw = m.group(0).strip()
            # quick reject: contains / or - (date-like) or : (time)
            if "/" in raw or ":" in raw:
                continue
            candidates.append({"raw": raw, "line_idx": idx, "type": "decimal"})
        # integer candidates: 4-6 digits optionally with space/dot thousand (e.g., 40 000, 40.000, 40000)
        # Only consider integers that could be amount without decimal (task examples 40 000)
        for m in re.finditer(r"\b\d{1,3}(?:[ \.]\d{3})+\b", txt):
            raw = m.group(0).strip()
            if "/" in raw or ":" in raw:
                continue
            # avoid duplicate if already covered by decimal candidate (overlap)
            if any(c["raw"] == raw and c["line_idx"] == idx for c in candidates):
                continue
            # Heuristic: integer candidate must be 4-7 chars with space/dot, not phone fragment
            # Keep for scoring (weak)
            candidates.append({"raw": raw, "line_idx": idx, "type": "integer"})
        for m in re.finditer(r"\b\d{4,6}\b", txt):
            raw = m.group(0).strip()
            if any(c["raw"] == raw and c["line_idx"] == idx for c in candidates):
                continue
            # Avoid cheque_number/account duplicates later via scoring
            candidates.append({"raw": raw, "line_idx": idx, "type": "integer"})

    # Also add fallback from full_text if no line candidates (should be rare)
    if not candidates:
        for m in re.finditer(r"\b\d[\d\s\.,]*\d[,\.]\d{2}\b", full_text):
            raw = m.group(0).strip()
            if "/" in raw:
                continue
            candidates.append({"raw": raw, "line_idx": -1, "type": "decimal"})

    best = None
    best_score = float("-inf")
    best_norm = None

    # Precompute line upper texts for context lookup
    line_uppers = [(l.get("text", "") or "").upper() for l in lines]

    for cand in candidates:
        raw = cand["raw"]
        li = cand["line_idx"]
        typ = cand["type"]
        norm = _normalize_amount(raw)
        if norm is None:
            continue
        # Validate norm is plausible amount (not account)
        try:
            val = float(norm)
        except ValueError:
            continue
        # Score
        score = 0
        # Base: decimal strong, integer weak
        if typ == "decimal":
            score += 10
        else:
            score += 2
            # integer without decimal only viable near currency; will get bonus below
        # Negative context: immediate line contains CAPITAL -> heavy penalty / reject
        neg_penalty = 0
        pos_bonus = 0
        # Check lines in window ±3 for positive/negative markers
        for d in range(0, 4):
            for offset in ([0] if d == 0 else [-d, d]):
                j = li + offset if li >= 0 else -1
                if j < 0 or j >= len(line_uppers):
                    continue
                up = line_uppers[j]
                if any(nt in up for nt in negative_terms):
                    if d == 0:
                        neg_penalty = max(neg_penalty, 80)
                    elif d == 1:
                        neg_penalty = max(neg_penalty, 20)
                    elif d == 2:
                        neg_penalty = max(neg_penalty, 8)
                    else:
                        neg_penalty = max(neg_penalty, 0)
                if any(pt in up for pt in moroccan_currency):
                    # distance-based bonus
                    if d == 0:
                        pos_bonus = max(pos_bonus, 10)
                    elif d == 1:
                        pos_bonus = max(pos_bonus, 7)
                    elif d == 2:
                        pos_bonus = max(pos_bonus, 4)
                    elif d == 3:
                        pos_bonus = max(pos_bonus, 2)
                # Additional textual amount context
                if any(pt in up for pt in positive_terms):
                    if d <= 1:
                        pos_bonus = max(pos_bonus, 6) if "MONTANT" in up or "SOMME" in up else pos_bonus
                        # mille/cent/dirham near amount
                        if any(k in up for k in ("MILLE", "CENT")):
                            pos_bonus = max(pos_bonus, 5)
        score += pos_bonus
        score -= neg_penalty
        # Extra penalty if candidate line itself is account/phone/date region
        if li >= 0:
            up_line = line_uppers[li]
            if "COMPTE" in up_line or "RIB" in up_line:
                score -= 50
            if "TEL" in up_line or "TÉL" in up_line or re.search(r"0\d\s*\d{2}\s*\d{2}", up_line):
                # phone pattern
                score -= 20
            if re.search(r"\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{2,4}", up_line):
                # date line often contains candidate date, not amount
                # but amount candidate regex excludes '/', so this is just penalty if near
                pass
            # If candidate raw is exactly the account number (16 digits) or cheque number, penalize integer
            if typ == "integer" and len(re.sub(r"[^\d]", "", raw)) >= 8 and "COMPTE" in up_line:
                score -= 40
        # Penalize implausibly small integer without currency (e.g., 1234 near no marker)
        if typ == "integer" and pos_bonus == 0:
            score -= 8
        # Keep best
        if score > best_score:
            best_score = score
            best = cand
            best_norm = norm

    # Decision: require positive context or at least not heavily penalized
    # If best_score < 0 or pos_bonus==0 and typ==integer -> no reliable amount
    found_amount = False
    if best is not None and best_score >= 5:
        # Ensure not pure negative
        fields["amount_text"] = best["raw"].strip()
        fields["amount_numeric"] = best_norm
        found_amount = True
    if not found_amount:
        # DH-zone glued recovery (additive fallback only; generic winners keep
        # priority and ambiguous pages stay unresolved — see _dh_zone_digits).
        dh_raw = _dh_zone_digits(lines)
        if dh_raw is not None:
            dh_norm = _normalize_amount(dh_raw)
            if dh_norm is not None:
                fields["amount_text"] = dh_raw.strip()
                fields["amount_numeric"] = dh_norm
                found_amount = True
    # Account: try per-line, handle split across lines (COMPTE N° on one line, number on next).
    # A BLIND-XXXX reference detected above is authoritative and never overwritten
    # by the generic digit-only logic below.
    for idx, line in enumerate(lines):
        if fields["account_number"] is not None:
            break
        lt = line.get("text", "").upper()
        if any(k in lt for k in ("COMPTE", "ACCOUNT", "RIB")):
            m_acc = re.search(r"(?:COMPTE|ACCOUNT|RIB|N[°º]\s*COMPTE)[\s:]*([0-9\s\-]{8,34})", lt)
            if m_acc:
                cand = re.sub(r"[\s\-]", "", m_acc.group(1))
                if 8 <= len(cand) <= 30 and cand.isdigit():
                    fields["account_number"] = cand
                    break
            # Check next line if current has label but no digits
            if idx + 1 < len(lines):
                nxt = lines[idx + 1].get("text", "")
                cand2 = re.sub(r"[\s\-]", "", nxt)
                if 8 <= len(cand2) <= 30 and cand2.isdigit():
                    fields["account_number"] = cand2
                    break
    if fields["account_number"] is None:
        m_acc = re.search(r"(?:COMPTE|ACCOUNT|RIB|N[°º]\s*COMPTE)[\s:]*([0-9\s\-]{8,34})", text_upper)
        if m_acc:
            cand = re.sub(r"[\s\-]", "", m_acc.group(1))
            if 8 <= len(cand) <= 30 and cand.isdigit():
                fields["account_number"] = cand
    return fields


def run_ocr(image_bytes: bytes) -> Dict[str, Any]:
    """Validate, proxy to Docker worker, parse, extract fields."""
    global _ocr_error
    start = time.perf_counter()
    # Validate in main process (fast, no worker call)
    validate_and_decode(image_bytes)

    settings = get_settings()
    worker_url = settings.ocr_worker_url.rstrip("/") + "/ocr"
    timeout = float(settings.ocr_worker_timeout)

    # Proxy to worker via HTTP multipart
    try:
        files = {"file": ("cheque.png", image_bytes, "image/png")}
        data = {"lang": "fr"}
        resp = httpx.post(worker_url, files=files, data=data, timeout=timeout)
        if resp.status_code == 413:
            raise OcrImageError("Le fichier dépasse la taille maximale autorisée (10 Mo).")
        if resp.status_code == 415:
            raise OcrImageError("Type de contenu non supporté.")
        if resp.status_code == 400:
            # Worker validation failed
            try:
                err_detail = resp.json().get("detail", "Image invalide.")
            except Exception:
                err_detail = resp.text[:500] or "Image invalide."
            raise OcrImageError(err_detail)
        if resp.status_code in (502, 503, 504):
            raise OcrUnavailableError("Service OCR indisponible.")
        if resp.status_code != 200:
            logger.warning("OCR worker HTTP %s body=%s", resp.status_code, resp.text[:500])
            raise OcrUnavailableError(f"Service OCR indisponible (HTTP {resp.status_code}).")

        try:
            wdata = resp.json()
        except Exception as exc:
            raise OcrUnavailableError("Réponse OCR invalide.") from exc

        if not wdata.get("success"):
            err = wdata.get("error") or wdata.get("detail") or "OCR échoué"
            raise OcrUnavailableError(err)

        lines = wdata.get("lines", [])
        full_text = wdata.get("full_text", "")
        # Re-parse fields in main process with Moroccan logic (authoritative)
        fields = _extract_fields(full_text, lines)
        # CMC7 constrained parse (authoritative for batch import identity;
        # generic fields above are intentionally left untouched).
        try:
            fields.update(extract_cmc7(lines, full_text))
        except Exception as exc:
            logger.warning("CMC7 extraction failed safely: %s", exc)
        # Prefer worker processing_ms but fallback to our timing
        processing_ms = wdata.get("processing_ms") or int((time.perf_counter() - start) * 1000)

        _ocr_error = None
        return {
            "success": True,
            "message": "OCR effectuée avec succès.",
            "full_text": full_text,
            "lines": lines,
            "fields": fields,
            "processing_ms": processing_ms,
            "lang": wdata.get("lang", "fr"),
            "device": wdata.get("device", "cpu"),
        }
    except OcrImageError:
        raise
    except OcrUnavailableError:
        raise
    except httpx.TimeoutException as exc:
        logger.warning("OCR worker timeout: %s", exc)
        raise OcrUnavailableError("Service OCR indisponible (timeout).") from exc
    except httpx.ConnectError as exc:
        logger.warning("OCR worker connect error: %s", exc)
        raise OcrUnavailableError("Service OCR indisponible (connexion).") from exc
    except OcrImageError:
        raise
    except Exception as exc:
        logger.exception("Unexpected OCR worker error")
        raise OcrUnavailableError(f"Erreur interne OCR : {exc}") from exc
