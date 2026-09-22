"""Generate DEMO-V5-CONFORME-001 full synthetic Moroccan-style cheque.

Reuses validated Extraction V2.4 geometry from:
  services/verification-api/ai/v4/dataset.py::make_synthetic_cheque
  services/verification-api/ai/reports/v2/end_to_end/run_end_to_end_benchmark.py::make_synthetic_cheque

Constraints:
- Do NOT redraw/regenerate candidate signature — composite exact CEDAR original_2_6.png ink.
- Signature placement inside validated region (660,1520) x (290,590) compatible with V2.4 ROI 0.40/0.40/0.98/0.98.
- Dimensions 1600x700 validated synthetic canvas.
- Must be importable via ASP.NET upload (PNG).
"""

import hashlib
import zlib
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
CANDIDATE_SRC = CEDAR_ROOT / "2" / "original_2_6.png"
OUT_PATH = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-CONFORME-001.png")

# Reference sources (for documentation, not copied here)
REF_SRCS = [CEDAR_ROOT / "2" / f"original_2_{i}.png" for i in range(1, 6)]

CHEQUE_NUMBER = "DEMO-V5-CONFORME-001"
DATE_STR = "03/09/2026"
AMOUNT_STR = "40 000,00"
AMOUNT_CURRENCY = "MAD"
AMOUNT_WORDS = "Quarante mille dirhams"

def stable_seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))

def make_moroccan_demo_cheque(sig_gray: np.ndarray, rng: np.random.Generator):
    """Validated base + Moroccan test annotations, signature composited last."""
    W, H = 1600, 700
    base = rng.normal(244.0, 2.5, (H, W)).astype(np.float32)

    # Light guilloche waves (background texture) — validated part
    for i in range(14):
        y0 = 30 + i * 48 + int(rng.integers(-6, 7))
        amp = rng.uniform(3.0, 9.0)
        phase = rng.uniform(0, 2 * np.pi)
        xs = np.arange(W)
        ys = (y0 + amp * np.sin(xs / rng.uniform(60, 140) + phase)).astype(int)
        ok = (ys >= 0) & (ys < H)
        base[ys[ok], xs[ok]] -= rng.uniform(6.0, 12.0)

    img = np.clip(base, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Border box
    cv2.rectangle(color, (18, 18), (W - 18, H - 18), (140, 140, 140), 1)
    cv2.rectangle(color, (20, 20), (W - 20, H - 20), (175, 175, 175), 1)

    # ------------------------------------------------------------------
    # Moroccan-style banking header (outside ROI where possible)
    # ------------------------------------------------------------------
    # Top banner: Banque DEMO
    cv2.putText(color, "BANQUE DEMO TEST  -  SPECIMEN", (42, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(color, "AGENCE CENTRALE DEMO  |  TEST ONLY - NOT A REAL BANK CHEQUE", (42, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (95, 95, 95), 1, cv2.LINE_AA)

    # Cheque number (top, clearly synthetic)
    cv2.putText(color, f"CHEQUE N: {CHEQUE_NUMBER}", (1120, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)

    # Date line (validated position, enhanced label)
    cv2.line(color, (60, 70), (420, 70), (150, 150, 150), 1)
    # slight offset above line
    cv2.putText(color, f"Date: {DATE_STR}", (60, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(color, "Casablanca", (285, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)

    # Payee line — "Payez contre ce cheque..."
    cv2.putText(color, "Payez contre ce cheque la somme de :", (42, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(color, (60, 300), (900, 300), (150, 150, 150), 1)
    cv2.putText(color, AMOUNT_WORDS, (70, 292),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(color, "A  M. / Mme  TEST DEMO CLIENT", (70, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)

    # Amount box (top-right, outside ROI)
    cv2.rectangle(color, (1050, 90), (1540, 170), (150, 150, 150), 1)
    # Amount scribble background (validated)
    for k in range(6):
        p1 = (1070 + k * 75, 155 - int(rng.integers(10, 55)))
        p2 = (1120 + k * 75, 105 + int(rng.integers(5, 45)))
        cv2.line(color, p1, p2, (120, 120, 120), 1)
    # Amount text inside box
    cv2.putText(color, f"{AMOUNT_STR} {AMOUNT_CURRENCY}", (1080, 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    # small "* MAD *" label
    cv2.putText(color, "Dirhams Marocains", (1085, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (105, 105, 105), 1, cv2.LINE_AA)

    # Cheque number repeated near bottom for realism
    cv2.putText(color, f"N CHEQUE {CHEQUE_NUMBER}  |  COMPTE DEMO 011 123456789  |  40 000,00 MAD  |  03/09/2026", (48, 630),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (95, 95, 95), 1, cv2.LINE_AA)

    # TEST / DEMO watermark — large, light, diagonal, MUST be background (not ink)
    # Draw before signature so signature stays on top. Use light gray 185 so Otsu ignores it.
    watermark = "TEST  -  DEMO  -  NOT A REAL CHEQUE"
    # Create overlay for rotated text
    overlay = color.copy()
    # Use a light gray that stays in background
    # Put semi-transparent watermark across center, horizontal for now (rotation would clip)
    # We'll place two lines: one central diagonal simulation via positioning
    cv2.putText(overlay, watermark, (320, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (185, 185, 185), 2, cv2.LINE_AA)
    cv2.putText(overlay, "SPECIMEN  -  DOCUMENT DE TEST  -  NE PAS ENCAISSER", (380, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    # Blend lightly
    color = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)

    # Red stamp-like TEST mark top-center
    cv2.rectangle(color, (640, 38), (880, 78), (160, 90, 90), 1)
    cv2.putText(color, "TEST / DEMO - SPECIMEN", (652, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 60, 60), 1, cv2.LINE_AA)

    # Footer TEST bar at bottom (outside signature ROI vertical? ROI y1=686, footer y 655-682)
    # Keep footer inside MICR risk zone so extraction will filter it as MICR if needed, but light
    cv2.putText(color, "--- CE CHEQUE EST UN SPECIMEN DE TEST - DOCUMENT SYNTHETIQUE SANS VALEUR BANCAIRE ---", (150, 682),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)

    # MICR-style digit band near the bottom (inside ROI bottom-exclusion zone, validated)
    micr_y = 655
    x = 120
    while x < W - 160:
        digit = str(int(rng.integers(0, 10)))
        cv2.putText(color, digit, (x, micr_y), cv2.FONT_HERSHEY_SIMPLEX,
                    rng.uniform(0.9, 1.3), (30, 30, 30), 2, cv2.LINE_AA)
        x += int(rng.integers(38, 58))

    # Signature line and labels in signature region (light, not ink)
    cv2.line(color, (700, 595), (1480, 595), (155, 155, 155), 1)
    cv2.putText(color, "Signature du tireur  /  Signature du client", (860, 615),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(color, "(Ne pas deborder du cadre)", (970, 630),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (125, 125, 125), 1, cv2.LINE_AA)

    # Paste the signature into the lower-right region (inside the ROI) — VALIDATED
    # Realistic rendering: pen ink is DARK. CEDAR scans have light-gray strokes, so we binarize
    region_x0, region_x1 = 660, 1520
    region_y0, region_y1 = 290, 590
    target_h = int(rng.integers(130, 195))
    scale = min(target_h / sig_gray.shape[0],
                (region_x1 - region_x0) / sig_gray.shape[1])
    new_w = max(1, int(round(sig_gray.shape[1] * scale)))
    new_h = max(1, int(round(sig_gray.shape[0] * scale)))
    resized = cv2.resize(sig_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(resized.astype(np.float32), (3, 3), 0)
    otsu_thr, _ = cv2.threshold(blurred.astype(np.uint8), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alpha = (blurred < otsu_thr).astype(np.float32)              # ink = 1
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)                   # soft edges

    px = int(rng.integers(region_x0, max(region_x0 + 1, region_x1 - new_w)))
    py = int(rng.integers(region_y0, max(region_y0 + 1, region_y1 - new_h)))

    roi = color[py:py + new_h, px:px + new_w].astype(np.float32)
    ink_color = np.array([40.0, 25.0, 15.0])                     # dark blue-black BGR
    blended = roi * (1.0 - alpha[..., None]) + ink_color * alpha[..., None]
    color[py:py + new_h, px:px + new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    paste_rect = (px, py, new_w, new_h)
    return color, paste_rect


def main():
    assert CANDIDATE_SRC.exists(), f"Missing candidate {CANDIDATE_SRC}"
    for p in REF_SRCS:
        assert p.exists(), f"Missing ref {p}"

    sig_gray = cv2.imread(str(CANDIDATE_SRC), cv2.IMREAD_GRAYSCALE)
    assert sig_gray is not None, f"Failed to load {CANDIDATE_SRC}"
    print(f"Candidate loaded: {CANDIDATE_SRC} shape={sig_gray.shape} dtype={sig_gray.dtype}")

    rng = np.random.default_rng(stable_seed(f"DEMO-V5-CONFORME-001"))

    cheque, paste_rect = make_moroccan_demo_cheque(sig_gray, rng)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(OUT_PATH), cheque)
    assert ok, "imwrite failed"
    print(f"Saved: {OUT_PATH}")
    print(f"Dimensions: {cheque.shape[1]}x{cheque.shape[0]} (WxH), channels={cheque.shape[2]}")
    print(f"Paste rect (x,y,w,h): {paste_rect}")
    print(f"Cheque number: {CHEQUE_NUMBER}")
    print(f"Candidate source: {CANDIDATE_SRC}")
    # also print hash
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"SHA256: {sha}")
    print(f"File size: {OUT_PATH.stat().st_size} bytes")
    # Quick open test
    reloaded = cv2.imread(str(OUT_PATH))
    assert reloaded is not None
    assert reloaded.shape[0] == 700 and reloaded.shape[1] == 1600
    print("Open check: PASS")

if __name__ == "__main__":
    main()
