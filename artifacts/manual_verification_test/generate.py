"""Generate controlled genuine vs forgery cheque test images.
Reuses make_synthetic_cheque from ai/reports/v2/end_to_end/run_end_to_end_benchmark.py
"""
import hashlib
import zlib
from pathlib import Path

import cv2
import numpy as np

# Paths
CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
OUT_DIR = Path(__file__).parent
GENUINE_SRC = CEDAR_ROOT / "7" / "original_7_12.png"
FORGERY_SRC = CEDAR_ROOT / "7" / "forgeries_7_12.png"
REFERENCE_SRC = CEDAR_ROOT / "7" / "original_7_5.png"

GENUINE_CHEQUE = OUT_DIR / "CHQ-GENUINE-TEST.png"
FORGERY_CHEQUE = OUT_DIR / "CHQ-FORGERY-TEST.png"

def stable_seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))

def make_synthetic_cheque(sig_gray: np.ndarray, rng: np.random.Generator):
    """Exact copy from ai/reports/v2/end_to_end/run_end_to_end_benchmark.py"""
    W, H = 1600, 700
    base = rng.normal(244.0, 2.5, (H, W)).astype(np.float32)
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
    cv2.rectangle(color, (18, 18), (W - 18, H - 18), (140, 140, 140), 1)
    cv2.line(color, (60, 70), (420, 70), (150, 150, 150), 1)
    cv2.line(color, (60, 620 - 320), (900, 300), (150, 150, 150), 1)
    cv2.rectangle(color, (1050, 90), (1540, 170), (150, 150, 150), 1)
    for k in range(6):
        p1 = (1070 + k * 75, 155 - int(rng.integers(10, 55)))
        p2 = (1120 + k * 75, 105 + int(rng.integers(5, 45)))
        cv2.line(color, p1, p2, (120, 120, 120), 1)
    micr_y = 655
    x = 120
    while x < W - 160:
        digit = str(int(rng.integers(0, 10)))
        cv2.putText(color, digit, (x, micr_y), cv2.FONT_HERSHEY_SIMPLEX,
                    rng.uniform(0.9, 1.3), (30, 30, 30), 2, cv2.LINE_AA)
        x += int(rng.integers(38, 58))
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
    alpha = (blurred < otsu_thr).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    px = int(rng.integers(region_x0, max(region_x0 + 1, region_x1 - new_w)))
    py = int(rng.integers(region_y0, max(region_y0 + 1, region_y1 - new_h)))
    roi = color[py:py + new_h, px:px + new_w].astype(np.float32)
    ink_color = np.array([40.0, 25.0, 15.0])
    blended = roi * (1.0 - alpha[..., None]) + ink_color * alpha[..., None]
    color[py:py + new_h, px:px + new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    paste_rect = (px, py, new_w, new_h)
    return color, paste_rect

def main():
    # Use same seed for both to ensure same background/placement algorithm; separate RNG instances
    seed = stable_seed("controlled-genuine-vs-forgery-v1")
    # Load signatures
    genuine_gray = cv2.imread(str(GENUINE_SRC), cv2.IMREAD_GRAYSCALE)
    forgery_gray = cv2.imread(str(FORGERY_SRC), cv2.IMREAD_GRAYSCALE)
    reference_gray = cv2.imread(str(REFERENCE_SRC), cv2.IMREAD_GRAYSCALE)
    assert genuine_gray is not None, f"Failed to load {GENUINE_SRC}"
    assert forgery_gray is not None, f"Failed to load {FORGERY_SRC}"
    assert reference_gray is not None

    # Ensure reference not used
    assert GENUINE_SRC.name != REFERENCE_SRC.name
    assert FORGERY_SRC.name != REFERENCE_SRC.name

    # Generate cheques with same seed (separate RNGs)
    rng_g = np.random.default_rng(seed)
    rng_f = np.random.default_rng(seed)
    genuine_cheque, rect_g = make_synthetic_cheque(genuine_gray, rng_g)
    forgery_cheque, rect_f = make_synthetic_cheque(forgery_gray, rng_f)

    # Write
    cv2.imwrite(str(GENUINE_CHEQUE), genuine_cheque)
    cv2.imwrite(str(FORGERY_CHEQUE), forgery_cheque)

    # Validate
    for p in [GENUINE_CHEQUE, FORGERY_CHEQUE]:
        assert p.exists()
        img = cv2.imread(str(p))
        assert img is not None
        assert img.shape[0] == 700 and img.shape[1] == 1600

    # Same dimensions check
    g_img = cv2.imread(str(GENUINE_CHEQUE))
    f_img = cv2.imread(str(FORGERY_CHEQUE))
    assert g_img.shape == f_img.shape

    # SHA256
    def sha256(p: Path):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"GENUINE_CHEQUE: {GENUINE_CHEQUE}")
    print(f"  source: {GENUINE_SRC.name}")
    print(f"  dimensions: {g_img.shape[1]}x{g_img.shape[0]}")
    print(f"  SHA256: {sha256(GENUINE_CHEQUE)}")
    print(f"  paste_rect: {rect_g}")
    print(f"FORGERY_CHEQUE: {FORGERY_CHEQUE}")
    print(f"  source: {FORGERY_SRC.name}")
    print(f"  dimensions: {f_img.shape[1]}x{f_img.shape[0]}")
    print(f"  SHA256: {sha256(FORGERY_CHEQUE)}")
    print(f"  paste_rect: {rect_f}")
    print(f"REFERENCE_KEPT_INDEPENDENT: {REFERENCE_SRC.name} not used: {GENUINE_CHEQUE.name != REFERENCE_SRC.name and FORGERY_CHEQUE.name != REFERENCE_SRC.name}")
    # Ensure not byte-identical to reference cheque (would be same paste? no)
    print(f"SAME_TEMPLATE: YES")
    print(f"SAME_DIMENSIONS: YES")
    print(f"SAME_PLACEMENT_ALGORITHM: YES")

    # Dry-run extraction if service available
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "services" / "verification-api"))
        from app.services.signature_extraction_service import extract_signature
        for name, path in [("Genuine", GENUINE_CHEQUE), ("Forgery", FORGERY_CHEQUE)]:
            img = cv2.imread(str(path))
            res = extract_signature(img, 0.40, 0.40, 0.98, 0.98)
            print(f"EXTRACTION {name}: success={res.success} crop={res.signature_bbox.width if res.signature_bbox else 'None'}x{res.signature_bbox.height if res.signature_bbox else 'None'} quality={res.extraction_quality:.4f} paste_rect={rect_g if name=='Genuine' else rect_f}")
    except Exception as e:
        print(f"EXTRACTION DRY-RUN NOT RUN: {e}")

if __name__ == "__main__":
    main()
