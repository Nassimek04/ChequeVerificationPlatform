"""Phase 2 (experiment-only): deterministic blue-stamp suppression variants.

Reads the EXISTING benchmark crops (never overwrites them) and writes, per
cheque, into stamp_suppression/<sid>/ :
  original_crop.png   - verbatim copy of the benchmark crop (traceability)
  mask_A.png          - HSV blue/cyan stamp mask (white = stamp)
  candidate_A.png     - crop with mask_A pixels whitened
  mask_B.png          - Lab-chromatic stamp mask
  candidate_B.png     - crop with mask_B pixels whitened
  mask_C.png          - union(A,B) on non-dark pixels (primary variant)
  candidate_C.png     - FINAL handwriting candidate (primary)
  mask_D.png / candidate_D.png - C + 1px dilation (stamp edge catch)

Rules (all measurements from phase 1):
  - dark pixels (V<80) are ALWAYS kept: handwriting + printed black survive
    color suppression by construction (nothing invented, nothing destroyed);
  - only saturated blue non-dark pixels are whitened.

No production code is touched; outputs stay inside stamp_suppression/.
"""
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "extracted"
OUT_DIR = HERE

SIDS = ["sig_0001", "sig_0003", "sig_0005", "sig_0007",
        "sig_0009", "sig_0011", "sig_0013"]


def masks(bgr: np.ndarray):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b = lab[:, :, 2].astype(np.int16) - 128
    a = lab[:, :, 1].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b)
    dark = V < 80
    mask_a = (H >= 85) & (H <= 130) & (S > 70) & (~dark)
    mask_b = (b <= -20) & (chroma >= 25) & (~dark)
    mask_c = (mask_a | mask_b)
    k = np.ones((3, 3), np.uint8)
    mask_d = cv2.dilate(mask_c.astype(np.uint8), k, iterations=1).astype(bool)
    return {"A": mask_a, "B": mask_b, "C": mask_c, "D": mask_d}, dark


def apply(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    out[mask] = (255, 255, 255)
    return out


def main() -> None:
    for sid in SIDS:
        src = SRC_DIR / f"{sid}.png"
        if not src.exists():
            print(f"{sid}: MISSING source crop, skipped")
            continue
        d = OUT_DIR / sid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, d / "original_crop.png")
        bgr = cv2.imread(str(src))
        ms, dark = masks(bgr)
        stats = []
        for v in ("A", "B", "C", "D"):
            m = ms[v]
            cv2.imwrite(str(d / f"mask_{v}.png"), (m.astype(np.uint8)) * 255)
            cv2.imwrite(str(d / f"candidate_{v}.png"), apply(bgr, m))
            stats.append(f"{v}={m.mean():.3f}")
        print(f"{sid}: dark_kept={dark.mean():.3f} " + " ".join(stats))


if __name__ == "__main__":
    main()
