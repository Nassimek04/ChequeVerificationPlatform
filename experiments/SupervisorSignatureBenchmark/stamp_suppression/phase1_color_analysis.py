"""Phase 1 (read-only): color-separability analysis of stamp vs handwriting."""
import glob
import os

import cv2
import numpy as np

SRC = r"C:\Dev\ChequeVerificationPlatform\experiments\SupervisorSignatureBenchmark\extracted"

for p in sorted(glob.glob(os.path.join(SRC, "sig_*.png"))):
    img = cv2.imread(p)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b = lab[:, :, 2].astype(np.int16) - 128  # negative = blue
    a = lab[:, :, 1].astype(np.int16) - 128
    chroma = np.sqrt(a * a + b * b)
    dark = V < 80
    blue = (H >= 85) & (H <= 130) & (S > 70) & (V >= 80)
    blue_dark = (H >= 85) & (H <= 130) & (S > 70) & (V < 80)
    n = img.shape[0] * img.shape[1]
    sat = S > 70
    hist, _ = np.histogram(H[sat], bins=12, range=(0, 180))
    print(os.path.basename(p), f"px={n}",
          f"dark={dark.mean():.3f}", f"blue={blue.mean():.3f}",
          f"blue_dark={blue_dark.mean():.3f}",
          f"meanLabB_blue={b[blue].mean() if blue.any() else float('nan'):.1f}",
          f"meanLabB_dark={b[dark].mean() if dark.any() else float('nan'):.1f}",
          f"meanChroma_blue={chroma[blue].mean() if blue.any() else float('nan'):.1f}",
          f"meanChroma_dark={chroma[dark].mean() if dark.any() else float('nan'):.1f}")
    print("   hue-hist of saturated px (12 bins over 0-179):", hist.tolist())
