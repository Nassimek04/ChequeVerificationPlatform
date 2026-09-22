"""V4 domain-aware augmentation (benchmark-only, training-time).

Only transformations supported by domain-alignment evidence:
- scale 0.92-1.08
- aspect 0.97-1.03
- stroke blur sigma 0.7-1.1 (prob 0.3)
- darken 0.68-0.86 (prob 0.3)

Applied to grayscale image BEFORE shared preprocessing (binarize/crop/fit).
All deterministic via seed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thr, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return blurred < thr


def apply_v4_augmentation(gray: np.ndarray, cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = gray.copy()

    # Scale/aspect perturbation (prob 0.5)
    if rng.random() < cfg.aug_scale_aspect_prob:
        scale = rng.uniform(cfg.aug_scale_min, cfg.aug_scale_max)
        aspect = rng.uniform(cfg.aug_aspect_min, cfg.aug_aspect_max)
        h, w = out.shape
        # Find ink bbox to scale around center
        mask = _ink_mask(out)
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            # Scale/aspect via resize of whole image around center (simpler: resize whole image)
            # We'll resize whole image
            new_w = max(8, int(round(w * scale * aspect)))
            new_h = max(8, int(round(h * scale)))
            resized = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            # Center crop/pad back to original size
            canvas = np.full((h, w), 255, dtype=np.uint8)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            # Clip
            src_y0 = max(0, -y_off)
            src_x0 = max(0, -x_off)
            dst_y0 = max(0, y_off)
            dst_x0 = max(0, x_off)
            rh = min(new_h - src_y0, h - dst_y0)
            rw = min(new_w - src_x0, w - dst_x0)
            canvas[dst_y0:dst_y0+rh, dst_x0:dst_x0+rw] = resized[src_y0:src_y0+rh, src_x0:src_x0+rw]
            out = canvas

    # Stroke blur
    if rng.random() < cfg.aug_blur_prob:
        sigma = rng.uniform(cfg.aug_blur_min, cfg.aug_blur_max)
        out = cv2.GaussianBlur(out, (3, 3), sigma)

    # Darken ink
    if rng.random() < cfg.aug_darken_prob:
        darken = rng.uniform(cfg.aug_darken_min, cfg.aug_darken_max)
        mask = _ink_mask(out)
        # Darken ink pixels: multiply
        out_f = out.astype(np.float32)
        out_f[mask] = np.clip(out_f[mask] * darken, 0, 255)
        # Slight background shift
        bg_shift = int(rng.integers(-4, 5))
        out_f[~mask] = np.clip(out_f[~mask] + bg_shift, 0, 255)
        out = out_f.astype(np.uint8)

    return out
