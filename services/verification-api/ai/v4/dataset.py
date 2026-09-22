"""V4 dataset — same split, domain-aware augmentation and synthetic cheque mixing."""

from __future__ import annotations

import random
import re
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from ai.dataset import WriterIndex, load_writer_index
from ai.preprocessing import binarize_ink, crop_to_ink, fit_to_canvas, canvas_to_tensor, _affine_augment
from ai.v2.dataset import WriterSplitMismatch, WriterSamples, build_samples

from .config import V4Config
from .augment import apply_v4_augmentation

# Reuse V2 split assertion but adapt to V4Config
def assert_exact_v1_split(cfg: V4Config) -> None:
    from ai.v2.config import TRAIN_WRITERS, VAL_WRITERS, TEST_WRITERS
    if sorted(cfg.train_writers) != sorted(TRAIN_WRITERS):
        raise WriterSplitMismatch("train writers differ from V1")
    if sorted(cfg.val_writers) != sorted(VAL_WRITERS):
        raise WriterSplitMismatch("val writers differ from V1")
    if sorted(cfg.test_writers) != sorted(TEST_WRITERS):
        raise WriterSplitMismatch("test writers differ from V1")
    st, sv, ste = set(cfg.train_writers), set(cfg.val_writers), set(cfg.test_writers)
    assert st.isdisjoint(sv)
    assert st.isdisjoint(ste)
    assert sv.isdisjoint(ste)
    assert cfg.guarantee_test_writer in ste
    assert st | sv | ste == set(range(1, 56))

def rng_for(seed: int, *parts) -> random.Random:
    return random.Random(":".join(str(p) for p in (seed, *parts)))

def stable_seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))

# Synthetic cheque generation (same as domain_alignment, for V4-B)
def make_synthetic_cheque(sig_gray: np.ndarray, rng: np.random.Generator):
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
        cv2.putText(color, digit, (x, micr_y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.9, 1.3), (30, 30, 30), 2, cv2.LINE_AA)
        x += int(rng.integers(38, 58))
    region_x0, region_x1 = 660, 1520
    region_y0, region_y1 = 290, 590
    target_h = int(rng.integers(130, 195))
    scale = min(target_h / sig_gray.shape[0], (region_x1 - region_x0) / sig_gray.shape[1])
    new_w = max(1, int(round(sig_gray.shape[1] * scale)))
    new_h = max(1, int(round(sig_gray.shape[0] * scale)))
    resized = cv2.resize(sig_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized.astype(np.float32), (3, 3), 0)
    otsu_thr, _ = cv2.threshold(blurred.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

def extract_signature_benchmark(cheque_bgr: np.ndarray):
    # Import here to avoid circular
    from app.services.signature_extraction_service import extract_signature
    res = extract_signature(cheque_bgr, 0.40, 0.40, 0.98, 0.98)
    return res.signature_png_bytes

def preprocess_with_v4_augment(gray: np.ndarray, cfg: V4Config, seed: int, augment: bool) -> torch.Tensor:
    """Preprocess with V4 domain augmentation applied before binarization."""
    if augment:
        # Apply V4 augmentations deterministically
        gray = apply_v4_augmentation(gray, cfg, seed)
    # Shared preprocessing (binarize, crop, fit, affine)
    bin_img = binarize_ink(gray)
    ink = crop_to_ink(bin_img)
    canvas = fit_to_canvas(ink, cfg.canvas_width, cfg.canvas_height)
    if augment:
        # Keep V2 affine augmentation as well
        canvas = _affine_augment(canvas, cfg, seed + 9999)
    return canvas_to_tensor(canvas, cfg)

def get_synthetic_extracted_gray(clean_gray: np.ndarray, seed_key: str) -> np.ndarray:
    """Generate synthetic cheque and extract signature, return grayscale of extracted crop."""
    rng = np.random.default_rng(stable_seed(seed_key))
    cheque, _ = make_synthetic_cheque(clean_gray, rng)
    png_bytes = extract_signature_benchmark(cheque)
    # Decode png_bytes to grayscale
    arr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback to clean
        return clean_gray
    return img
