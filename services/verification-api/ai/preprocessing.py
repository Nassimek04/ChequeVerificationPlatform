"""Shared, deterministic signature preprocessing + train-only augmentation.

One implementation is used by training AND evaluation (no train/test drift).

Pipeline (identical for every image):
  1. decode (grayscale)
  2. light blur (3x3)
  3. Otsu binarization + THRESH_BINARY_INV  -> ink (foreground) becomes 255
  4. crop to the ink bounding box
  5. resize preserving aspect ratio (NO stretching)
  6. center-pad to a fixed W x H canvas

Augmentation is applied AFTER the canvas is built, ONLY during training
(constructive random rotation ~±5deg, small translation, mild scale). It never
changes writer identity (no flips, no large rotation, no perspective warp).

The returned tensor is [C, H, W] float32 in [0, 1] (ink high), broadcast to 3
channels (choice: keep the pretrained ImageNet ResNet18 conv1 untouched) and
normalized with ImageNet statistics.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ExperimentConfig,
)

_INK = 255.0


def read_gray(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return img


def binarize_ink(img: np.ndarray) -> np.ndarray:
    """Light blur + Otsu + THRESH_BINARY_INV -> ink = 255 (foreground)."""
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(bin_img)


def crop_to_ink(bin_img: np.ndarray) -> np.ndarray:
    ys, xs = np.where(bin_img > 0)
    if ys.size == 0:
        raise ValueError("no ink content found in image")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return bin_img[y0:y1, x0:x1]


def fit_to_canvas(ink: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-ratio-preserving resize + center-pad onto a fixed canvas.

    The ink is scaled to fit *inside* the canvas (max scale = min(1) of the
    axis ratios) and centered; nothing is ever stretched.
    """
    h, w = ink.shape
    scale = min(width / w, height / h, 1.0)  # never upscale beyond canvas
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(ink, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((height, width), dtype=np.float32)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized / _INK
    return canvas


def _affine_augment(canvas: np.ndarray, cfg: ExperimentConfig, seed: int) -> np.ndarray:
    """Signature-safe affine augmentation on the float canvas (0=bg, 1=ink)."""
    rng = np.random.default_rng(seed)
    h, w = canvas.shape
    angle = rng.uniform(-cfg.aug_rotation_deg, cfg.aug_rotation_deg)
    tx = rng.uniform(-cfg.aug_translation_px, cfg.aug_translation_px)
    ty = rng.uniform(-cfg.aug_translation_px, cfg.aug_translation_px)
    scale = rng.uniform(cfg.aug_scale_min, cfg.aug_scale_max)

    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty

    aug = cv2.warpAffine(
        canvas,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    if cfg.aug_blur_prob > 0.0 and rng.uniform() < cfg.aug_blur_prob:
        ksize = 3
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)
    return aug


def canvas_to_tensor(canvas: np.ndarray, cfg: ExperimentConfig) -> torch.Tensor:
    """[H,W] float canvas (0..1, ink high) -> [3,H,W] ImageNet-normalized tensor."""
    canvas = np.clip(canvas, 0.0, 1.0)
    gray = torch.from_numpy(canvas).float().unsqueeze(0)  # [1,H,W]
    rgb = gray.repeat(cfg.input_channels, 1, 1)           # [3,H,W]
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (rgb - mean) / std


def preprocess(
    path,
    cfg: ExperimentConfig,
    augment: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Full shared preprocessing -> [C,H,W] tensor.

    Deterministic unless `augment` is True (train only).
    """
    gray = read_gray(path)
    bin_img = binarize_ink(gray)
    ink = crop_to_ink(bin_img)
    canvas = fit_to_canvas(ink, cfg.canvas_width, cfg.canvas_height)
    if augment:
        canvas = _affine_augment(canvas, cfg, seed)
    return canvas_to_tensor(canvas, cfg)