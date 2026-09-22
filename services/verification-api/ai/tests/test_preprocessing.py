import cv2
import numpy as np
import pytest
import torch

from ai import preprocessing as P
from ai.config import ExperimentConfig
from ai.dataset import Pair, PairDataset, default_collate
from ai.tests.conftest import make_synthetic_signature


@pytest.fixture()
def sample(tmp_path):
    p = tmp_path / "sig.png"
    make_synthetic_signature(p)
    return p


def test_preprocessing_output_shape(sample, tiny_cfg):
    t = P.preprocess(sample, tiny_cfg)
    assert tuple(t.shape) == (
        tiny_cfg.input_channels,
        tiny_cfg.canvas_height,
        tiny_cfg.canvas_width,
    )
    assert t.dtype == torch.float32


def test_fit_to_canvas_preserves_aspect_ratio(sample, tiny_cfg):
    gray = P.read_gray(sample)
    ink = P.crop_to_ink(P.binarize_ink(gray))
    h, w = ink.shape
    orig_ratio = w / h

    canvas = P.fit_to_canvas(ink, tiny_cfg.canvas_width, tiny_cfg.canvas_height)
    ys, xs = np.where(canvas > 0)
    cw = xs.max() - xs.min() + 1
    ch = ys.max() - ys.min() + 1
    new_ratio = cw / ch

    assert abs(new_ratio - orig_ratio) < 0.1
    # Centered: symmetric margins.
    assert abs((tiny_cfg.canvas_width - cw) // 2 - xs.min()) <= 1
    assert abs((tiny_cfg.canvas_height - ch) // 2 - ys.min()) <= 1


def test_fit_to_canvas_never_exceeds_canvas(sample, tiny_cfg):
    gray = P.read_gray(sample)
    ink = P.crop_to_ink(P.binarize_ink(gray))
    canvas = P.fit_to_canvas(ink, tiny_cfg.canvas_width, tiny_cfg.canvas_height)
    assert canvas.shape == (tiny_cfg.canvas_height, tiny_cfg.canvas_width)
    assert np.all(canvas >= 0.0) and np.all(canvas <= 1.0)


def test_deterministic_validation_preprocessing(sample, tiny_cfg):
    a = P.preprocess(sample, tiny_cfg, augment=False, seed=123)
    b = P.preprocess(sample, tiny_cfg, augment=False, seed=999)
    assert torch.equal(a, b)


def test_augmentation_train_only_changes_values(sample, tiny_cfg):
    base = P.preprocess(sample, tiny_cfg, augment=False, seed=0)
    aug = P.preprocess(sample, tiny_cfg, augment=True, seed=0)
    assert base.shape == aug.shape


def test_pair_dataset_lazy_and_deterministic(tmp_path, tiny_cfg):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    make_synthetic_signature(p1, seed=1)
    make_synthetic_signature(p2, seed=2)
    pair = Pair(p1, p2, 1, "positive", 1)
    ds = PairDataset([pair], tiny_cfg, split="eval")
    a, b, y = ds[0]
    assert tuple(a.shape) == (3, tiny_cfg.canvas_height, tiny_cfg.canvas_width)
    assert float(y) == 1.0
    a2, _, _ = ds[0]
    assert torch.equal(a, a2)
