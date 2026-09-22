import os
import sys

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import cv2
import numpy as np
import pytest
import torch

from ai.config import ExperimentConfig
from ai.dataset import (
    Pair,
    PAIR_POSITIVE,
    PAIR_RANDOM_IMPOSTOR,
    PAIR_SKILLED_FORGERY,
    compute_writer_split,
    load_writer_index,
)
from ai.model import build_model


def make_synthetic_signature(path, width=400, height=200, thick=8, seed=0):
    """Deterministic synthetic 'signature': dark strokes on white."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 255, dtype=np.uint8)
    pts = []
    y = rng.uniform(height * 0.3, height * 0.7)
    for x in range(0, width, 6):
        y = np.clip(y + rng.normal(0, 12), height * 0.15, height * 0.85)
        pts.append((int(x), int(y)))
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        cv2.line(img, (x1, int(y1)), (x2, int(y2)), 0, thick)
    cv2.imwrite(str(path), img)
    return img


@pytest.fixture(scope="session")
def tiny_cfg():
    return ExperimentConfig(
        canvas_width=64,
        canvas_height=32,
        embedding_dim=32,
        augment=False,
    )


@pytest.fixture(scope="session")
def index():
    cfg = ExperimentConfig()
    try:
        return load_writer_index(cfg.dataset_root)
    except FileNotFoundError:
        pytest.skip("CEDAR dataset not available")


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("cedar_synthetic")
    for w in (1, 2, 3, 4):
        d = root / str(w)
        d.mkdir()
        for i in range(1, 9):
            make_synthetic_signature(d / f"original_{w}_{i}.png", seed=w * 100 + i)
            make_synthetic_signature(d / f"forgeries_{w}_{i}.png", seed=w * 1000 + i)
    return root


@pytest.fixture(scope="module")
def model(tiny_cfg):
    torch.manual_seed(0)
    m = build_model(tiny_cfg)
    m.eval()
    return m