"""V2 configuration.

The writer split is FROZEN to the exact V1 split (hard-asserted in dataset.py).
V2 keeps the V1 backbone/preprocessing and changes ONLY the metric-learning
objective and the batch/mining design.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

SEED = 42
DATASET_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")

# ---------------------------------------------------------------------------
# EXACT V1 writer split (writer-independent, whole writers). Do not change.
# ---------------------------------------------------------------------------
TRAIN_WRITERS = [
    1, 4, 5, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 49, 50, 51, 52, 54, 55,
]
VAL_WRITERS = [3, 6, 28, 35, 38, 44, 46, 53]
TEST_WRITERS = [2, 7, 8, 9, 15, 16, 18, 41, 48]
GUARANTEE_TEST_WRITER = 7  # writer 7 must remain test-only

# ---------------------------------------------------------------------------
# Preprocessing (REUSED from V1, unchanged)
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 256
CANVAS_HEIGHT = 128
INPUT_CHANNELS = 3
PREPROCESSING_VERSION = "v1"

AUGMENT = True
AUG_ROTATION_DEG = 5.0
AUG_TRANSLATION_PX = 4.0
AUG_SCALE_MIN = 0.95
AUG_SCALE_MAX = 1.05
AUG_BLUR_PROB = 0.0

# ---------------------------------------------------------------------------
# Model (unchanged from V1)
# ---------------------------------------------------------------------------
BACKBONE = "resnet18"
EMBEDDING_DIM = 128
PRETRAINED = True

# ---------------------------------------------------------------------------
# Triplet + mining
# ---------------------------------------------------------------------------
TRIPLET_MARGIN = 0.3        # d(a,p) + margin < d(a,n) ; validated experimentally
WRITERS_PER_BATCH = 6       # P writers sampled per batch
GENUINES_PER_WRITER = 5     # K genuine samples per writer
FORGERIES_PER_WRITER = 8    # M skilled forgeries per writer (>= negatives_per_anchor*skilled_frac)
NEGATIVES_PER_ANCHOR = 12   # mined negatives kept per anchor
SKILLED_NEGATIVE_FRACTION = 0.65  # ~65% skilled forgeries / ~35% random impostors
BATCHES_PER_EPOCH = 100

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
EPOCHS = 20
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
EARLY_STOPPING_PATIENCE = 6
MIXED_PRECISION = True
CUDA_DETERMINISTIC = True

# ---------------------------------------------------------------------------
# Multi-reference evaluation
# ---------------------------------------------------------------------------
EVAL_KS = (1, 3, 5)
AGGREGATIONS = ("max", "mean", "median", "top2_mean", "prototype")
NORMALIZATIONS = ("raw", "z")     # "z" uses enrollment pairwise statistics (K>=3)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AI_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v2.pt"
REPORTS_DIR = AI_DIR / "reports" / "v2"
V1_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "siamese_resnet18_v1.pt"
MODEL_VERSION = "ai_metric_v2"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class V2Config:
    seed: int = SEED
    dataset_root: Path = DATASET_ROOT
    train_writers: tuple = tuple(TRAIN_WRITERS)
    val_writers: tuple = tuple(VAL_WRITERS)
    test_writers: tuple = tuple(TEST_WRITERS)
    guarantee_test_writer: int = GUARANTEE_TEST_WRITER
    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    input_channels: int = INPUT_CHANNELS
    backbone: str = BACKBONE
    embedding_dim: int = EMBEDDING_DIM
    pretrained: bool = PRETRAINED
    model_version: str = MODEL_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    augment: bool = AUGMENT
    aug_rotation_deg: float = AUG_ROTATION_DEG
    aug_translation_px: float = AUG_TRANSLATION_PX
    aug_scale_min: float = AUG_SCALE_MIN
    aug_scale_max: float = AUG_SCALE_MAX
    aug_blur_prob: float = AUG_BLUR_PROB
    margin: float = TRIPLET_MARGIN
    writers_per_batch: int = WRITERS_PER_BATCH
    genuines_per_writer: int = GENUINES_PER_WRITER
    forgeries_per_writer: int = FORGERIES_PER_WRITER
    negatives_per_anchor: int = NEGATIVES_PER_ANCHOR
    skilled_negative_fraction: float = SKILLED_NEGATIVE_FRACTION
    batches_per_epoch: int = BATCHES_PER_EPOCH
    epochs: int = EPOCHS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    mixed_precision: bool = MIXED_PRECISION
    cuda_deterministic: bool = CUDA_DETERMINISTIC
    eval_ks: tuple = tuple(EVAL_KS)
    aggregations: tuple = tuple(AGGREGATIONS)
    normalizations: tuple = tuple(NORMALIZATIONS)
    checkpoint_path: Path = CHECKPOINT_PATH
    reports_dir: Path = REPORTS_DIR
    v1_checkpoint_path: Path = V1_CHECKPOINT_PATH

    @property
    def images_per_batch(self) -> int:
        return self.writers_per_batch * (self.genuines_per_writer + self.forgeries_per_writer)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("dataset_root", "checkpoint_path", "reports_dir", "v1_checkpoint_path"):
            d[k] = str(getattr(self, k))
        return d


def ensure_dirs(cfg: V2Config) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")