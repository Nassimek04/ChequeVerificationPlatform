"""V3 configuration.

V3 keeps the V1/V2 backbone (ResNet18, ImageNet pretrained, 128-D L2-normalized,
256x128 preprocessing) and the EXACT V1/V2 writer split. It changes the
MINING (hard-negative candidate bank + scheduled phases), the LOSS margin
(small controlled study), and introduces MULTI-REFERENCE ENROLLMENT as the
primary evaluation protocol with enrollment-only CALIBRATION (z / MAD /
prototype-relative / centroid).

V1 and V2 checkpoints/reports are never written by V3.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from ai.v2.config import (  # noqa: F401  (reuse exact V1/V2 split constants)
    DATASET_ROOT,
    TRAIN_WRITERS,
    VAL_WRITERS,
    TEST_WRITERS,
    GUARANTEE_TEST_WRITER,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    INPUT_CHANNELS,
    AUG_ROTATION_DEG,
    AUG_TRANSLATION_PX,
    AUG_SCALE_MIN,
    AUG_SCALE_MAX,
    AUG_BLUR_PROB,
)

SEED = 42
PREPROCESSING_VERSION = "v1"

# ---------------------------------------------------------------------------
# Model (unchanged from V1/V2)
# ---------------------------------------------------------------------------
BACKBONE = "resnet18"
EMBEDDING_DIM = 128
PRETRAINED = True

# ---------------------------------------------------------------------------
# Triplet loss (small controlled margin study)
# ---------------------------------------------------------------------------
MARGIN_STUDY = (0.2, 0.3, 0.4)
DEFAULT_MARGIN = 0.3

# ---------------------------------------------------------------------------
# Mining (improved: candidate bank + scheduled phases)
# ---------------------------------------------------------------------------
WRITERS_PER_BATCH = 6       # P writers per batch
GENUINES_PER_WRITER = 5     # K genuine samples per writer (in batch)
FORGERIES_PER_WRITER = 8    # M in-batch skilled forgeries per writer
NEGATIVES_PER_ANCHOR = 12   # mined negatives kept per anchor
SKILLED_NEGATIVE_FRACTION = 0.65  # ~65% skilled forgeries / ~35% random impostors
BATCHES_PER_EPOCH = 100
BANK_CAP_PER_WRITER = 64    # max hard forged candidates cached per writer
BANK_REFRESH_EVERY = 2      # re-encode bank images every N epochs
# Schedule (fractions of total epochs)
SCHEDULE_EARLY_FRAC = 0.25  # early: random + semi-hard mix
SCHEDULE_MID_FRAC = 0.60    # mid: raise hard/semi-hard proportion
# late: refresh hardest negatives with current checkpoint

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
# Multi-reference evaluation + calibration
# ---------------------------------------------------------------------------
EVAL_KS = (1, 3, 5)
PRIMARY_K = 5
AGGREGATIONS = ("max", "mean", "median", "top2_mean", "prototype")
CALIBRATIONS = ("raw", "z", "mad", "proto_relative", "centroid")
FRR_LIMITS = (0.10, 0.15, 0.20)   # Policy B security-oriented constraints
MAD_SCALE = 1.4826                # |z| of the median absolute deviation
CALIB_EPS = 1e-6                  # dispersion floor

# ---------------------------------------------------------------------------
# Additional deterministic robustness splits (optional, separate from primary)
# ---------------------------------------------------------------------------
# seed->(train,val,test) writer tuples, generated deterministically, writer 7
# always in test, 38/8/9 balance, disjoint, full coverage.
EXTRA_SPLIT_SEEDS = (101, 202)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AI_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v3.pt"
REPORTS_DIR = AI_DIR / "reports" / "v3"
V1_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "siamese_resnet18_v1.pt"
V2_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v2.pt"
MODEL_VERSION = "ai_metric_v3"


def detect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class V3Config:
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
    augment: bool = True
    aug_rotation_deg: float = AUG_ROTATION_DEG
    aug_translation_px: float = AUG_TRANSLATION_PX
    aug_scale_min: float = AUG_SCALE_MIN
    aug_scale_max: float = AUG_SCALE_MAX
    aug_blur_prob: float = AUG_BLUR_PROB
    margin: float = DEFAULT_MARGIN
    margin_study: tuple = tuple(MARGIN_STUDY)
    writers_per_batch: int = WRITERS_PER_BATCH
    genuines_per_writer: int = GENUINES_PER_WRITER
    forgeries_per_writer: int = FORGERIES_PER_WRITER
    negatives_per_anchor: int = NEGATIVES_PER_ANCHOR
    skilled_negative_fraction: float = SKILLED_NEGATIVE_FRACTION
    batches_per_epoch: int = BATCHES_PER_EPOCH
    bank_cap_per_writer: int = BANK_CAP_PER_WRITER
    bank_refresh_every: int = BANK_REFRESH_EVERY
    schedule_early_frac: float = SCHEDULE_EARLY_FRAC
    schedule_mid_frac: float = SCHEDULE_MID_FRAC
    epochs: int = EPOCHS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    mixed_precision: bool = MIXED_PRECISION
    cuda_deterministic: bool = CUDA_DETERMINISTIC
    eval_ks: tuple = tuple(EVAL_KS)
    primary_k: int = PRIMARY_K
    aggregations: tuple = tuple(AGGREGATIONS)
    calibrations: tuple = tuple(CALIBRATIONS)
    frr_limits: tuple = tuple(FRR_LIMITS)
    mad_scale: float = MAD_SCALE
    calib_eps: float = CALIB_EPS
    extra_split_seeds: tuple = tuple(EXTRA_SPLIT_SEEDS)
    checkpoint_path: Path = CHECKPOINT_PATH
    reports_dir: Path = REPORTS_DIR
    v1_checkpoint_path: Path = V1_CHECKPOINT_PATH
    v2_checkpoint_path: Path = V2_CHECKPOINT_PATH

    @property
    def images_per_batch(self) -> int:
        return self.writers_per_batch * (self.genuines_per_writer + self.forgeries_per_writer)

    def schedule_phase(self, epoch: int) -> str:
        """Return 'early' | 'mid' | 'late' for the given 1-based epoch."""
        e = float(epoch)
        total = float(self.epochs)
        if e <= self.schedule_early_frac * total:
            return "early"
        if e <= self.schedule_mid_frac * total:
            return "mid"
        return "late"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("dataset_root", "checkpoint_path", "reports_dir",
                  "v1_checkpoint_path", "v2_checkpoint_path"):
            d[k] = str(getattr(self, k))
        return d


def ensure_dirs(cfg: V3Config) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")