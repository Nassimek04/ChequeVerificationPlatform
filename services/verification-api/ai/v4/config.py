"""V4 configuration — domain-aware training.

Keeps V1/V2 backbone (ResNet18, 128-D L2, triplet) and EXACT V1/V2 writer split.
Changes ONLY training domain: V4-A adds supported augmentation (scale/aspect + stroke),
V4-B adds synthetic cheque-domain mixing.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from ai.v2.config import (
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
)

SEED = 42
PREPROCESSING_VERSION = "v1"

BACKBONE = "resnet18"
EMBEDDING_DIM = 128
PRETRAINED = True

# V2 base hyperparams preserved
TRIPLET_MARGIN = 0.3
WRITERS_PER_BATCH = 6
GENUINES_PER_WRITER = 5
FORGERIES_PER_WRITER = 8
NEGATIVES_PER_ANCHOR = 12
SKILLED_NEGATIVE_FRACTION = 0.65
BATCHES_PER_EPOCH = 100

EPOCHS = 20
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
EARLY_STOPPING_PATIENCE = 6
MIXED_PRECISION = True
CUDA_DETERMINISTIC = True

# V4-A augmentation — ONLY supported transforms
# Scale/aspect: from domain alignment T5 (0.92-1.08 scale, 0.97-1.03 aspect)
V4_AUG_SCALE_MIN = 0.92
V4_AUG_SCALE_MAX = 1.08
V4_AUG_ASPECT_MIN = 0.97
V4_AUG_ASPECT_MAX = 1.03
V4_AUG_SCALE_ASPECT_PROB = 0.5  # apply scale/aspect perturbation with 50% prob

# Stroke rendering: blur sigma 0.7-1.1, darken 0.68-0.86
V4_AUG_STROKE_BLUR_MIN = 0.7
V4_AUG_STROKE_BLUR_MAX = 1.1
V4_AUG_STROKE_BLUR_PROB = 0.3
V4_AUG_DARKEN_MIN = 0.68
V4_AUG_DARKEN_MAX = 0.86
V4_AUG_DARKEN_PROB = 0.3

# Keep V2 rotation/translation
V4_AUG_ROTATION_DEG = AUG_ROTATION_DEG
V4_AUG_TRANSLATION_PX = AUG_TRANSLATION_PX
# Preserve V2 scale range as fallback for non-domain-aware? For V4 we override scale range to 0.92-1.08, so V2's 0.95-1.05 is superseded.

# V4-B synthetic domain mixture
V4B_SYNTHETIC_RATIO = 0.5  # 50% clean, 50% synthetic extracted
V4B_SYNTHETIC_SEED_OFFSET = 1000  # offset for deterministic synthetic generation

EVAL_KS = (1, 5)
AGGREGATIONS = ("max", "mean")
NORMALIZATIONS = ("raw",)

AI_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH_A = AI_DIR / "checkpoints" / "metric_resnet18_v4a.pt"
CHECKPOINT_PATH_B = AI_DIR / "checkpoints" / "metric_resnet18_v4b.pt"
REPORTS_DIR = AI_DIR / "reports" / "v4"
V1_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "siamese_resnet18_v1.pt"
V2_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v2.pt"
V3_CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v3.pt"

def detect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclasses.dataclass(frozen=True)
class V4Config:
    variant: str = "a"  # "a" or "b"
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
    model_version: str = "ai_metric_v4a"
    preprocessing_version: str = PREPROCESSING_VERSION
    # augmentation (V4-A)
    aug_rotation_deg: float = V4_AUG_ROTATION_DEG
    aug_translation_px: float = V4_AUG_TRANSLATION_PX
    aug_scale_min: float = V4_AUG_SCALE_MIN
    aug_scale_max: float = V4_AUG_SCALE_MAX
    aug_aspect_min: float = V4_AUG_ASPECT_MIN
    aug_aspect_max: float = V4_AUG_ASPECT_MAX
    aug_scale_aspect_prob: float = V4_AUG_SCALE_ASPECT_PROB
    aug_blur_min: float = V4_AUG_STROKE_BLUR_MIN
    aug_blur_max: float = V4_AUG_STROKE_BLUR_MAX
    aug_blur_prob: float = V4_AUG_STROKE_BLUR_PROB
    aug_darken_min: float = V4_AUG_DARKEN_MIN
    aug_darken_max: float = V4_AUG_DARKEN_MAX
    aug_darken_prob: float = V4_AUG_DARKEN_PROB
    # training
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
    # synthetic
    synthetic_ratio: float = V4B_SYNTHETIC_RATIO
    eval_ks: tuple = tuple(EVAL_KS)
    aggregations: tuple = tuple(AGGREGATIONS)
    normalizations: tuple = tuple(NORMALIZATIONS)
    checkpoint_path: Path = CHECKPOINT_PATH_A
    reports_dir: Path = REPORTS_DIR
    v1_checkpoint_path: Path = V1_CHECKPOINT_PATH
    v2_checkpoint_path: Path = V2_CHECKPOINT_PATH
    v3_checkpoint_path: Path = V3_CHECKPOINT_PATH

    @property
    def images_per_batch(self) -> int:
        return self.writers_per_batch * (self.genuines_per_writer + self.forgeries_per_writer)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("dataset_root","checkpoint_path","reports_dir","v1_checkpoint_path","v2_checkpoint_path","v3_checkpoint_path"):
            d[k]=str(getattr(self,k))
        return d

def get_v4_config(variant: str) -> V4Config:
    variant=variant.lower()
    if variant=="a":
        return V4Config(variant="a", model_version="ai_metric_v4a", checkpoint_path=CHECKPOINT_PATH_A)
    elif variant=="b":
        return V4Config(variant="b", model_version="ai_metric_v4b", checkpoint_path=CHECKPOINT_PATH_B)
    else:
        raise ValueError(f"unknown variant {variant}")

def ensure_dirs(cfg: V4Config) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND","Agg")
