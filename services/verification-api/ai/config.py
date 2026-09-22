"""Central configuration for the AI signature-verification experiment V1.

Everything hyper-parameter related lives here so that experiments are easy
to reproduce and document. No production code imports this package.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Deterministic writer-split + dataset
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")

SEED = 42

# Target writer proportions. 55 CEDAR writers -> nearest whole-writer split is
# train=38 (69.1%), validation=8 (14.5%), test=9 (16.4%).
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# Writer #7 is guaranteed to land in TEST so the signer-7 secondary benchmark
# is a true unseen-writer evaluation (see section 14 of the experiment brief).
GUARANTEE_TEST_WRITER = 7

# ---------------------------------------------------------------------------
# Preprocessing (shared, deterministic for validation/test)
# ---------------------------------------------------------------------------
CANVAS_WIDTH = 256
CANVAS_HEIGHT = 128
PREPROCESSING_VERSION = "v1"

# Grayscale input is broadcast to 3 channels so the pretrained ImageNet
# ResNet18 first layer can be used unchanged (documented choice).
INPUT_CHANNELS = 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Train-only augmentation (signature-safe, conservative)
# ---------------------------------------------------------------------------
AUGMENT = True
AUG_ROTATION_DEG = 5.0       # uniform(-5, 5) degrees
AUG_TRANSLATION_PX = 4.0     # uniform(-4, 4) px on each axis
AUG_SCALE_MIN = 0.95
AUG_SCALE_MAX = 1.05
AUG_BLUR_PROB = 0.0          # optional very mild blur, disabled by default

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
BACKBONE = "resnet18"
EMBEDDING_DIM = 128
PRETRAINED = True            # torchvision ImageNet V1 weights (verified loadable)
MODEL_VERSION = "siamese_resnet18_v1"

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
# Label semantics: same-writer/genuine pair -> y = 1 ; non-match -> y = 0.
# L = y * d^2 + (1 - y) * max(margin - d, 0)^2   (d = Euclidean distance)
LOSS_MARGIN = 1.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 64
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
TRAIN_PAIRS_PER_EPOCH = 4000   # 50% positive, 25% skilled forgery, 25% impostor
EARLY_STOPPING_PATIENCE = 6    # monitored on validation EER
MIXED_PRECISION = True         # AMP on CUDA
CUDA_DETERMINISTIC = True      # cudnn deterministic, lower variance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AI_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = AI_DIR / "checkpoints"
REPORTS_DIR = AI_DIR / "reports"
CHECKPOINT_PATH = CHECKPOINT_DIR / "siamese_resnet18_v1.pt"


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """Snapshot of every tunable used by one training run."""

    seed: int = SEED
    dataset_root: Path = DEFAULT_DATASET_ROOT
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    guarantee_test_writer: int = GUARANTEE_TEST_WRITER
    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    input_channels: int = INPUT_CHANNELS
    backbone: str = BACKBONE
    embedding_dim: int = EMBEDDING_DIM
    pretrained: bool = PRETRAINED
    model_version: str = MODEL_VERSION
    margin: float = LOSS_MARGIN
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    train_pairs_per_epoch: int = TRAIN_PAIRS_PER_EPOCH
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    mixed_precision: bool = MIXED_PRECISION
    cuda_deterministic: bool = CUDA_DETERMINISTIC
    augment: bool = AUGMENT
    aug_rotation_deg: float = AUG_ROTATION_DEG
    aug_translation_px: float = AUG_TRANSLATION_PX
    aug_scale_min: float = AUG_SCALE_MIN
    aug_scale_max: float = AUG_SCALE_MAX
    aug_blur_prob: float = AUG_BLUR_PROB
    preprocessing_version: str = PREPROCESSING_VERSION
    checkpoint_path: Path = CHECKPOINT_PATH
    reports_dir: Path = REPORTS_DIR

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["dataset_root"] = str(self.dataset_root)
        d["checkpoint_path"] = str(self.checkpoint_path)
        d["reports_dir"] = str(self.reports_dir)
        return d


def ensure_dirs(cfg: ExperimentConfig) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")