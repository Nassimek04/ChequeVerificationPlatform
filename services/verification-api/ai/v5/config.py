"""V5-A configuration — frozen after Phase 2B.

Identity-disjoint, deterministic seed 42. Signer 7 locked.
"""
from __future__ import annotations
import dataclasses
import os
from pathlib import Path
import torch

SEED = 42

# CEDAR root (writer 1..55, 24+24)
CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
# SSBI sources root (signatures/genuine + forged labels)
SSBI_ROOT = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_source_inspect\bank-check-security-1.0.0\data\sources\signatures")

# Frozen splits — DO NOT CHANGE
CEDAR_TRAIN = [1,4,5,10,11,12,13,14,17,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,36,37,39,40,42,43,45,47,49,50,51,52,54,55]
CEDAR_VAL = [3,6,28,35,38,44,46,53]
CEDAR_TEST = [2,7,8,9,15,16,18,41,48]

SSBI_TRAIN = [4,5,6,9,10,11,12,13,15,16,17,18]  # 12: 8 both +4 only
SSBI_VAL = [1,8,19]  # 3: 2 both +1 only
SSBI_TEST = [0,3,14]  # 3: 2 both +1 only
SSBI_LOCKED = [7]

# Preprocessing
CANVAS_WIDTH = 256
CANVAS_HEIGHT = 128
INPUT_CHANNELS = 3
PREPROCESSING_VERSION = "v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation — asymmetric
# REFERENCE light
REF_ROT = 2.0
REF_TRANS = 2.0
# CANDIDATE full V4-A
CAND_ROT = 5.0
CAND_TRANS = 4.0
CAND_SCALE_MIN = 0.92
CAND_SCALE_MAX = 1.08
CAND_ASPECT_MIN = 0.97
CAND_ASPECT_MAX = 1.03
CAND_SCALE_ASPECT_PROB = 0.5
CAND_BLUR_MIN = 0.7
CAND_BLUR_MAX = 1.1
CAND_BLUR_PROB = 0.3
CAND_DARKEN_MIN = 0.68
CAND_DARKEN_MAX = 0.86
CAND_DARKEN_PROB = 0.3
CAND_JPEG_MIN = 70
CAND_JPEG_MAX = 95
CAND_JPEG_PROB = 0.3

# Model
BACKBONE = "resnet18"
EMBEDDING_DIM = 128
PRETRAINED = True
MODEL_VERSION = "ai_metric_v5a"

# Loss
TRIPLET_MARGIN = 0.3
SUPCON_TEMP = 0.07
CE_WEIGHT = 0.2

# Training
P = 6  # writers per batch
K = 5  # genuines per writer
M = 8  # forgeries per writer (capped per writer)
NEGATIVES_PER_ANCHOR = 12
SKILLED_FRAC = 0.65
BATCHES_PER_EPOCH = 100
EPOCHS = 20
LR = 1.0e-4
WEIGHT_DECAY = 1.0e-4
EARLY_STOPPING_PATIENCE = 6
MIXED_PRECISION = True
CUDA_DETERMINISTIC = True

# Eval
EVAL_KS = (1,3,5)
AGGREGATIONS = ("mean",)

AI_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = AI_DIR / "checkpoints" / "metric_resnet18_v5a.pt"
REPORTS_DIR = AI_DIR / "reports" / "v5"
V2_CHECKPOINT = AI_DIR / "checkpoints" / "metric_resnet18_v2.pt"

def detect_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@dataclasses.dataclass(frozen=True)
class V5Config:
    seed: int = SEED
    cedar_root: Path = CEDAR_ROOT
    ssbi_root: Path = SSBI_ROOT
    cedar_train: tuple = tuple(CEDAR_TRAIN)
    cedar_val: tuple = tuple(CEDAR_VAL)
    cedar_test: tuple = tuple(CEDAR_TEST)
    ssbi_train: tuple = tuple(SSBI_TRAIN)
    ssbi_val: tuple = tuple(SSBI_VAL)
    ssbi_test: tuple = tuple(SSBI_TEST)
    ssbi_locked: tuple = tuple(SSBI_LOCKED)
    canvas_width: int = CANVAS_WIDTH
    canvas_height: int = CANVAS_HEIGHT
    input_channels: int = INPUT_CHANNELS
    backbone: str = BACKBONE
    embedding_dim: int = EMBEDDING_DIM
    pretrained: bool = PRETRAINED
    model_version: str = MODEL_VERSION
    ref_rot: float = REF_ROT
    ref_trans: float = REF_TRANS
    cand_rot: float = CAND_ROT
    cand_trans: float = CAND_TRANS
    cand_scale_min: float = CAND_SCALE_MIN
    cand_scale_max: float = CAND_SCALE_MAX
    cand_aspect_min: float = CAND_ASPECT_MIN
    cand_aspect_max: float = CAND_ASPECT_MAX
    cand_scale_aspect_prob: float = CAND_SCALE_ASPECT_PROB
    cand_blur_min: float = CAND_BLUR_MIN
    cand_blur_max: float = CAND_BLUR_MAX
    cand_blur_prob: float = CAND_BLUR_PROB
    cand_darken_min: float = CAND_DARKEN_MIN
    cand_darken_max: float = CAND_DARKEN_MAX
    cand_darken_prob: float = CAND_DARKEN_PROB
    cand_jpeg_min: int = CAND_JPEG_MIN
    cand_jpeg_max: int = CAND_JPEG_MAX
    cand_jpeg_prob: float = CAND_JPEG_PROB
    triplet_margin: float = TRIPLET_MARGIN
    supcon_temp: float = SUPCON_TEMP
    ce_weight: float = CE_WEIGHT
    writers_per_batch: int = P
    genuines_per_writer: int = K
    forgeries_per_writer: int = M
    negatives_per_anchor: int = NEGATIVES_PER_ANCHOR
    skilled_negative_fraction: float = SKILLED_FRAC
    batches_per_epoch: int = BATCHES_PER_EPOCH
    epochs: int = EPOCHS
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    mixed_precision: bool = MIXED_PRECISION
    cuda_deterministic: bool = CUDA_DETERMINISTIC
    eval_ks: tuple = tuple(EVAL_KS)
    checkpoint_path: Path = CHECKPOINT_PATH
    reports_dir: Path = REPORTS_DIR
    v2_checkpoint: Path = V2_CHECKPOINT

    def to_dict(self):
        d = dataclasses.asdict(self)
        for k in ("cedar_root","ssbi_root","checkpoint_path","reports_dir","v2_checkpoint"):
            d[k]=str(getattr(self,k))
        return d

def ensure_dirs(cfg: V5Config) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND","Agg")
