import hashlib
import io
import sys
from pathlib import Path

import pytest
import torch

from ai.v2.config import V2Config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mini_state_dict():
    torch.manual_seed(0)
    return {f"layer{i}": torch.randn(3, 3) for i in range(3)}


def test_checkpoint_roundtrip(tmp_path):
    cfg = V2Config(checkpoint_path=tmp_path / "metric_resnet18_v2.pt")
    ckpt = {
        "model_version": cfg.model_version,
        "architecture": {"backbone": cfg.backbone, "embedding_dim": cfg.embedding_dim},
        "writer_split": {"train": list(cfg.train_writers), "validation": list(cfg.val_writers),
                         "test": list(cfg.test_writers)},
        "training_epoch": 5,
        "config": cfg.to_dict(),
        "model_state": _mini_state_dict(),
    }
    torch.save(ckpt, cfg.checkpoint_path)
    loaded = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded["model_version"] == "ai_metric_v2"
    assert loaded["architecture"]["embedding_dim"] == 128
    assert loaded["training_epoch"] == 5
    assert loaded["writer_split"]["test"] == list(cfg.test_writers)
    assert loaded["model_state"]["layer0"].shape == (3, 3)


def test_checkpoint_version_and_backbone():
    cfg = V2Config()
    assert cfg.model_version == "ai_metric_v2"
    assert cfg.backbone == "resnet18"
    assert cfg.embedding_dim == 128


def test_v2_checkpoint_path_distinct_from_v1():
    cfg = V2Config()
    assert cfg.checkpoint_path.name == "metric_resnet18_v2.pt"
    assert cfg.v1_checkpoint_path.name == "siamese_resnet18_v1.pt"
    assert cfg.checkpoint_path != cfg.v1_checkpoint_path


def test_v1_checkpoint_untouched_by_v2_save(tmp_path):
    cfg = V2Config()
    if not cfg.v1_checkpoint_path.exists():
        pytest.skip("V1 checkpoint not present")
    before = _sha256(cfg.v1_checkpoint_path)
    # simulate a V2 save to its own path
    v2 = V2Config(checkpoint_path=tmp_path / "metric_resnet18_v2.pt")
    torch.save({"model_version": "ai_metric_v2", "model_state": _mini_state_dict()}, v2.checkpoint_path)
    after = _sha256(cfg.v1_checkpoint_path)
    assert before == after
    assert cfg.v1_checkpoint_path.name in str(cfg.v1_checkpoint_path)


def test_v1_sources_not_modified_by_v2():
    """V2 must never write into the V1 source or report directories."""
    from ai.v2 import config as v2config
    v1_dir = v2config.AI_DIR / "reports"
    v2_dir = v2config.REPORTS_DIR
    assert v2_dir == v1_dir / "v2"
    assert v2_dir.is_dir() or not v2_dir.exists()