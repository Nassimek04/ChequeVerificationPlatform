import pytest
import torch

from ai.v3.config import V3Config
from ai.v3.model import build_model
from ai.v3 import train as T


def test_backbone_unchanged_from_v1_v2():
    cfg = V3Config()
    assert cfg.backbone == "resnet18"
    assert cfg.embedding_dim == 128
    assert cfg.pretrained is True
    assert cfg.canvas_width == 256 and cfg.canvas_height == 128
    assert cfg.model_version == "ai_metric_v3"


def test_model_encode_shapes(v3_cfg):
    model = build_model(v3_cfg)
    model.eval()
    with torch.no_grad():
        e = model.encode(torch.zeros(4, 3, v3_cfg.canvas_height, v3_cfg.canvas_width))
    assert e.shape == (4, v3_cfg.embedding_dim)
    assert torch.allclose(e.norm(dim=1), torch.ones(4), atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    cfg = V3Config()
    model = build_model(cfg)
    path = tmp_path / "ck.pt"
    torch.save({
        "model_version": cfg.model_version,
        "margin": 0.3,
        "model_state": model.state_dict(),
        "best_epoch": 5,
        "history": [{"epoch": 1, "val_eer": 0.1}],
        "config": cfg.to_dict(),
    }, path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["model_version"] == "ai_metric_v3"
    assert ck["margin"] == 0.3
    m2 = build_model(cfg)
    m2.load_state_dict(ck["model_state"])
    assert ck["best_epoch"] == 5


def test_v3_config_serialization_roundtrip(v3_cfg):
    d = v3_cfg.to_dict()
    cfg2 = V3Config(**{k: v for k, v in d.items() if k in V3Config.__dataclass_fields__})
    # Path fields round-tripped as strings -> normalize back for comparison
    import pathlib
    for k in ("dataset_root", "checkpoint_path", "reports_dir", "v1_checkpoint_path", "v2_checkpoint_path"):
        assert isinstance(d[k], str)
        assert pathlib.Path(d[k]) == getattr(v3_cfg, k)
    assert cfg2.seed == v3_cfg.seed
    assert cfg2.margin == v3_cfg.margin


def test_margin_study_ordered_and_default():
    cfg = V3Config()
    assert cfg.margin_study == (0.2, 0.3, 0.4)
    assert cfg.margin == 0.3


def test_validation_objective_ordering():
    """Smaller EER must sort first in the checkpoint-selection objective."""
    scores = [0.9, 0.9, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    skilled = [0.3, 0.25]
    o1 = T.validation_objective(
        __import__("numpy").array(scores + [0.8, 0.8, 0.1, 0.05]),
        __import__("numpy").array(labels + [1, 1, 0, 0]),
        __import__("numpy").array(skilled + [0.3, 0.25]),
    )
    o2 = T.validation_objective(
        __import__("numpy").array(scores + [0.8, 0.8, 0.1, 0.05]),
        __import__("numpy").array(labels + [1, 1, 0, 0]),
        __import__("numpy").array([0.9, 0.9]),
    )
    assert (o1[0], o1[1]) < (o2[0], o2[1]) or o1[0] == o2[0]