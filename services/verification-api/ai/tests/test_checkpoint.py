import torch

from ai.model import build_model


def test_checkpoint_save_load_roundtrip(model, tiny_cfg, tmp_path):
    ckpt_path = tmp_path / "test.pt"
    torch.save(
        {
            "model_version": tiny_cfg.model_version,
            "config": tiny_cfg.to_dict(),
            "model_state": model.state_dict(),
            "training_epoch": 3,
            "validation_metric": {"eer": 0.25},
            "threshold_derived_from_validation": 0.7,
            "preprocessing": {"version": tiny_cfg.preprocessing_version},
        },
        ckpt_path,
    )
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["training_epoch"] == 3
    assert ckpt["validation_metric"]["eer"] == 0.25
    assert ckpt["threshold_derived_from_validation"] == 0.7
    assert ckpt["config"]["model_version"] == tiny_cfg.model_version

    m2 = build_model(tiny_cfg)
    m2.load_state_dict(ckpt["model_state"])
    m2.eval()
    model.eval()

    x = torch.randn(4, 3, 32, 64)
    with torch.inference_mode():
        a = model.encode(x)
        b = m2.encode(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_contains_required_metadata(model, tiny_cfg, tmp_path):
    ckpt_path = tmp_path / "meta.pt"
    torch.save(
        {
            "architecture": {"backbone": tiny_cfg.backbone, "embedding_dim": tiny_cfg.embedding_dim},
            "canvas_size": {"width": tiny_cfg.canvas_width, "height": tiny_cfg.canvas_height},
            "training_epoch": 5,
            "validation_metric": {"eer": 0.1, "roc_auc": 0.9, "threshold": 0.8},
            "threshold_derived_from_validation": 0.8,
            "preprocessing": {"version": tiny_cfg.preprocessing_version, "config": tiny_cfg.to_dict()},
            "model_version": tiny_cfg.model_version,
            "model_state": model.state_dict(),
        },
        ckpt_path,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in (
        "architecture",
        "canvas_size",
        "training_epoch",
        "validation_metric",
        "threshold_derived_from_validation",
        "preprocessing",
        "model_version",
        "model_state",
    ):
        assert key in ckpt
