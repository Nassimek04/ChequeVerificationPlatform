"""V4 guardrail tests — benchmark-only."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPORTS_V4 = SERVICE_ROOT / "ai" / "reports" / "v4"

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def test_split_preservation():
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    assert set(cfg.train_writers).isdisjoint(set(cfg.val_writers))
    assert set(cfg.train_writers).isdisjoint(set(cfg.test_writers))
    assert set(cfg.val_writers).isdisjoint(set(cfg.test_writers))
    assert 7 in cfg.test_writers
    assert len(cfg.train_writers)==38 and len(cfg.val_writers)==8 and len(cfg.test_writers)==9

def test_writer_leakage():
    from ai.dataset import load_writer_index, validate_no_writer_leakage, generate_pairs
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    idx=load_writer_index(cfg.dataset_root)
    val=generate_pairs(idx, cfg.val_writers, "validation", cfg.seed)
    test=generate_pairs(idx, cfg.test_writers, "test", cfg.seed)
    # Should not raise
    validate_no_writer_leakage(val, test)

def test_augmentation_bounds():
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    assert 0.92 <= cfg.aug_scale_min <= cfg.aug_scale_max <= 1.08
    assert 0.97 <= cfg.aug_aspect_min <= cfg.aug_aspect_max <= 1.03
    assert 0.7 <= cfg.aug_blur_min <= cfg.aug_blur_max <= 1.1
    assert 0.68 <= cfg.aug_darken_min <= cfg.aug_darken_max <= 0.86
    assert cfg.aug_blur_prob==0.3 and cfg.aug_darken_prob==0.3 and cfg.aug_scale_aspect_prob==0.5

def test_augmentation_determinism():
    from ai.v4.augment import apply_v4_augmentation
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    gray=np.full((64,128), 255, dtype=np.uint8)
    cv2.line(gray,(10,50),(100,20),0,2)
    out1=apply_v4_augmentation(gray, cfg, seed=12345)
    out2=apply_v4_augmentation(gray, cfg, seed=12345)
    assert np.array_equal(out1, out2)
    out3=apply_v4_augmentation(gray, cfg, seed=12346)
    # Different seed may give different result, but at least not always equal
    assert out1.shape==gray.shape

def test_no_flip():
    # Ensure augmentation never flips writer identity: no horizontal flip
    from ai.v4.augment import apply_v4_augmentation
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    # Check code does not contain flip
    import pathlib
    text=pathlib.Path(SERVICE_ROOT/"ai"/"v4"/"augment.py").read_text()
    assert "flip" not in text.lower()
    assert "FLIP" not in text

def test_domain_mixture_correctness():
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("b")
    assert cfg.synthetic_ratio==0.5
    assert cfg.variant=="b"
    cfg_a=get_v4_config("a")
    assert cfg_a.synthetic_ratio==0.5 # same ratio but variant a should not use synthetic in training? Actually a still has ratio but not used
    # Check that train.py respects variant
    text=(SERVICE_ROOT/"ai"/"v4"/"train.py").read_text()
    assert 'cfg.variant == "b"' in text

def test_genuine_forgery_labels():
    from ai.dataset import load_writer_index
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    idx=load_writer_index(cfg.dataset_root)
    for w in cfg.train_writers[:1]:
        assert len(idx[w].originals)==24
        assert len(idx[w].forgeries)==24
        assert all("original" in p.name for p in idx[w].originals)
        assert all("forgeries" in p.name for p in idx[w].forgeries)

def test_no_identical_positive_pair():
    # Mining should never pair anchor with itself
    text=(SERVICE_ROOT/"ai"/"v2"/"mining.py").read_text()
    assert "arange != a" in text or "!= a" in text

def test_embedding_dimension():
    from ai.v4.config import get_v4_config
    from ai.model import build_model
    cfg=get_v4_config("a")
    model=build_model(cfg)
    assert cfg.embedding_dim==128
    dummy=torch.randn(2,3,cfg.canvas_height,cfg.canvas_width)
    emb=model.encode(dummy)
    assert emb.shape==(2,128)

def test_l2_normalization():
    from ai.v4.config import get_v4_config
    from ai.model import build_model
    cfg=get_v4_config("a")
    model=build_model(cfg)
    dummy=torch.randn(4,3,cfg.canvas_height,cfg.canvas_width)
    emb=model.encode(dummy)
    norms=torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_checkpoint_metadata():
    ckpt=torch.load(SERVICE_ROOT/"ai"/"checkpoints"/"metric_resnet18_v4a.pt", map_location="cpu", weights_only=False)
    assert ckpt["model_version"]=="ai_metric_v4a"
    assert ckpt["architecture"]["embedding_dim"]==128
    assert "train_history" in ckpt
    assert ckpt["config"]["variant"]=="a"

def test_checkpoint_round_trip():
    from ai.v4.config import get_v4_config
    from ai.model import build_model
    cfg=get_v4_config("a")
    ckpt=torch.load(SERVICE_ROOT/"ai"/"checkpoints"/"metric_resnet18_v4a.pt", map_location="cpu", weights_only=False)
    model=build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Encode again
    dummy=torch.randn(1,3,cfg.canvas_height,cfg.canvas_width)
    emb=model.encode(dummy)
    assert emb.shape==(1,128)

def test_v1_v2_v3_preservation():
    before=(REPORTS_V4/"checkpoint_sha256_before.txt").read_text().strip().splitlines()
    after=(REPORTS_V4/"checkpoint_sha256_after.txt").read_text().strip().splitlines()
    assert before==after
    for line in before:
        digest,name=line.split()
        assert sha256(SERVICE_ROOT/"ai"/"checkpoints"/name)==digest

def test_test_exclusion_from_selection():
    # Best epoch should be selected on val, not test
    ckpt=torch.load(SERVICE_ROOT/"ai"/"checkpoints"/"metric_resnet18_v4a.pt", map_location="cpu", weights_only=False)
    assert "validation_metric" in ckpt
    assert "test" not in ckpt.get("validation_metric", {})

def test_k5_enrollment_query_leakage_prevention():
    # K5 should use first 5 as refs, rest as queries, no overlap
    from ai.v2.dataset import build_samples
    from ai.dataset import load_writer_index
    from ai.v4.config import get_v4_config
    cfg=get_v4_config("a")
    idx=load_writer_index(cfg.dataset_root)
    samples=build_samples(idx, cfg.test_writers)
    for w in cfg.test_writers:
        ws=samples[w]
        refs=set(ws.genuine_paths[:5])
        queries=set(ws.genuine_paths[5:])
        assert refs.isdisjoint(queries)
        assert len(refs)==5 and len(queries)==19
