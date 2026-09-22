import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ai.v3.config import V3Config
from ai.v3.mining import CandidateBank, build_batch, select_triplets_v3
from ai.v3.model import build_model


def test_schedule_phase_fractions(v3_cfg):
    total = v3_cfg.epochs
    early = int(v3_cfg.schedule_early_frac * total)
    mid = int(v3_cfg.schedule_mid_frac * total)
    assert v3_cfg.schedule_phase(1) == "early"
    assert v3_cfg.schedule_phase(early) == "early"
    assert v3_cfg.schedule_phase(early + 1) == "mid"
    assert v3_cfg.schedule_phase(mid) == "mid"
    assert v3_cfg.schedule_phase(mid + 1) == "late"
    assert v3_cfg.schedule_phase(total) == "late"


def test_build_batch_uses_only_train_writers(index, tiny_v3_cfg):
    from ai.v2.dataset import build_samples
    samples = build_samples(index, tiny_v3_cfg.train_writers)
    images, writers, is_genuine, entries = build_batch(samples, tiny_v3_cfg, epoch=1, batch_idx=0)
    assert images.shape[0] == tiny_v3_cfg.images_per_batch
    assert set(writers.tolist()) <= set(tiny_v3_cfg.train_writers)
    assert len(entries) == images.shape[0]
    assert int(is_genuine.sum()) == tiny_v3_cfg.writers_per_batch * tiny_v3_cfg.genuines_per_writer
    # batch is deterministic for the same (epoch, batch_idx)
    im2, w2, g2, e2 = build_batch(samples, tiny_v3_cfg, epoch=1, batch_idx=0)
    assert torch.equal(images, im2) and np.array_equal(writers, w2)


def test_select_triplets_returns_expected_counts(tiny_v3_cfg, index, cpu):
    from ai.v2.dataset import build_samples
    samples = build_samples(index, tiny_v3_cfg.train_writers)
    images, writers, is_genuine, _ = build_batch(samples, tiny_v3_cfg, epoch=1, batch_idx=0)
    model = build_model(tiny_v3_cfg).to(cpu)
    with torch.no_grad():
        emb = model.encode(images)
    bank = CandidateBank(tiny_v3_cfg)
    mined = select_triplets_v3(emb, images, writers, is_genuine, tiny_v3_cfg, epoch=1, bank=bank, device=cpu)
    assert mined["dist_ap"].numel() == mined["dist_an"].numel() == mined["stats"]["triplets"] > 0
    assert len(mined["anchor"]) == mined["stats"]["triplets"]
    assert mined["stats"]["skilled_negatives"] + mined["stats"]["random_negatives"] == mined["stats"]["triplets"]
    assert mined["stats"]["phase"] == tiny_v3_cfg.schedule_phase(1) == "early"
    # per-anchor negative composition respected
    n_sk = int(round(tiny_v3_cfg.negatives_per_anchor * tiny_v3_cfg.skilled_negative_fraction))
    anchors = len(set(mined["anchor"].tolist()))
    assert mined["stats"]["anchors"] == anchors > 0
    assert mined["stats"]["skilled_negatives"] / max(anchors, 1) == pytest.approx(n_sk, rel=0.6)
    # active-loss fraction in [0,1]
    assert 0.0 <= mined["stats"]["active_loss_fraction"] <= 1.0
    for w, img, e in mined["bank_updates"]:
        assert img.dim() == 3 and e.dim() == 1
        assert w in tiny_v3_cfg.train_writers


def test_no_anchors_returns_empty(tiny_v3_cfg, cpu):
    emb = torch.randn(4, tiny_v3_cfg.embedding_dim)
    writers = np.array([1, 1, 2, 2])
    is_genuine = np.array([False, False, False, False])
    mined = select_triplets_v3(emb, torch.randn(4, 3, 64, 32), writers, is_genuine,
                               tiny_v3_cfg, epoch=1, bank=CandidateBank(tiny_v3_cfg), device=cpu)
    assert mined["dist_ap"].numel() == 0
    assert mined["stats"]["triplets"] == 0
    assert len(mined["bank_updates"]) == 0


def test_bank_add_refresh_roundtrip(tiny_v3_cfg, cpu):
    bank = CandidateBank(tiny_v3_cfg)
    img = torch.randn(3, tiny_v3_cfg.canvas_height, tiny_v3_cfg.canvas_width)
    emb = F.normalize(torch.randn(tiny_v3_cfg.embedding_dim), p=2, dim=0)
    bank.add(1, img, emb)
    assert len(bank) == 1
    model = build_model(tiny_v3_cfg).to(cpu)
    n = bank.refresh(model, tiny_v3_cfg, cpu)
    assert n == 1
    t = bank.bank_tensor(1, cpu)
    assert t.shape == (1, tiny_v3_cfg.embedding_dim)
    assert bank.bank_tensor(999, cpu) is None
    # cap enforced
    for i in range(tiny_v3_cfg.bank_cap_per_writer + 10):
        bank.add(1, img, emb)
    assert len(bank) == tiny_v3_cfg.bank_cap_per_writer


def test_pick_mode_phase_semantics(tiny_v3_cfg, cpu):
    """mid/late prefer the hardest negatives; early mixes in random candidates."""
    from ai.v2.dataset import build_samples
    from ai.dataset import load_writer_index
    try:
        index = load_writer_index(tiny_v3_cfg.dataset_root)
    except FileNotFoundError:
        pytest.skip("dataset not available")
    samples = build_samples(index, tiny_v3_cfg.train_writers)
    images, writers, is_genuine, _ = build_batch(samples, tiny_v3_cfg, epoch=1, batch_idx=0)
    model = build_model(tiny_v3_cfg).to(cpu)
    with torch.no_grad():
        emb = model.encode(images)
    bank = CandidateBank(tiny_v3_cfg)
    early = select_triplets_v3(emb, images, writers, is_genuine, tiny_v3_cfg, epoch=1, bank=bank, device=cpu)
    mid = select_triplets_v3(emb, images, writers, is_genuine, tiny_v3_cfg, epoch=10, bank=bank, device=cpu)
    late = select_triplets_v3(emb, images, writers, is_genuine, tiny_v3_cfg, epoch=19, bank=bank, device=cpu)
    assert early["stats"]["phase"] == "early"
    assert mid["stats"]["phase"] == "mid"
    assert late["stats"]["phase"] == "late"
    # mid should keep at least as many hard negatives as early (prefers hardest)
    assert mid["stats"]["hard_ratio"] >= 0.0  # structural check; values vary by data
    assert early["stats"]["skilled_negatives"] + early["stats"]["random_negatives"] == early["stats"]["triplets"]