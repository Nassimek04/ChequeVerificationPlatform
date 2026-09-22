import math

import pytest
import torch
import torch.nn.functional as F

from ai.v3.calibration import (
    pairwise_ref_stats,
    calibrate,
    apply_calibration,
    aggregate,
    is_valid_candidate,
    valid_candidates,
)


def make_refs(k, device="cpu"):
    angles = torch.linspace(0.0, 0.3, k)
    r = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return F.normalize(r, p=2, dim=1)


def test_pairwise_stats_only_use_references():
    refs = make_refs(5)
    stats = pairwise_ref_stats(refs, __import__("ai.v3.config", fromlist=["V3Config"]).V3Config())
    assert "mu" in stats and "sigma" in stats and "med" in stats and "mad" in stats
    assert "mu_p" in stats and "sigma_p" in stats and "prototype" in stats
    # stats depend only on refs (no query/label input exists in the API)
    assert stats["sigma"] >= 1e-6
    assert stats["sigma_p"] >= 1e-6


def test_prototype_is_l2_normalized():
    refs = make_refs(5)
    cfg = __import__("ai.v3.config", fromlist=["V3Config"]).V3Config()
    stats = pairwise_ref_stats(refs, cfg)
    assert torch.allclose(stats["prototype"].norm(p=2), torch.tensor(1.0), atol=1e-5)


def test_calibration_formulas():
    refs = make_refs(5)
    cfg = __import__("ai.v3.config", fromlist=["V3Config"]).V3Config()
    stats = pairwise_ref_stats(refs, cfg)
    s = torch.tensor([0.7, 0.5, 0.2])
    # z = (s - mu)/sigma
    z = calibrate(s, stats, "z", cfg)
    assert torch.allclose(z, (s - stats["mu"]) / stats["sigma"], atol=1e-5)
    # mad = (s - med)/(1.4826*mad)
    m = calibrate(s, stats, "mad", cfg)
    assert torch.allclose(m, (s - stats["med"]) / (cfg.mad_scale * stats["mad"]), atol=1e-5)
    # proto_relative = (s - mu_p)/sigma_p
    p = calibrate(s, stats, "proto_relative", cfg)
    assert torch.allclose(p, (s - stats["mu_p"]) / stats["sigma_p"], atol=1e-5)
    # centroid = s - mu
    c = calibrate(s, stats, "centroid", cfg)
    assert torch.allclose(c, s - stats["mu"], atol=1e-5)
    # raw = identity
    assert torch.equal(calibrate(s, stats, "raw", cfg), s.float())


def test_zero_dispersion_safety():
    cfg = __import__("ai.v3.config", fromlist=["V3Config"]).V3Config()
    # identical refs -> zero dispersion; scores must stay finite (eps floor)
    refs = F.normalize(torch.ones(5, 8), p=2, dim=1)
    stats = pairwise_ref_stats(refs, cfg)
    q = F.normalize(torch.randn(3, 8), p=2, dim=1)
    for agg in ("mean", "max", "median"):
        for cal in ("z", "mad", "proto_relative", "centroid", "raw"):
            if not is_valid_candidate(5, agg, cal, cfg):
                continue
            out, st = apply_calibration(q, refs, agg, cal, cfg)
            assert torch.isfinite(out).all(), (agg, cal)
    assert stats["sigma"] == cfg.calib_eps
    assert stats["sigma_p"] == cfg.calib_eps


def test_aggregate_functions():
    sims = torch.tensor([[0.9, 0.7, 0.5], [0.2, 0.4, 0.6]])
    assert torch.equal(aggregate(sims, "max"), torch.tensor([0.9, 0.6]))
    assert torch.allclose(aggregate(sims, "mean"), torch.tensor([0.7, 0.4]))
    assert torch.allclose(aggregate(sims, "median"), torch.tensor([0.7, 0.4]))
    assert torch.allclose(aggregate(sims, "top2_mean"), torch.tensor([0.8, 0.5]))
    with pytest.raises(ValueError):
        aggregate(sims, "prototype")  # handled by apply_calibration
    with pytest.raises(ValueError):
        aggregate(sims, "bogus")


def test_candidate_validity_constraints(v3_cfg):
    for k, agg, cal in valid_candidates(v3_cfg):
        assert is_valid_candidate(k, agg, cal, v3_cfg)
        if k < 3:
            assert cal == "raw"
        if agg == "top2_mean":
            assert k >= 3
    all_c = {(k, a, c) for k, a, c in valid_candidates(v3_cfg)}
    assert (1, "mean", "raw") in all_c
    assert (5, "prototype", "proto_relative") in all_c
    assert (5, "max", "z") in all_c
    assert (5, "top2_mean", "mad") in all_c
    assert (1, "mean", "z") not in all_c  # K=1 degenerates
    assert (5, "top2_mean", "raw") in all_c


def test_apply_calibration_shape():
    cfg = __import__("ai.v3.config", fromlist=["V3Config"]).V3Config()
    refs = make_refs(5)
    q = F.normalize(torch.randn(7, 2), p=2, dim=1)
    for agg in ("mean", "max", "median", "top2_mean", "prototype"):
        for cal in ("raw", "z", "mad", "proto_relative", "centroid"):
            if not is_valid_candidate(5, agg, cal, cfg):
                continue
            out, stats = apply_calibration(q, refs, agg, cal, cfg)
            assert out.shape == (7,), (agg, cal)
            assert stats["k"] == 5