import numpy as np
import pytest
import torch

from ai import metrics as M
from ai.v3 import evaluate as E
from ai.v3.calibration import valid_candidates, is_valid_candidate
from ai.v3.tests.helpers import make_writer_emb, fake_emb_map


def test_aggregate_strategies_parity_with_v2():
    sims = torch.tensor([[0.9, 0.7, 0.5], [0.2, 0.4, 0.6]])
    assert torch.equal(E.aggregate(sims, "max"), torch.tensor([0.9, 0.6]))
    assert torch.allclose(E.aggregate(sims, "mean"), torch.tensor([0.7, 0.4]))
    assert torch.allclose(E.aggregate(sims, "median"), torch.tensor([0.7, 0.4]))
    assert torch.allclose(E.aggregate(sims, "top2_mean"), torch.tensor([0.8, 0.5]))


def test_score_writer_separates_genuine(v3_cfg):
    g, f = make_writer_emb(seed=1)
    for k, agg, cal in valid_candidates(v3_cfg):
        res = E.score_writer(g, f, k, agg, cal, v3_cfg)
        assert res["genuine"].shape[0] == 24 - k
        assert res["forgery"].shape[0] == 24
        assert res["genuine"].mean() > res["forgery"].mean(), (k, agg, cal)


def test_score_writer_genuine_only_enrollment(v3_cfg):
    """Enrollment references are the first k genuines; forgeries are queries only."""
    g, f = make_writer_emb(seed=2)
    refs = g[:5]
    k = 5
    for agg, cal in [(a, c) for a in v3_cfg.aggregations for c in v3_cfg.calibrations
                     if is_valid_candidate(k, a, c, v3_cfg)]:
        from ai.v3.calibration import apply_calibration
        out, stats = apply_calibration(g[5:], refs, agg, cal, v3_cfg)
        assert out.shape[0] == 19
        # stats derived from refs only; compare against direct pairwise computation
        from ai.v3.calibration import pairwise_ref_stats
        stats2 = pairwise_ref_stats(refs, v3_cfg)
        assert stats["mu"] == stats2["mu"] and stats["sigma"] == stats2["sigma"]


def test_evaluate_multiref_sizes_and_alignment(v3_cfg):
    emb = fake_emb_map([1, 2], seed=3)
    data = E.evaluate_multiref(emb, [1, 2], 3, "mean", "raw", v3_cfg)
    assert len(data["score"]) == len(data["label"]) == len(data["writer"])
    assert data["n_genuine_queries"] == 2 * (24 - 3)
    assert data["n_forgery_queries"] == 2 * 24
    for w in (1, 2):
        pos = data["writer"] == w
        assert pos.sum() == (24 - 3) + 24
        assert (data["label"][pos] == 1).sum() == 24 - 3
        assert (data["label"][pos] == 0).sum() == 24


def test_evaluate_multiref_deterministic(v3_cfg):
    emb = fake_emb_map([1, 2, 3], seed=4)
    d1 = E.evaluate_multiref(emb, [1, 2, 3], 5, "prototype", "z", v3_cfg)
    d2 = E.evaluate_multiref(emb, [1, 2, 3], 5, "prototype", "z", v3_cfg)
    assert np.array_equal(d1["score"], d2["score"])
    assert np.array_equal(d1["label"], d2["label"])


def test_candidate_metrics_reports_skilled_far(v3_cfg):
    data = {"score": np.array([0.9, 0.8, 0.2, 0.1]), "label": np.array([1, 1, 0, 0]),
            "genuine_scores": np.array([0.9, 0.8]), "forgery_scores": np.array([0.2, 0.1])}
    met = E.candidate_metrics(data, v3_cfg)
    assert "skilled_forgery_far_at_eer" in met
    assert "accuracy_at_eer" in met


def test_selection_uses_only_validation(v3_cfg):
    val_emb = fake_emb_map(v3_cfg.val_writers, seed=5)
    sel = E.select_on_validation(val_emb, v3_cfg)
    assert sel["best_candidate"] in valid_candidates(v3_cfg)
    assert sel["policy_a"]["threshold"] is not None
    assert len(sel["policy_b"]) == len(v3_cfg.frr_limits)
    for k in sel["policy_b"]:
        cap = float(k.split("frr")[-1])
        assert sel["policy_b"][k]["frr"] <= cap + 1e-6 + 0.001
    # table covers every valid candidate
    assert len(sel["table"]) == len(valid_candidates(v3_cfg))


def test_policy_b_respects_frr_cap(v3_cfg):
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.5, 0.2, 200), rng.normal(-0.5, 0.2, 200)])
    labels = np.concatenate([np.ones(200), np.zeros(200)])
    pol = E.threshold_policy_b(scores, labels, v3_cfg.frr_limits)
    for cap in v3_cfg.frr_limits:
        v = pol[f"B_frr{cap:.2f}"]
        assert v["frr"] <= cap + 1e-6 + 0.001
    # stricter cap -> lower threshold (accepts fewer forgeries) -> higher FRR
    assert pol["B_frr0.10"]["threshold"] <= pol["B_frr0.20"]["threshold"]


def test_policy_a_threshold_is_eer(v3_cfg):
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.5, 0.2, 100), rng.normal(-0.5, 0.2, 100)])
    labels = np.concatenate([np.ones(100), np.zeros(100)])
    a = E.threshold_policy_a(scores, labels)
    met = M.full_metrics(scores, labels)
    assert a["threshold"] == pytest.approx(met["eer_threshold"], abs=1e-9)


def test_frozen_test_application(v3_cfg):
    val_emb = fake_emb_map(v3_cfg.val_writers, seed=6)
    test_emb = fake_emb_map(v3_cfg.test_writers, seed=7)
    sel = E.select_on_validation(val_emb, v3_cfg)
    fr = E.evaluate_test_frozen(test_emb, v3_cfg.test_writers, sel, v3_cfg, "A")
    assert fr["best_candidate"] == sel["best_candidate"]
    assert fr["threshold"] == sel["policy_a"]["threshold"]
    assert len(fr["per_writer"]) == len(v3_cfg.test_writers)
    for row in fr["per_writer"]:
        assert row["writer"] in v3_cfg.test_writers
        assert 0.0 <= row["accuracy"] <= 1.0
    # skilled FAR at the frozen threshold cannot exceed the overall FAR of the
    # skill pool combined with random impostors trivially; sanity bounds
    assert 0.0 <= fr["skilled_forgery_far_at_frozen_threshold"] <= 1.0


def test_random_impostor_far_sane(v3_cfg):
    test_emb = fake_emb_map(v3_cfg.test_writers[:3], seed=8)
    sel = E.select_on_validation(fake_emb_map(v3_cfg.val_writers, seed=9), v3_cfg)
    k, agg, cal = sel["best_candidate"]
    ri = E.random_impostor_far(test_emb, list(test_emb.keys()), k, agg, cal, v3_cfg, 0.0)
    assert ri is not None and 0.0 <= ri <= 1.0


def test_random_impostor_far_single_writer_returns_none(v3_cfg):
    g, f = make_writer_emb(seed=10)
    ri = E.random_impostor_far({7: (g, f)}, [7], 5, "mean", "raw", v3_cfg, 0.5)
    assert ri is None


def test_cross_writer_stability_metrics(v3_cfg):
    emb = fake_emb_map(v3_cfg.test_writers, seed=11)
    data = E.evaluate_multiref(emb, v3_cfg.test_writers, 5, "mean", "raw", v3_cfg)
    st = E.cross_writer_stability(data, v3_cfg.test_writers)
    assert len(st["per_writer"]) == len(v3_cfg.test_writers)
    assert st["std_of_genuine_means"] is not None
    assert st["mean_separation"] is not None
    assert st["std_of_separations"] is not None


def test_writer7_report_grid_and_frozen(v3_cfg):
    test_emb = fake_emb_map([7], seed=12)
    val_emb = fake_emb_map(v3_cfg.val_writers, seed=13)
    sel = E.select_on_validation(val_emb, v3_cfg)
    rep = E.writer7_report(test_emb, sel, v3_cfg)
    assert rep["writer"] == 7
    assert len(rep["grid"]) == len(valid_candidates(v3_cfg))
    for key, cell in rep["grid"].items():
        assert "auc" in cell and "eer" in cell and "genuine_mean" in cell
    for policy in ("A", "B_frr0.10", "B_frr0.15", "B_frr0.20"):
        assert policy in rep["frozen"]
    assert "interpretation" in rep


def test_encode_paths_reused_from_v2():
    from ai.v2 import evaluate as V2E
    assert E.encode_paths is V2E.encode_paths
    assert E.encode_all_writers is V2E.encode_all_writers