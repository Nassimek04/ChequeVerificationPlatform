import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ai.v2 import evaluate as E


def make_writer_emb(n_gen=6, n_forg=4, device=torch.device("cpu")):
    """Genuines near (1,0,0), forgeries near (0,1,0)."""
    g = torch.stack([
        F.normalize(torch.tensor([1.0, 0.05 * i], device=device), p=2, dim=0)
        for i in range(n_gen)
    ])
    f = torch.stack([
        F.normalize(torch.tensor([0.05, 1.0 + 0.05 * i], device=device), p=2, dim=0)
        for i in range(n_forg)
    ])
    g = torch.cat([g, torch.zeros(n_gen, 1)], dim=1)
    f = torch.cat([f, torch.zeros(n_forg, 1)], dim=1)
    return g, f


def test_aggregate_strategies():
    sims = torch.tensor([[0.9, 0.7], [0.2, 0.4]])
    assert torch.equal(E._aggregate(sims, "max"), torch.tensor([0.9, 0.4]))
    assert torch.allclose(E._aggregate(sims, "mean"), torch.tensor([0.8, 0.3]))
    assert torch.allclose(E._aggregate(sims, "median"), torch.tensor([0.7, 0.2]))
    assert torch.allclose(E._aggregate(sims, "top2_mean"), torch.tensor([0.8, 0.3]))
    with pytest.raises(ValueError):
        E._aggregate(sims, "bogus")


def test_score_writer_separates_authentic():
    g, f = make_writer_emb()
    res = E.score_writer(g, f, k=3, strategy="mean", norm="raw")
    assert res["genuine"].mean() > res["forgery"].mean()


def test_score_writer_prototype():
    g, f = make_writer_emb()
    res = E.score_writer(g, f, k=3, strategy="prototype", norm="raw")
    assert res["genuine"].mean() > 0.95
    assert res["forgery"].mean() < 0.2


def test_score_writer_z_normalized():
    g, f = make_writer_emb()
    res = E.score_writer(g, f, k=3, strategy="mean", norm="z")
    assert res["genuine"].mean() > res["forgery"].mean()
    # z-score: genuine should be near 0 mean; forgery strongly negative
    assert res["forgery"].mean() < -1.0


def test_valid_candidates_constraints(v2_cfg):
    cands = E.valid_candidates(v2_cfg)
    for k, strat, norm in cands:
        if k < 3:
            assert norm == "raw" and strat != "top2_mean"
    all_c = {(k, s, n) for k, s, n in cands}
    assert (1, "mean", "raw") in all_c
    assert (3, "top2_mean", "z") in all_c
    assert (3, "prototype", "z") in all_c


def test_evaluate_multiref_embeddings_sizes():
    emb = {1: make_writer_emb(), 2: make_writer_emb()}
    data = E.evaluate_multiref_embeddings(emb, [1, 2], 3, "mean", "raw")
    assert len(data["score"]) == len(data["label"]) == len(data["writer"])
    assert data["n_genuine_queries"] == 2 * (6 - 3)
    assert data["n_forgery_queries"] == 2 * 4


def test_evaluate_multiref_writer_alignment():
    """writer/label/score arrays must be mutually aligned per query."""
    emb = {1: make_writer_emb(n_gen=6, n_forg=4), 2: make_writer_emb(n_gen=6, n_forg=4)}
    data = E.evaluate_multiref_embeddings(emb, [1, 2], 3, "mean", "raw")
    for w in (1, 2):
        pos = data["writer"] == w
        assert pos.sum() == (6 - 3) + 4
        assert (data["label"][pos & (data["label"] == 1)].sum() == (6 - 3))
        assert (data["label"][pos & (data["label"] == 0)].sum() == 0)
        # genuine queries carry label 1 and belong to writer w
        g = pos & (data["label"] == 1)
        assert g.sum() == 6 - 3
        # forgeries carry label 0 and belong to writer w
        f = pos & (data["label"] == 0)
        assert f.sum() == 4


def test_select_on_validation_picks_known_candidate(v2_cfg):
    emb = {w: make_writer_emb() for w in (3, 6, 28)}
    sel = E.select_on_validation(emb, [3, 6, 28], v2_cfg)
    assert sel["best_candidate"] in E.valid_candidates(v2_cfg)
    assert 0.0 < sel["threshold"] <= 1.0
    assert "eer" in sel["validation_metrics"]
    assert sel["validation_metrics"]["eer"] >= 0.0


def test_evaluate_test_frozen_never_recomputes_selection():
    val_emb = {w: make_writer_emb() for w in (3, 6)}
    test_emb = {w: make_writer_emb() for w in (2, 7)}
    sel = E.select_on_validation(val_emb, [3, 6], pytest.importorskip("ai.v2.config").V2Config())
    frozen = E.evaluate_test_frozen(test_emb, [2, 7], sel)
    assert frozen["best_candidate"] == sel["best_candidate"]
    assert len(frozen["per_writer"]) == 2
    for row in frozen["per_writer"]:
        assert row["writer"] in (2, 7)
        assert 0.0 <= row["accuracy"] <= 1.0


def test_signature7_report():
    g, f = make_writer_emb(n_gen=24, n_forg=24)
    sel = {"best_candidate": (5, "mean", "raw"), "threshold": 0.5}
    rep = E.signer7_report(g, f, 0.5, sel)
    assert rep["writer"] == 7
    pb = rep["pair_based_original_7_5"]
    assert pb["n_genuine"] == 23 and pb["n_forgery"] == 24
    assert pb["genuine_similarity"]["mean"] > pb["forgery_similarity"]["mean"]
    assert "metrics" in pb and "roc_auc" in pb["metrics"]
    assert "multi_reference" in rep and "K1_mean" in rep["multi_reference"]


def test_candidate_metrics_includes_skilled_far():
    data = {"score": np.array([0.9, 0.8, 0.2, 0.1]), "label": np.array([1, 1, 0, 0]),
            "genuine_scores": np.array([0.9, 0.8]), "forgery_scores": np.array([0.2, 0.1])}
    met = E.candidate_metrics(data)
    assert "skilled_forgery_far_at_eer" in met