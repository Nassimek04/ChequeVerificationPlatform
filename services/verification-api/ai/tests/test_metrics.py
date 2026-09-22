import numpy as np

from ai.metrics import accuracy_at, far_frr, full_metrics, roc_auc, roc_curve


def test_metrics_perfect_separation():
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    scores = np.array([0.9, 0.95, 0.99, 0.1, 0.2, 0.3])
    m = full_metrics(scores, labels)
    assert m["roc_auc"] == 1.0
    assert m["eer"] <= 1e-9
    far, frr = far_frr(scores, labels, 0.5)
    assert far == 0.0 and frr == 0.0
    assert accuracy_at(scores, labels, 0.5) == 1.0


def test_metrics_random_scores_bounds():
    rng = np.random.default_rng(0)
    labels = (rng.random(2000) > 0.5).astype(float)
    scores = rng.random(2000)
    m = full_metrics(scores, labels)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["eer"] <= 0.5
    assert 0.0 <= m["far_at_eer"] <= 1.0
    assert 0.0 <= m["frr_at_eer"] <= 1.0
    assert 0.0 <= m["accuracy_at_eer"] <= 1.0


def test_roc_monotonic():
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    scores = np.array([0.9, 0.95, 0.99, 0.1, 0.2, 0.3])
    fpr, tpr, thr = roc_curve(scores, labels)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= -1e-12)
    assert roc_auc(fpr, tpr) == 1.0
