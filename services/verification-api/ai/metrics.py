"""Verification metrics: ROC, AUC, FAR, FRR, EER.

Scores are SIMILARITY values (higher = more likely the same writer).
Labels use the match convention: 1 = same/genuine, 0 = non-match.
A decision "accept as genuine" is made when similarity >= threshold.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def roc_curve(scores: np.ndarray, labels: np.ndarray):
    """Return (fpr, tpr, thresholds) with threshold on similarity."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    thresholds = np.unique(scores)
    thresholds = np.sort(thresholds)[::-1]
    if thresholds.size == 0:
        return np.array([0.0]), np.array([1.0]), np.array([np.inf])

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    fpr = np.zeros(thresholds.shape)
    tpr = np.zeros(thresholds.shape)
    for i, t in enumerate(thresholds):
        pred = scores >= t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        tpr[i] = tp / n_pos if n_pos else 0.0
        fpr[i] = fp / n_neg if n_neg else 0.0
    return fpr, tpr, thresholds


def roc_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    return float(np.trapezoid(tpr_s, fpr_s))


def far_frr(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    """FAR = fraction of non-match pairs accepted; FRR = genuine pairs rejected."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pred = scores >= threshold
    n_neg = int(np.sum(labels == 0))
    n_pos = int(np.sum(labels == 1))
    far = float(np.sum(pred & (labels == 0))) / n_neg if n_neg else 0.0
    frr = float(np.sum((~pred) & (labels == 1))) / n_pos if n_pos else 0.0
    return far, frr


def accuracy_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pred = scores >= threshold
    return float(np.mean(pred == labels))


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray):
    """EER threshold and rate; threshold on similarity."""
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def full_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute the full standard metric set (threshold selection from data)."""
    fpr, tpr, thr = roc_curve(scores, labels)
    auc = roc_auc(fpr, tpr)
    eer, eer_thr = eer_from_roc(fpr, tpr, thr)
    far_eer, frr_eer = far_frr(scores, labels, eer_thr)
    acc_eer = accuracy_at(scores, labels, eer_thr)

    # Greedy best equal-accuracy threshold as a reference (any tie broken by |FAR-FRR|).
    best = None
    best_delta = np.inf
    for t in thr:
        fa, fr = far_frr(scores, labels, t)
        delta = abs(fa - fr)
        if delta < best_delta or (delta == best_delta and best is None):
            best_delta = delta
            best = t
    far_best, frr_best = far_frr(scores, labels, best)
    acc_best = accuracy_at(scores, labels, best)

    return {
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_thr,
        "far_at_eer": far_eer,
        "frr_at_eer": frr_eer,
        "accuracy_at_eer": acc_eer,
        "best_equal_error_threshold": best,
        "far_at_best": far_best,
        "frr_at_best": frr_best,
        "accuracy_at_best": acc_best,
    }