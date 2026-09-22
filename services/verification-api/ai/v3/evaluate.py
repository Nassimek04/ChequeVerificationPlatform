"""V3 evaluation.

PRIMARY protocol = MULTI-REFERENCE ENROLLMENT:

  For an unseen writer: K genuine reference signatures (enrollment) +
  24-K remaining genuine queries + 24 skilled-forgery queries.

  Query score = aggregation over K per-reference cosine similarities
  (max / mean / median / top-2 mean / prototype), calibrated with
  enrollment-only statistics (raw / z / mad / proto_relative / centroid).

  Model, margin, aggregation, calibration and threshold are frozen using
  VALIDATION writers only; TEST writers are then scored once.

Threshold policies (validation only):
  A: EER-derived threshold.
  B: security-oriented thresholds — lowest FAR subject to a validation FRR cap
     (10% / 15% / 20%).

Cross-writer stability: per-writer score means/std are compared between raw
and calibrated scores to quantify inter-writer scale shift.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ai import metrics as M
from ai.dataset import Pair

from ai.v2.evaluate import (  # noqa: F401 (reuse)
    encode_paths,
    encode_all_writers,
    pair_based_metrics,
    pair_based_apply_frozen,
)

from .calibration import apply_calibration, valid_candidates, aggregate  # noqa: F401
from .config import V3Config


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_writer(
    gen_emb: torch.Tensor,
    forg_emb: torch.Tensor,
    k: int,
    agg: str,
    cal: str,
    cfg: V3Config,
) -> Dict:
    """Score one writer: refs = first k genuines, queries = rest + forgeries."""
    refs = gen_emb[:k]
    q_gen = gen_emb[k:]
    q_forg = forg_emb

    def _cal(q: torch.Tensor) -> torch.Tensor:
        return apply_calibration(q, refs, agg, cal, cfg)[0]

    return {"genuine": _cal(q_gen), "forgery": _cal(q_forg)}


def evaluate_multiref(
    emb_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    writers: Sequence[int],
    k: int,
    agg: str,
    cal: str,
    cfg: V3Config,
) -> Dict:
    """Pooled scores across writers for a (K, agg, cal) candidate.

    score/label/writer arrays are mutually aligned (genuines first, then
    forgeries, per writer, in the same order).
    """
    scores_gen, scores_forg, wgen, wforg = [], [], [], []
    for w in writers:
        g, f = emb_map[w]
        res = score_writer(g, f, k, agg, cal, cfg)
        scores_gen.append(res["genuine"].numpy())
        scores_forg.append(res["forgery"].numpy())
        wgen.extend([w] * res["genuine"].shape[0])
        wforg.extend([w] * res["forgery"].shape[0])
    sg = np.concatenate(scores_gen)
    sf = np.concatenate(scores_forg)
    writer_ids = wgen + wforg
    labels = np.concatenate([np.ones_like(sg), np.zeros_like(sf)])
    scores = np.concatenate([sg, sf])
    return {
        "writer": np.asarray(writer_ids),
        "label": labels,
        "score": scores,
        "genuine_scores": sg,
        "forgery_scores": sf,
        "n_genuine_queries": int(len(sg)),
        "n_forgery_queries": int(len(sf)),
    }


def candidate_metrics(data: Dict, cfg: V3Config) -> Dict:
    met = M.full_metrics(data["score"], data["label"])
    sf = data["forgery_scores"]
    met["skilled_forgery_far_at_eer"] = M.far_frr(sf, np.zeros_like(sf), met["eer_threshold"])[0]
    met["accuracy_at_eer"] = M.accuracy_at(data["score"], data["label"], met["eer_threshold"])
    return met


# ---------------------------------------------------------------------------
# Threshold policies (validation only)
# ---------------------------------------------------------------------------
def threshold_policy_a(scores: np.ndarray, labels: np.ndarray) -> Dict:
    met = M.full_metrics(scores, labels)
    return {
        "policy": "A",
        "threshold": float(met["eer_threshold"]),
        "eer": met["eer"],
        "far": met["far_at_eer"],
        "frr": met["frr_at_eer"],
    }


def threshold_policy_b(scores: np.ndarray, labels: np.ndarray, limits: Sequence[float]) -> Dict:
    """Security-oriented: lowest FAR subject to a validation FRR cap.

    Genuines accepted when score >= t, so FRR(t) grows with t. For each cap,
    pick the LARGEST t with FRR(t) <= cap (tightest accept threshold that
    respects the cap). Scores are the candidate thresholds.
    """
    uniq = np.unique(scores)
    gen = scores[labels == 1]
    out = {}
    for cap in limits:
        valid_ts = [t for t in uniq if M.far_frr(scores, labels, t)[1] <= cap]
        if not valid_ts:
            valid_ts = [uniq.min()]
        t = float(valid_ts[-1])  # largest t satisfying FRR <= cap
        far, frr = M.far_frr(scores, labels, t)
        out[f"B_frr{cap:.2f}"] = {
            "policy": f"B (FRR<=%.0f%%)" % (cap * 100),
            "threshold": t,
            "far": far,
            "frr": frr,
            "accuracy": M.accuracy_at(scores, labels, t),
            "frr_cap": cap,
        }
    return out


# ---------------------------------------------------------------------------
# Selection on validation
# ---------------------------------------------------------------------------
def select_on_validation(val_emb, cfg: V3Config) -> Dict:
    """Choose (K, agg, cal) on validation by (EER, skilled FAR at EER)."""
    results = {}
    for cand in valid_candidates(cfg):
        data = evaluate_multiref(val_emb, cfg.val_writers, *cand, cfg)
        results[cand] = {"data": data, "metrics": candidate_metrics(data, cfg)}
    best = min(
        results,
        key=lambda c: (results[c]["metrics"]["eer"], results[c]["metrics"]["skilled_forgery_far_at_eer"]),
    )
    data = results[best]["data"]
    pol_a = threshold_policy_a(data["score"], data["label"])
    pol_b = threshold_policy_b(data["score"], data["label"], cfg.frr_limits)
    return {
        "best_candidate": best,
        "policy_a": pol_a,
        "policy_b": pol_b,
        "validation_metrics": results[best]["metrics"],
        "table": {f"{k}-{a}-{c}": results[(k, a, c)]["metrics"] for (k, a, c) in valid_candidates(cfg)},
        "candidate_data": data,
    }


# ---------------------------------------------------------------------------
# Frozen test application
# ---------------------------------------------------------------------------
def evaluate_test_frozen(
    test_emb,
    test_writers: Sequence[int],
    selection: Dict,
    cfg: V3Config,
    policy: str,
) -> Dict:
    """Apply frozen (K, agg, cal) + selected threshold to unseen test writers."""
    k, agg, cal = selection["best_candidate"]
    if policy.startswith("A"):
        thr = selection["policy_a"]["threshold"]
    else:
        thr = selection["policy_b"][policy]["threshold"]

    data = evaluate_multiref(test_emb, test_writers, k, agg, cal, cfg)
    met = M.full_metrics(data["score"], data["label"])
    far, frr = M.far_frr(data["score"], data["label"], thr)
    acc = M.accuracy_at(data["score"], data["label"], thr)
    sf = data["forgery_scores"]
    skilled_far, _ = M.far_frr(sf, np.zeros_like(sf), thr)
    # random-impostor FAR from pair-based protocol (cross-writer genuines)
    ri_far = random_impostor_far(test_emb, test_writers, k, agg, cal, cfg, thr)

    per_writer = []
    for w in test_writers:
        m = data["writer"] == w
        ws, wl = data["score"][m], data["label"][m]
        w_far, w_frr = M.far_frr(ws, wl, thr)
        gs = ws[wl == 1]; fs = ws[wl == 0]
        per_writer.append({
            "writer": w,
            "n_genuine": int((wl == 1).sum()),
            "n_forgery": int((wl == 0).sum()),
            "far": w_far,
            "frr": w_frr,
            "accuracy": M.accuracy_at(ws, wl, thr),
            "mean_genuine_score": float(gs.mean()) if len(gs) else None,
            "std_genuine_score": float(gs.std()) if len(gs) else None,
            "mean_forgery_score": float(fs.mean()) if len(fs) else None,
            "std_forgery_score": float(fs.std()) if len(fs) else None,
        })
    stability = cross_writer_stability(data, test_writers)
    return {
        "best_candidate": (k, agg, cal),
        "policy": policy,
        "threshold": thr,
        "metrics": met,
        "far_at_frozen_threshold": far,
        "frr_at_frozen_threshold": frr,
        "accuracy_at_frozen_threshold": acc,
        "skilled_forgery_far_at_frozen_threshold": skilled_far,
        "random_impostor_far_at_frozen_threshold": ri_far,
        "per_writer": per_writer,
        "stability": stability,
        "all_scores": data,
    }


def random_impostor_far(test_emb, writers, k, agg, cal, cfg, threshold) -> float:
    """Cross-writer impostor FAR: a genuine query of writer A vs references of
    writer B (A != B), pooled over all ordered pairs. Enrollment stats come only
    from B's genuines (enrollment refs)."""
    rej = []
    for wb in writers:
        gb, _ = test_emb[wb]
        refs = gb[:k]
        for wa in writers:
            if wa == wb:
                continue
            ga, _ = test_emb[wa]
            for q in ga:
                s = apply_calibration(q.unsqueeze(0), refs, agg, cal, cfg)[0].item()
                rej.append(s)
    rej = np.array(rej)
    if rej.size == 0:
        return None
    return float((rej >= threshold).mean())


# ---------------------------------------------------------------------------
# Cross-writer stability
# ---------------------------------------------------------------------------
def cross_writer_stability(data: Dict, writers: Sequence[int]) -> Dict:
    """Per-writer score stats; inter-writer variance of genuine/forgery means."""
    rows = []
    for w in writers:
        m = data["writer"] == w
        ws, wl = data["score"][m], data["label"][m]
        gs = ws[wl == 1]; fs = ws[wl == 0]
        rows.append({
            "writer": w,
            "genuine_mean": float(gs.mean()) if len(gs) else None,
            "genuine_std": float(gs.std()) if len(gs) else None,
            "forgery_mean": float(fs.mean()) if len(fs) else None,
            "forgery_std": float(fs.std()) if len(fs) else None,
            "sep": float(gs.mean() - fs.mean()) if len(gs) and len(fs) else None,
        })
    gm = np.array([r["genuine_mean"] for r in rows if r["genuine_mean"] is not None])
    fm = np.array([r["forgery_mean"] for r in rows if r["forgery_mean"] is not None])
    sep = np.array([r["sep"] for r in rows if r["sep"] is not None])
    return {
        "per_writer": rows,
        "std_of_genuine_means": float(gm.std()) if len(gm) else None,
        "std_of_forgery_means": float(fm.std()) if len(fm) else None,
        "mean_separation": float(sep.mean()) if len(sep) else None,
        "std_of_separations": float(sep.std()) if len(sep) else None,
    }


# ---------------------------------------------------------------------------
# Writer 7
# ---------------------------------------------------------------------------
def writer7_report(test_emb, selection, cfg: V3Config) -> Dict:
    g, f = test_emb[7]
    k, agg, cal = selection["best_candidate"]
    report = {"writer": 7, "best_candidate": (k, agg, cal)}

    # K=1/3/5 x aggregation x calibration grid (diagnostic)
    grid = {}
    for kk in cfg.eval_ks:
        for aa in cfg.aggregations:
            for cc in cfg.calibrations:
                from .calibration import is_valid_candidate
                if not is_valid_candidate(kk, aa, cc, cfg):
                    continue
                res = score_writer(g, f, kk, aa, cc, cfg)
                sg, sf = res["genuine"].numpy(), res["forgery"].numpy()
                s_all = np.concatenate([sg, sf])
                lab = np.concatenate([np.ones_like(sg), np.zeros_like(sf)])
                met = M.full_metrics(s_all, lab)
                grid[f"K{kk}_{aa}_{cc}"] = {
                    "n_genuine": int(len(sg)),
                    "n_forgery": int(len(sf)),
                    "genuine_mean": float(sg.mean()),
                    "genuine_std": float(sg.std()),
                    "forgery_mean": float(sf.mean()),
                    "forgery_std": float(sf.std()),
                    "auc": met["roc_auc"],
                    "eer": met["eer"],
                }

    # Frozen protocol (selected K/agg/cal + both threshold policies)
    frozen_rows = {}
    for policy in ["A", "B_frr0.10", "B_frr0.15", "B_frr0.20"]:
        fr = evaluate_test_frozen({7: (g, f)}, [7], selection, cfg, policy)
        frozen_rows[policy] = {
            "threshold": fr["threshold"],
            "far": fr["far_at_frozen_threshold"],
            "frr": fr["frr_at_frozen_threshold"],
            "accuracy": fr["accuracy_at_frozen_threshold"],
            "skilled_far": fr["skilled_forgery_far_at_frozen_threshold"],
        }

    return {
        "writer": 7,
        "grid": grid,
        "frozen": frozen_rows,
        "interpretation": (
            "writer 7 is TEST-only: the model and protocol were frozen without "
            "seeing writer 7. Thresholds come from validation writers; writer-7 "
            "labels are used only for descriptive reporting."
        ),
    }