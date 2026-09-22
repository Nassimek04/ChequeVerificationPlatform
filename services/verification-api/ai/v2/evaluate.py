"""V2 evaluation.

Two complementary protocols:

(A) PAIR-BASED (V1-comparable) — scores the EXACT same pair pools as V1 with
    the V2 model and re-derives a threshold on VALIDATION writers. This is the
    like-for-like embedding-quality comparison.

(B) MULTI-REFERENCE (application-style) — simulates a customer with K enrolled
    reference signatures. For each unseen writer, K genuines are the enrollment,
    remaining genuines + all forgeries are queries. Query score = aggregation
    over the K per-reference cosine similarities (max / mean / median /
    top-2 mean / prototype) with optional enrollment-based z-normalization.
    Aggregation + calibration + threshold are selected on VALIDATION writers
    and then FROZEN for the TEST writers. No test forgery labels are ever used
    for calibration.

For efficiency, all writer images are embedded ONCE per split; candidate
strategies are then evaluated in numpy.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ai import metrics as M
from ai.dataset import Pair
from ai.evaluate import score_pairs as v1_score_pairs

from .config import V2Config
from .dataset import WriterSamples

EPS = 1e-6


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
@torch.inference_mode()
def encode_paths(model, paths, device: torch.device, cfg: V2Config, batch_size: int = 64) -> torch.Tensor:
    model.eval()
    embs = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        from ai.dataset import PairDataset, default_collate

        ds = PairDataset([Pair(p, p, 1, "positive", 0) for p in chunk], cfg, split="eval")
        batch = default_collate([ds[j] for j in range(len(ds))])
        x = batch[0].to(device)
        embs.append(model.encode(x).cpu())
    return torch.cat(embs, dim=0)


def encode_all_writers(
    model, cfg: V2Config, device: torch.device, writers: Sequence[int], index
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """{writer: (genuine_emb, forgery_emb)} in deterministic filename order."""
    out = {}
    for w in writers:
        samples = WriterSamples(w, list(index[w].originals), list(index[w].forgeries))
        g = encode_paths(model, samples.genuine_paths, device, cfg)
        f = encode_paths(model, samples.forgery_paths, device, cfg)
        out[w] = (g, f)
    return out


# ---------------------------------------------------------------------------
# Multi-reference scoring (numpy-friendly via torch on embeddings)
# ---------------------------------------------------------------------------
def _aggregate(sims: torch.Tensor, strategy: str) -> torch.Tensor:
    """sims: [Q, K] cosine similarities. Returns [Q]."""
    if strategy == "max":
        return sims.max(dim=1).values
    if strategy == "mean":
        return sims.mean(dim=1)
    if strategy == "median":
        return sims.median(dim=1).values
    if strategy == "top2_mean":
        return sims.topk(2, dim=1).values.mean(dim=1)
    raise ValueError(f"unexpected aggregation strategy: {strategy}")


def _ref_pairwise_stats(refs: torch.Tensor) -> Tuple[float, float]:
    """Mean/std of cosine similarities among the K reference embeddings."""
    k = refs.shape[0]
    if k < 3:
        return 0.0, 1.0
    sims = refs @ refs.T
    idx = torch.triu_indices(k, k, offset=1)
    vals = sims[idx[0], idx[1]]
    return float(vals.mean()), float(max(vals.std(unbiased=False), EPS))


def score_writer(
    gen_emb: torch.Tensor,
    forg_emb: torch.Tensor,
    k: int,
    strategy: str,
    norm: str,
) -> Dict:
    """Score one writer: references = first k genuines, queries = rest + forgeries."""
    refs = gen_emb[:k]
    q_gen = gen_emb[k:]
    q_forg = forg_emb
    mu, sigma = _ref_pairwise_stats(refs)

    def _agg(q: torch.Tensor) -> torch.Tensor:
        sims = q @ refs.T  # [Q,K]
        if strategy == "prototype":
            proto = F.normalize(refs.mean(dim=0), p=2, dim=0)
            raw = q @ proto
        else:
            raw = _aggregate(sims, strategy)
        if norm == "z":
            return (raw - mu) / sigma
        return raw

    return {"genuine": _agg(q_gen), "forgery": _agg(q_forg)}


def valid_candidates(cfg: V2Config) -> List[Tuple[int, str, str]]:
    out = []
    for k in cfg.eval_ks:
        for strat in cfg.aggregations:
            if strat == "top2_mean" and k < 3:
                continue
            for norm in cfg.normalizations:
                if norm == "z" and k < 3:
                    continue
                out.append((k, strat, norm))
    return out


def evaluate_multiref_embeddings(
    emb_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    writers: Sequence[int],
    k: int,
    strategy: str,
    norm: str,
) -> Dict:
    """Pooled scores across writers for a (K, strategy, norm) candidate."""
    scores_gen, scores_forg, writer_ids_gen, writer_ids_forg = [], [], [], []
    for w in writers:
        g, f = emb_map[w]
        res = score_writer(g, f, k, strategy, norm)
        scores_gen.append(res["genuine"].numpy())
        scores_forg.append(res["forgery"].numpy())
        writer_ids_gen.extend([w] * res["genuine"].shape[0])
        writer_ids_forg.extend([w] * res["forgery"].shape[0])
    sg = np.concatenate(scores_gen)
    sf = np.concatenate(scores_forg)
    # scores are ordered [all genuine queries, all forgery queries] ->
    # writer ids must follow the same ordering to stay aligned.
    writer_ids = writer_ids_gen + writer_ids_forg
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


def candidate_metrics(data: Dict) -> Dict:
    met = M.full_metrics(data["score"], data["label"])
    skilled_far, _ = M.far_frr(
        data["forgery_scores"], np.zeros_like(data["forgery_scores"]), met["eer_threshold"]
    )
    met["skilled_forgery_far_at_eer"] = skilled_far
    return met


# ---------------------------------------------------------------------------
# Selection on validation, frozen application on test
# ---------------------------------------------------------------------------
def select_on_validation(
    val_emb: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    val_writers: Sequence[int],
    cfg: V2Config,
) -> Dict:
    """Pick (K, strategy, norm) minimizing validation EER; tie-break by
    skilled-forgery FAR at EER threshold. Never touches test data."""
    results = {}
    for cand in valid_candidates(cfg):
        data = evaluate_multiref_embeddings(val_emb, val_writers, *cand)
        results[cand] = {"data": data, "metrics": candidate_metrics(data)}
    best = min(
        results,
        key=lambda c: (results[c]["metrics"]["eer"], results[c]["metrics"]["skilled_forgery_far_at_eer"]),
    )
    return {
        "best_candidate": best,
        "threshold": float(results[best]["metrics"]["eer_threshold"]),
        "validation_metrics": results[best]["metrics"],
        "table": {f"{k}-{s}-{n}": results[(k, s, n)]["metrics"] for (k, s, n) in valid_candidates(cfg)},
    }


def evaluate_test_frozen(
    test_emb: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    test_writers: Sequence[int],
    selection: Dict,
) -> Dict:
    """Apply the frozen validation selection to unseen test writers."""
    k, strat, norm = selection["best_candidate"]
    thr = selection["threshold"]
    data = evaluate_multiref_embeddings(test_emb, test_writers, k, strat, norm)
    met = M.full_metrics(data["score"], data["label"])
    far, frr = M.far_frr(data["score"], data["label"], thr)
    acc = M.accuracy_at(data["score"], data["label"], thr)
    sf_scores = data["forgery_scores"]
    skilled_far, _ = M.far_frr(sf_scores, np.zeros_like(sf_scores), thr)

    per_writer = []
    for w in test_writers:
        m = data["writer"] == w
        w_scores = data["score"][m]
        w_labels = data["label"][m]
        w_far, w_frr = M.far_frr(w_scores, w_labels, thr)
        w_acc = M.accuracy_at(w_scores, w_labels, thr)
        gs = w_scores[w_labels == 1]
        fs = w_scores[w_labels == 0]
        per_writer.append({
            "writer": w,
            "n_genuine": int((w_labels == 1).sum()),
            "n_forgery": int((w_labels == 0).sum()),
            "far": w_far,
            "frr": w_frr,
            "accuracy": w_acc,
            "mean_genuine_score": float(gs.mean()) if len(gs) else None,
            "mean_forgery_score": float(fs.mean()) if len(fs) else None,
        })
    return {
        "best_candidate": (k, strat, norm),
        "threshold": thr,
        "metrics": met,
        "far_at_frozen_threshold": far,
        "frr_at_frozen_threshold": frr,
        "accuracy_at_frozen_threshold": acc,
        "skilled_forgery_far_at_frozen_threshold": skilled_far,
        "per_writer": per_writer,
        "n_genuine_queries": data["n_genuine_queries"],
        "n_forgery_queries": data["n_forgery_queries"],
        "all_scores": data,
    }


# ---------------------------------------------------------------------------
# Pair-based (V1-comparable)
# ---------------------------------------------------------------------------
def pair_based_metrics(model, cfg, device, pairs: Sequence[Pair]) -> Dict:
    scores = v1_score_pairs(model, pairs, device, cfg)
    met = M.full_metrics(scores["similarity"], scores["label"])
    skilled_mask = scores["pair_type"] == "skilled_forgery"
    imp_mask = scores["pair_type"] == "random_impostor"
    met["skilled_forgery_far_at_eer"] = M.far_frr(
        scores["similarity"][skilled_mask], scores["label"][skilled_mask], met["eer_threshold"]
    )[0]
    met["random_impostor_far_at_eer"] = M.far_frr(
        scores["similarity"][imp_mask], scores["label"][imp_mask], met["eer_threshold"]
    )[0]
    return {"metrics": met, "scores": scores}


def pair_based_apply_frozen(model, cfg, device, pairs, threshold: float) -> Dict:
    scores = v1_score_pairs(model, pairs, device, cfg)
    far, frr = M.far_frr(scores["similarity"], scores["label"], threshold)
    acc = M.accuracy_at(scores["similarity"], scores["label"], threshold)
    skilled_mask = scores["pair_type"] == "skilled_forgery"
    imp_mask = scores["pair_type"] == "random_impostor"
    sf_far, _ = M.far_frr(scores["similarity"][skilled_mask], scores["label"][skilled_mask], threshold)
    ri_far, _ = M.far_frr(scores["similarity"][imp_mask], scores["label"][imp_mask], threshold)
    return {
        "far": far, "frr": frr, "accuracy": acc,
        "skilled_forgery_far": sf_far, "random_impostor_far": ri_far,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Signer-7
# ---------------------------------------------------------------------------
def signer7_report(g: torch.Tensor, f: torch.Tensor, val_threshold: float, frozen: Dict) -> Dict:
    """Writer-7 report from precomputed embeddings (g, f)."""
    samples = None  # writer index not needed; embeddings already ordered
    # V1-protocol single reference original_7_5 (index 4 in filename order)
    ref = 4
    s_gen_all = (g[ref] * g).sum(-1)
    s_gen = torch.cat([s_gen_all[:ref], s_gen_all[ref + 1 :]])
    s_forg = (g[ref] * f).sum(-1)
    s_all = torch.cat([s_gen, s_forg]).numpy()
    lab = np.concatenate([np.ones(len(s_gen)), np.zeros(len(s_forg))])
    met = M.full_metrics(s_all, lab)
    far, frr = M.far_frr(s_all, lab, val_threshold)

    krefs = {}
    for k in (1, 3, 5):
        for strat in ("mean", "prototype"):
            res = score_writer(g, f, k, strat, "raw")
            krefs[f"K{k}_{strat}"] = {
                "genuine_mean": float(res["genuine"].mean()),
                "forgery_mean": float(res["forgery"].mean()),
                "n_genuine": int(len(res["genuine"])),
                "n_forgery": int(len(res["forgery"])),
            }

    k7, s7, n7 = frozen["best_candidate"]
    res7 = score_writer(g, f, k7, s7, n7)
    sg7 = res7["genuine"].numpy()
    sf7 = res7["forgery"].numpy()
    a7 = np.concatenate([sg7, sf7])
    l7 = np.concatenate([np.ones_like(sg7), np.zeros_like(sf7)])
    far7, frr7 = M.far_frr(a7, l7, frozen["threshold"])

    return {
        "writer": 7,
        "pair_based_original_7_5": {
            "genuine_similarity": {
                "mean": float(s_gen.mean()),
                "std": float(s_gen.std()),
                "min": float(s_gen.min()),
                "max": float(s_gen.max()),
            },
            "forgery_similarity": {
                "mean": float(s_forg.mean()),
                "std": float(s_forg.std()),
                "min": float(s_forg.min()),
                "max": float(s_forg.max()),
            },
            "metrics": met,
            "far_at_global_val_threshold": far,
            "frr_at_global_val_threshold": frr,
            "n_genuine": int(len(s_gen)),
            "n_forgery": int(len(s_forg)),
        },
        "multi_reference": krefs,
        "frozen_selection": {"candidate": (k7, s7, n7), "far": far7, "frr": frr7},
        "interpretation": (
            "writer 7 is in the TEST split: the model never saw signer 7 during "
            "training. No operational threshold derived from writer-7 test labels."
        ),
    }
