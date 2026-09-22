"""Evaluation on unseen writers + signer-7 secondary benchmark + reports.

Threshold policy (section 12 of the experiment brief):
  - the OPERATIONAL threshold is derived ONLY from VALIDATION scores;
  - the test set is used for final evaluation with that validation threshold;
  - test-side EER/ROC are also reported as descriptive statistics but are NOT
    used to pick the operational threshold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import metrics as M
from .config import ExperimentConfig, ensure_dirs
from .dataset import Pair, PairSet, PAIR_POSITIVE, PAIR_RANDOM_IMPOSTOR, PAIR_SKILLED_FORGERY
from .model import cosine_similarity, euclidean_distance


def score_pairs(
    model,
    pairs: Sequence[Pair],
    device: torch.device,
    cfg: ExperimentConfig,
    batch_size: int = 128,
) -> Dict:
    """Score every pair -> arrays of (writer, pair_type, label, similarity, distance)."""
    model.eval()
    writer_ids, pair_types, labels, sims, dists = [], [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            from .dataset import PairDataset, default_collate

            ds = PairDataset(chunk, cfg, split="eval")
            loader_batch = [ds[j] for j in range(len(ds))]
            img_a, img_b, _ = default_collate(loader_batch)
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            e1, e2 = model(img_a, img_b)
            sims.append(cosine_similarity(e1, e2).cpu().numpy())
            dists.append(euclidean_distance(e1, e2).cpu().numpy())
            labels.extend(p.label for p in chunk)
            writer_ids.extend(p.writer for p in chunk)
            pair_types.extend(p.pair_type for p in chunk)
    return {
        "writer": np.asarray(writer_ids),
        "pair_type": np.asarray(pair_types),
        "label": np.asarray(labels, dtype=np.float64),
        "similarity": np.concatenate(sims),
        "distance": np.concatenate(dists),
    }


def _subsets(scores: Dict, pair_type: str) -> Tuple[np.ndarray, np.ndarray]:
    mask = scores["pair_type"] == pair_type
    return scores["similarity"][mask], scores["label"][mask]


def summary_block(name: str, scores: Dict) -> Dict:
    s = scores["similarity"]
    lab = scores["label"]
    return {f"{name}_metrics": M.full_metrics(s, lab)}


def run_evaluation(
    model,
    cfg: ExperimentConfig,
    device: torch.device,
    val_pairs: PairSet,
    test_pairs: PairSet,
) -> Dict:
    """Full evaluation: validation threshold + unseen-writer test + subgroups."""
    ensure_dirs(cfg)
    print("\n=== Validation scoring (for threshold selection) ===")
    t0 = time.time()
    val_scores = score_pairs(model, val_pairs.all(), device, cfg)
    print(f"  {len(val_pairs.all())} val pairs scored in {time.time()-t0:.1f}s")

    val_metrics = M.full_metrics(val_scores["similarity"], val_scores["label"])
    val_threshold = val_metrics["eer_threshold"]
    print(f"  val EER={val_metrics['eer']:.4f}  AUC={val_metrics['roc_auc']:.4f}")
    print(f"  VALIDATION-DERIVED THRESHOLD (similarity) = {val_threshold:.4f}")

    print("\n=== Unseen-writer TEST scoring ===")
    t0 = time.time()
    test_scores = score_pairs(model, test_pairs.all(), device, cfg)
    print(f"  {len(test_pairs.all())} test pairs scored in {time.time()-t0:.1f}s")

    test_metrics = M.full_metrics(test_scores["similarity"], test_scores["label"])

    far_vt, frr_vt = M.far_frr(
        test_scores["similarity"], test_scores["label"], val_threshold
    )
    acc_vt = M.accuracy_at(test_scores["similarity"], test_scores["label"], val_threshold)

    print("\n=== Subgroup analysis ===")
    subgroups: Dict[str, Dict] = {}
    for pt, name in (
        (PAIR_POSITIVE, "positive"),
        (PAIR_SKILLED_FORGERY, "skilled_forgery"),
        (PAIR_RANDOM_IMPOSTOR, "random_impostor"),
    ):
        ss, sl = _subsets(test_scores, pt)
        if np.any(sl == 1) and np.any(sl == 0):
            sub_met = M.full_metrics(ss, sl)
        else:
            # Single-class subset: ROC/EER are not meaningful.
            sub_met = None
        fa, fr = M.far_frr(ss, sl, val_threshold)
        acc = M.accuracy_at(ss, sl, val_threshold)
        subgroups[name] = {
            "count": int(len(ss)),
            "metrics": sub_met,
            "far_at_val_threshold": fa,
            "frr_at_val_threshold": fr,
            "accuracy_at_val_threshold": acc,
            "mean_similarity": float(np.mean(ss)),
            "std_similarity": float(np.std(ss)),
        }
        auc_s = f"{sub_met['roc_auc']:.4f}" if sub_met else "n/a"
        eer_s = f"{sub_met['eer']:.4f}" if sub_met else "n/a"
        print(
            f"  {name:<16} n={len(ss):<6} AUC={auc_s} "
            f"EER={eer_s} "
            f"FAR@vt={fa:.4f} FRR@vt={fr:.4f} acc@vt={acc:.4f}"
        )

    # Skilled-forgery FAR specifically (at the validation threshold).
    sf = subgroups["skilled_forgery"]
    ri = subgroups["random_impostor"]
    pos = subgroups["positive"]

    result = {
        "validation": {
            "count": int(len(val_pairs.all())),
            "metrics": val_metrics,
            "threshold": float(val_threshold),
        },
        "test": {
            "count": int(len(test_pairs.all())),
            "metrics": test_metrics,
            "far_at_val_threshold": far_vt,
            "frr_at_val_threshold": frr_vt,
            "accuracy_at_val_threshold": acc_vt,
            "skilled_forgery_far_at_val_threshold": sf["far_at_val_threshold"],
            "random_impostor_far_at_val_threshold": ri["far_at_val_threshold"],
            "positive_accuracy_at_val_threshold": pos["accuracy_at_val_threshold"],
        },
        "subgroups": subgroups,
        "test_scores": test_scores,
        "val_scores": val_scores,
    }

    # ROC data for plots.
    fpr, tpr, thr = M.roc_curve(test_scores["similarity"], test_scores["label"])
    result["roc"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
    }
    return result


def run_signer7_diagnostic(
    model,
    cfg: ExperimentConfig,
    device: torch.device,
    index,
    val_threshold: float,
) -> Dict:
    """Signer-7 benchmark (only valid when writer 7 is in the test split).

    reference = original_7_5 (matches the previous OpenCV baseline protocol).
    """
    wi = index.get(7)
    if wi is None:
        return {"skipped": "writer 7 not present"}
    ref = next((p for p in wi.originals if p.name == "original_7_5.png"), None)
    if ref is None:
        return {"skipped": "original_7_5.png not found"}

    genuine_pairs: List[Pair] = []
    forgery_pairs: List[Pair] = []
    for p in wi.originals:
        if p == ref:
            continue
        genuine_pairs.append(Pair(ref, p, 1, PAIR_POSITIVE, 7))
    for f in wi.forgeries:
        forgery_pairs.append(Pair(ref, f, 0, PAIR_SKILLED_FORGERY, 7))

    gen_scores = score_pairs(model, genuine_pairs, device, cfg)
    for_scores = score_pairs(model, forgery_pairs, device, cfg)
    gen_s = gen_scores["similarity"]
    for_s = for_scores["similarity"]

    all_s = np.concatenate([gen_s, for_s])
    all_l = np.concatenate([np.ones_like(gen_s), np.zeros_like(for_s)])
    met = M.full_metrics(all_s, all_l)
    far_vt, frr_vt = M.far_frr(all_s, all_l, val_threshold)

    return {
        "writer": 7,
        "n_genuine": int(len(gen_s)),
        "n_forgery": int(len(for_s)),
        "genuine_similarity": {
            "mean": float(np.mean(gen_s)),
            "std": float(np.std(gen_s)),
            "min": float(np.min(gen_s)),
            "max": float(np.max(gen_s)),
        },
        "forgery_similarity": {
            "mean": float(np.mean(for_s)),
            "std": float(np.std(for_s)),
            "min": float(np.min(for_s)),
            "max": float(np.max(for_s)),
        },
        "metrics": met,
        "far_at_val_threshold": far_vt,
        "frr_at_val_threshold": frr_vt,
        "interpretation": (
            "diagnostic only; writer 7 is in the TEST split so the model never saw "
            "signer 7 during training (valid unseen-writer comparison)"
        ),
    }


def write_reports(
    cfg: ExperimentConfig,
    result: Dict,
    signer7: Dict,
    train_history: List[Dict],
    test_pairs: PairSet,
    val_pairs: PairSet,
) -> Dict[str, Path]:
    """Write CSVs / JSON / plots to ai/reports. Returns created file paths."""
    ensure_dirs(cfg)
    out: Dict[str, Path] = {}
    rdir = cfg.reports_dir

    # training_history.csv
    import csv as _csv

    hist_path = rdir / "training_history.csv"
    with open(hist_path, "w", newline="") as fh:
        if train_history:
            w = _csv.DictWriter(fh, fieldnames=list(train_history[0].keys()))
            w.writeheader()
            w.writerows(train_history)
    out["training_history.csv"] = hist_path

    # test_pairs.csv (writer, pair_type, label, similarity, distance)
    ts = result["test_scores"]
    tp = rdir / "test_pairs.csv"
    with open(tp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["writer", "pair_type", "label", "similarity", "distance"])
        for p, s, d in zip(test_pairs.all(), ts["similarity"], ts["distance"]):
            w.writerow([p.writer, p.pair_type, p.label, f"{s:.6f}", f"{d:.6f}"])
    out["test_pairs.csv"] = tp

    # validation_pairs.csv
    vs = result["val_scores"]
    vp = rdir / "validation_pairs.csv"
    with open(vp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["writer", "pair_type", "label", "similarity", "distance"])
        for p, s, d in zip(val_pairs.all(), vs["similarity"], vs["distance"]):
            w.writerow([p.writer, p.pair_type, p.label, f"{s:.6f}", f"{d:.6f}"])
    out["validation_pairs.csv"] = vp

    # metrics.json
    payload = {
        "experiment": cfg.model_version,
        "config": cfg.to_dict(),
        "results": {k: v for k, v in result.items() if k not in ("test", "test_scores", "val_scores")},
        "test": {
            k: v for k, v in result["test"].items() if k not in ("_similarity", "_distance")
        },
        "signer7": signer7,
    }
    mj = rdir / "metrics.json"
    mj.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out["metrics.json"] = mj

    _write_plots(cfg, result)
    out["roc_curve.png"] = rdir / "roc_curve.png"
    out["score_distribution.png"] = rdir / "score_distribution.png"
    return out


def _write_plots(cfg: ExperimentConfig, result: Dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rdir = cfg.reports_dir
    ts = result["test_scores"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        result["roc"]["fpr"],
        result["roc"]["tpr"],
        label=f"test ROC (AUC={result['test']['metrics']['roc_auc']:.4f})",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.set_xlabel("FAR (false accept rate)")
    ax.set_ylabel("TPR (1 - FRR)")
    ax.set_title(f"Unseen-writer verification ROC — {cfg.model_version}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rdir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for pt, name, color in (
        (PAIR_POSITIVE, "genuine (same writer)", "tab:green"),
        (PAIR_SKILLED_FORGERY, "skilled forgery", "tab:red"),
        (PAIR_RANDOM_IMPOSTOR, "random impostor", "tab:orange"),
    ):
        mask = ts["pair_type"] == pt
        ax.hist(
            ts["similarity"][mask],
            bins=60,
            alpha=0.5,
            color=color,
            label=name,
            density=True,
        )
    vt = result["validation"]["threshold"]
    ax.axvline(vt, color="black", ls="--", label=f"validation threshold = {vt:.3f}")
    ax.set_xlabel("cosine similarity (higher = same writer)")
    ax.set_ylabel("density")
    ax.set_title(f"Test score distribution — {cfg.model_version}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rdir / "score_distribution.png", dpi=150)
    plt.close(fig)