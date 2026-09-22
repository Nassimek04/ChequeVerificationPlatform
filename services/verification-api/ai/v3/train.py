"""V3 entry point (python -m ai.v3.train).

Phases (choose one):
  study  : train margins {0.2,0.3,0.4}, save study checkpoints + study_results
  final  : select best margin (validation), train final model, run the complete
           V3 evaluation (multi-reference, calibrations, threshold policies,
           writer-7, cross-writer stability) and write reports
  eval   : eval-only from the saved final checkpoint

Run from services/verification-api:  python -m ai.v3.train --phase <phase>
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ai import metrics as M
from ai.config import ExperimentConfig
from ai.dataset import generate_pairs, load_writer_index, validate_no_writer_leakage

from . import evaluate as E
from .calibration import valid_candidates
from .config import V3Config, detect_device, ensure_dirs
from ai.v2.dataset import assert_exact_v1_split, build_samples
from .losses import TripletLoss
from .mining import CandidateBank, build_batch, select_triplets_v3
from .model import build_model


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI V3 multi-reference enrollment experiment")
    p.add_argument("--phase", choices=["study", "final", "eval"], default="study",
                   help="'study' = fast validation margin study (4 epochs per margin)")
    p.add_argument("--margin", type=float, default=None, help="override selected margin for --phase final")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--study-epochs", type=int, default=4, help="epochs per margin in the validation study")
    p.add_argument("--final-epochs", type=int, default=15, help="max epochs for the final model")
    p.add_argument("--final-patience", type=int, default=4, help="early-stopping patience for the final model")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--skip-v1v2-recompute", action="store_true")
    return p.parse_args()


def _cfg(args) -> V3Config:
    kw = {}
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.batches_per_epoch is not None:
        kw["batches_per_epoch"] = args.batches_per_epoch
    return V3Config(**kw)


def print_environment(cfg: V3Config, device: torch.device) -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print(f"  torch: {torch.__version__}  CUDA: {torch.version.cuda}  device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM: {round(torch.cuda.get_device_properties(0).total_memory/1e9,2)} GB")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def validation_objective(scores: np.ndarray, labels: np.ndarray, skilled_scores: np.ndarray) -> tuple:
    """(EER, skilled FAR at the FRR<=15% policy-B threshold)."""
    met = M.full_metrics(scores, labels)
    pol_b = E.threshold_policy_b(scores, labels, [0.15])
    t = pol_b["B_frr0.15"]["threshold"]
    skilled_far, _ = M.far_frr(skilled_scores, np.zeros_like(skilled_scores), t)
    return met["eer"], skilled_far


def train_model(
    model, cfg: V3Config, device, train_samples, val_pool, margin: float,
) -> dict:
    set_seeds(cfg.seed)
    loss_fn = TripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bank = CandidateBank(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    use_amp = cfg.mixed_precision and device.type == "cuda"

    history, mining_hist = [], []
    best_score, best_epoch, patience = (float("inf"), float("inf")), -1, 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        set_seeds(cfg.seed + epoch)
        model.train()
        agg = {k: 0.0 for k in
               ("triplets", "skilled_negatives", "random_negatives", "hard", "semi_hard",
                "easy", "anchors", "mean_d_ap", "mean_d_an_skilled", "active_loss_fraction")}
        n_b = 0
        for b in range(cfg.batches_per_epoch):
            images, writers_arr, is_genuine, _ = build_batch(train_samples, cfg, epoch, b)
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                emb = model.encode(images)
                mined = select_triplets_v3(emb, images, writers_arr, is_genuine, cfg, epoch, bank, device)
                if mined["dist_ap"].numel() == 0:
                    continue
                loss = loss_fn(mined["dist_ap"].float(), mined["dist_an"].float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for k in ("triplets", "skilled_negatives", "random_negatives", "hard", "semi_hard",
                      "easy", "anchors", "mean_d_ap", "mean_d_an_skilled", "active_loss_fraction"):
                v = mined["stats"][k]
                if k == "active_loss_fraction":
                    # fraction -> weighted count (active triplets), divided later
                    agg[k] += v * mined["stats"]["triplets"]
                elif k in ("mean_d_ap", "mean_d_an_skilled"):
                    agg[k] += v  # summed per-batch means, averaged over batches
                else:
                    agg[k] += v
            for w, img, e in mined["bank_updates"]:
                bank.add(w, img, e)
            n_b += 1

        if cfg.schedule_phase(epoch) == "late" or epoch % cfg.bank_refresh_every == 0:
            refreshed = bank.refresh(model, cfg, device)
        else:
            refreshed = 0

        # validation (pair-based, cheap, V1-comparable)
        vres = E.pair_based_metrics(model, cfg, device, val_pool.all())
        vmet = vres["metrics"]
        skilled = vres["scores"]["similarity"][vres["scores"]["pair_type"] == "skilled_forgery"]
        obj = validation_objective(vres["scores"]["similarity"], vres["scores"]["label"], skilled)
        if obj < best_score:
            best_score, best_epoch, patience = obj, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        n = max(n_b, 1)
        row = {
            "epoch": epoch, "phase": cfg.schedule_phase(epoch),
            "train_loss": round(float(loss.item()), 6),
            "hard_ratio": round(agg["hard"] / max(agg["triplets"], 1), 4),
            "semi_ratio": round(agg["semi_hard"] / max(agg["triplets"], 1), 4),
            "easy_ratio": round(agg["easy"] / max(agg["triplets"], 1), 4),
            "skilled_frac": round(agg["skilled_negatives"] / max(agg["triplets"], 1), 4),
            "mean_d_ap": round(agg["mean_d_ap"] / n, 4),
            "mean_d_an_skilled": round(agg["mean_d_an_skilled"] / n, 4),
            "active_loss_frac": round(agg["active_loss_fraction"] / max(agg["triplets"], 1), 4),
            "bank_size": len(bank),
            "val_eer": round(vmet["eer"], 6),
            "val_auc": round(vmet["roc_auc"], 6),
            "val_far_at_eer": round(vmet["far_at_eer"], 6),
            "val_frr_at_eer": round(vmet["frr_at_eer"], 6),
            "val_threshold_eer": round(vmet["eer_threshold"], 6),
            "val_skilled_far_at_b15": round(obj[1], 6),
            "time_s": round(time.time() - t0, 2),
        }
        history.append(row)
        mining_hist.append(row)
        print(
            f"  ep{epoch:>2} [{row['phase']}] loss={row['train_loss']:.4f} "
            f"h={row['hard_ratio']:.2f} s={row['semi_ratio']:.2f} e={row['easy_ratio']:.2f} "
            f"sk={row['skilled_frac']:.2f} dap={row['mean_d_ap']:.3f} dan={row['mean_d_an_skilled']:.3f} "
            f"act={row['active_loss_frac']:.2f} bank={row['bank_size']} valEER={vmet['eer']:.4f} "
            f"({'+' if obj == best_score else f'pat {patience}'}) [{row['time_s']:.0f}s]"
        )
        if patience >= cfg.early_stopping_patience:
            print(f"  early stopping at epoch {epoch}")
            break

    train_seconds = time.time() - t_start
    print(f"  finished {train_seconds/60:.1f} min; best epoch {best_epoch} "
          f"(val EER {best_score[0]:.4f}, val skilled FAR@B15 {best_score[1]:.4f})")
    model.load_state_dict(best_state)
    model.to(device)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "history": history,
        "mining_history": mining_hist,
        "best_val_eer": float(best_score[0]),
        "best_val_skilled_far_b15": float(best_score[1]),
        "train_seconds": train_seconds,
        "optimizer_state": optimizer.state_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_study(cfg: V3Config, device, index, val_pool, study_epochs: int = 4) -> dict:
    """FAST validation margin study: a few epochs per margin, validation-only.

    For each margin reports: best validation EER, AUC, skilled-forgery FAR,
    FRR, mining hard/semi-hard/easy ratios and active triplet-loss fraction.
    Selection by (validation multi-ref EER, skilled FAR at EER). Never touches
    test labels.
    """
    study_dir = cfg.reports_dir / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    train_samples = build_samples(index, cfg.train_writers)
    for margin in cfg.margin_study:
        print(f"\n=== MARGIN STUDY (validation, {study_epochs} epochs): margin={margin} ===")
        mcfg = dataclasses.replace(cfg, margin=margin, epochs=study_epochs)
        model = build_model(mcfg).to(device)
        res = train_model(model, mcfg, device, train_samples, val_pool, margin)
        # per-margin report from the best-epoch row of the training history
        hist_by_epoch = {h["epoch"]: h for h in res["history"]}
        best = hist_by_epoch[res["best_epoch"]]
        ck = study_dir / f"checkpoint_margin_{margin:.1f}.pt"
        torch.save({
            "model_version": cfg.model_version, "margin": margin,
            "model_state": res["model"].state_dict(), "best_epoch": res["best_epoch"],
            "history": res["history"], "optimizer_state": res["optimizer_state"],
        }, ck)
        # validation multi-reference evaluation for margin comparison
        val_emb = E.encode_all_writers(res["model"], mcfg, device, cfg.val_writers, index)
        sel = E.select_on_validation(val_emb, mcfg)
        vmet = sel["validation_metrics"]
        pol_a = sel["policy_a"]
        pol_b = sel["policy_b"]["B_frr0.15"]
        results[margin] = {
            "checkpoint": str(ck),
            "best_epoch": res["best_epoch"],
            # pair-based validation (V1-comparable protocol)
            "pair_val_eer": res["best_val_eer"],
            "pair_val_auc": best["val_auc"],
            "pair_val_far_at_eer": best["val_far_at_eer"],
            "pair_val_frr_at_eer": best["val_frr_at_eer"],
            "pair_val_skilled_far_at_b15": res["best_val_skilled_far_b15"],
            # mining ratios at the best epoch
            "hard_ratio": best["hard_ratio"],
            "semi_ratio": best["semi_ratio"],
            "easy_ratio": best["easy_ratio"],
            "active_loss_fraction": best["active_loss_frac"],
            "mean_d_ap": best["mean_d_ap"],
            "mean_d_an_skilled": best["mean_d_an_skilled"],
            # multi-reference validation (primary protocol)
            "multi_val_eer": vmet["eer"],
            "multi_val_auc": vmet["roc_auc"],
            "multi_val_skilled_far_at_eer": vmet["skilled_forgery_far_at_eer"],
            "multi_val_far_at_eer": vmet["far_at_eer"],
            "multi_val_frr_at_eer": vmet["frr_at_eer"],
            "multi_val_policy_a_far": pol_a["far"],
            "multi_val_policy_a_frr": pol_a["frr"],
            "multi_val_policy_b15_skilled_far": pol_b["far"],
            "multi_val_policy_b15_frr": pol_b["frr"],
            "best_candidate": list(sel["best_candidate"]),
        }
        r = results[margin]
        print(f"  margin {margin}: best_ep={r['best_epoch']} "
              f"pairEER={r['pair_val_eer']:.4f} (AUC={r['pair_val_auc']:.4f}) "
              f"multiEER={r['multi_val_eer']:.4f} skilledFAR@EER={r['multi_val_skilled_far_at_eer']:.4f} "
              f"cand={tuple(r['best_candidate'])}")
        print(f"    mining @best: hard={r['hard_ratio']:.3f} semi={r['semi_ratio']:.3f} "
              f"easy={r['easy_ratio']:.3f} act={r['active_loss_fraction']:.3f}")

    # select margin by (validation multi-ref EER, skilled FAR at EER)
    best_margin = min(results, key=lambda m: (results[m]["multi_val_eer"], results[m]["multi_val_skilled_far_at_eer"]))
    results["_selected_margin"] = best_margin
    (study_dir / "study_results.json").write_text(json.dumps(
        {str(k): (v if not isinstance(v, dict) else {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()})
         for k, v in results.items()}, indent=2), encoding="utf-8")
    fields = ["margin", "best_epoch", "pair_val_eer", "pair_val_auc", "pair_val_far_at_eer",
              "pair_val_frr_at_eer", "pair_val_skilled_far_at_b15", "hard_ratio", "semi_ratio",
              "easy_ratio", "active_loss_fraction", "mean_d_ap", "mean_d_an_skilled",
              "multi_val_eer", "multi_val_auc", "multi_val_skilled_far_at_eer",
              "multi_val_far_at_eer", "multi_val_frr_at_eer",
              "multi_val_policy_a_far", "multi_val_policy_a_frr",
              "multi_val_policy_b15_skilled_far", "multi_val_policy_b15_frr", "best_candidate"]
    with open(study_dir / "margin_study.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for m in cfg.margin_study:
            row = {"margin": m, **results[m]}
            w.writerow({f: row[f] for f in fields})
    print(f"\nSELECTED MARGIN = {best_margin} "
          f"(val EER {results[best_margin]['multi_val_eer']:.4f}, "
          f"skilled FAR@EER {results[best_margin]['multi_val_skilled_far_at_eer']:.4f})")
    return results


def run_final(cfg: V3Config, device, index, val_pool, test_pool, selected_margin, args) -> dict:
    train_samples = build_samples(index, cfg.train_writers)
    cfg = dataclasses.replace(
        cfg, margin=selected_margin,
        epochs=args.final_epochs, early_stopping_patience=args.final_patience,
    )
    print(f"  training final V3 model: margin={selected_margin}, "
          f"max {cfg.epochs} epochs, early-stop patience {cfg.early_stopping_patience}")
    model = build_model(cfg).to(device)
    res = train_model(model, cfg, device, train_samples, val_pool, selected_margin)
    model, best_epoch, history = res["model"], res["best_epoch"], res["history"]
    train_seconds = res["train_seconds"]

    # pair-based val threshold (V1-comparable protocol)
    val_pair = E.pair_based_metrics(model, cfg, device, val_pool.all())
    val_threshold = float(val_pair["metrics"]["eer_threshold"])
    print(f"  pair-based val EER={val_pair['metrics']['eer']:.4f} threshold={val_threshold:.4f}")

    # V1/V2 recompute on identical pools
    v1_pair = v2_pair = None
    if not args.skip_v1v2_recompute:
        for tag, cpath, cc in (("v1", cfg.v1_checkpoint_path, ExperimentConfig()),
                               ("v2", cfg.v2_checkpoint_path, None)):
            from ai.v2.config import V2Config as _V2C
            cc = cc or _V2C()
            ck = torch.load(cpath, map_location="cpu", weights_only=False)
            m = build_model(cc).to(device)
            m.load_state_dict(ck["model_state"])
            vp = E.pair_based_metrics(m, cc, device, val_pool.all())
            vt = E.pair_based_apply_frozen(m, cc, device, test_pool.all(), vp["metrics"]["eer_threshold"])
            if tag == "v1":
                v1_pair = {"val": vp, "test": vt}
            else:
                v2_pair = {"val": vp, "test": vt}
            print(f"  {tag} recomputed: test AUC={M.full_metrics(vt['scores']['similarity'], vt['scores']['label'])['roc_auc']:.4f}")

    # multi-reference selection on VALIDATION
    print("\nMulti-reference selection on VALIDATION...")
    val_emb = E.encode_all_writers(model, cfg, device, cfg.val_writers, index)
    selection = E.select_on_validation(val_emb, cfg)
    k_star, a_star, c_star = selection["best_candidate"]
    print(f"  best candidate: K={k_star} agg={a_star} cal={c_star} "
          f"val EER={selection['validation_metrics']['eer']:.4f} "
          f"skilled FAR@EER={selection['validation_metrics']['skilled_forgery_far_at_eer']:.4f}")
    print(f"  Policy A (EER): threshold={selection['policy_a']['threshold']:.4f} "
          f"FAR={selection['policy_a']['far']:.4f} FRR={selection['policy_a']['frr']:.4f}")
    for pk, pv in selection["policy_b"].items():
        print(f"  Policy {pv['policy']}: threshold={pv['threshold']:.4f} "
              f"FAR={pv['far']:.4f} FRR={pv['frr']:.4f}")

    test_emb = E.encode_all_writers(model, cfg, device, cfg.test_writers, index)
    results = {"cfg": cfg.to_dict(), "best_epoch": best_epoch, "train_seconds": train_seconds,
               "margin": selected_margin, "selection": selection, "train_history": history}
    for policy in ["A", "B_frr0.10", "B_frr0.15", "B_frr0.20"]:
        fr = E.evaluate_test_frozen(test_emb, cfg.test_writers, selection, cfg, policy)
        results[f"test_{policy}"] = fr
        print(f"  TEST policy {policy} (K{k_star}/{a_star}/{c_star}): "
              f"AUC={fr['metrics']['roc_auc']:.4f} EER={fr['metrics']['eer']:.4f} "
              f"FAR={fr['far_at_frozen_threshold']:.4f} FRR={fr['frr_at_frozen_threshold']:.4f} "
              f"skilledFAR={fr['skilled_forgery_far_at_frozen_threshold']:.4f} "
              f"riFAR={fr['random_impostor_far_at_frozen_threshold']}")

    w7 = E.writer7_report(test_emb, selection, cfg)
    results["writer7"] = w7

    ckpt = {
        "model_version": cfg.model_version,
        "architecture": {"backbone": cfg.backbone, "embedding_dim": cfg.embedding_dim},
        "pretrained": cfg.pretrained,
        "canvas_size": {"width": cfg.canvas_width, "height": cfg.canvas_height},
        "preprocessing_version": cfg.preprocessing_version,
        "loss": {"type": "triplet", "margin": selected_margin, "distance": "euclidean on L2 embeddings"},
        "mining": {
            "strategy": "candidate bank + scheduled phases (early/mid/late)",
            "writers_per_batch": cfg.writers_per_batch,
            "genuines_per_writer": cfg.genuines_per_writer,
            "forgeries_per_writer": cfg.forgeries_per_writer,
            "negatives_per_anchor": cfg.negatives_per_anchor,
            "skilled_negative_fraction": cfg.skilled_negative_fraction,
            "bank_cap_per_writer": cfg.bank_cap_per_writer,
            "bank_refresh_every": cfg.bank_refresh_every,
            "schedule_early_frac": cfg.schedule_early_frac,
            "schedule_mid_frac": cfg.schedule_mid_frac,
        },
        "writer_split": {"train": list(cfg.train_writers), "validation": list(cfg.val_writers),
                         "test": list(cfg.test_writers)},
        "training_epoch": best_epoch,
        "aggregation": a_star, "K": k_star, "calibration": c_star,
        "validation_threshold": selection["policy_a"]["threshold"],
        "threshold_policy": "A (EER) + B (FRR caps)",
        "validation_metrics": selection["validation_metrics"],
        "policy_a": selection["policy_a"],
        "policy_b": selection["policy_b"],
        "config": cfg.to_dict(),
        "model_state": model.state_dict(),
        "train_history": history,
        "seed": cfg.seed,
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.checkpoint_path)
    print(f"\nCheckpoint saved: {cfg.checkpoint_path}")

    write_reports(cfg, results, val_pair, v1_pair, v2_pair, val_pool, test_pool)
    return results


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj) if isinstance(obj, Path) else obj


def write_reports(cfg: V3Config, results, val_pair, v1_pair, v2_pair, val_pool, test_pool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ai.dataset import load_writer_index
    from .evaluate import score_writer

    rdir = cfg.reports_dir
    rdir.mkdir(parents=True, exist_ok=True)

    # training + mining history
    hist = results["train_history"]
    with open(rdir / "training_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    with open(rdir / "mining_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)

    # validation scores (selected candidate)
    vs = results["selection"]["candidate_data"]
    with open(rdir / "validation_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["writer", "label", "score"])
        for wi, la, sc in zip(vs["writer"], vs["label"], vs["score"]):
            w.writerow([wi, int(la), f"{sc:.6f}"])

    # test scores (selected candidate)
    ts = results["test_A"]["all_scores"]
    with open(rdir / "test_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["writer", "label", "score"])
        for wi, la, sc in zip(ts["writer"], ts["label"], ts["score"]):
            w.writerow([wi, int(la), f"{sc:.6f}"])

    with open(rdir / "per_writer_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results["test_A"]["per_writer"][0].keys()))
        w.writeheader()
        w.writerows(results["test_A"]["per_writer"])

    # calibration/multi-reference comparison (validation candidates)
    with open(rdir / "calibration_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "aggregation", "calibration", "auc", "eer", "eer_threshold",
                    "skilled_far_at_eer", "accuracy_at_eer"])
        for key, met in sorted(results["selection"]["table"].items()):
            k, a, c = key.split("-")
            w.writerow([k, a, c, f"{met['roc_auc']:.6f}", f"{met['eer']:.6f}",
                        f"{met['eer_threshold']:.6f}", f"{met['skilled_forgery_far_at_eer']:.6f}",
                        f"{met['accuracy_at_eer']:.6f}"])
    shutil.copyfile(rdir / "calibration_comparison.csv", rdir / "multi_reference_comparison.csv")

    # metrics.json
    payload = {
        "model_version": cfg.model_version,
        "margin": results["margin"],
        "best_epoch": results["best_epoch"],
        "train_seconds": results["train_seconds"],
        "pair_based_validation": {"eer": val_pair["metrics"]["eer"], "auc": val_pair["metrics"]["roc_auc"],
                                  "threshold": val_pair["metrics"]["eer_threshold"]},
        "selection": {
            "best_candidate": list(results["selection"]["best_candidate"]),
            "policy_a": results["selection"]["policy_a"],
            "policy_b": results["selection"]["policy_b"],
            "validation_metrics": results["selection"]["validation_metrics"],
        },
        "test": {k: _clean(v) for k, v in results.items() if k.startswith("test_")},
        "writer7": _clean(results["writer7"]),
        "v1_recomputed_pair_based": None if v1_pair is None else _clean({
            "test_metrics": M.full_metrics(v1_pair["test"]["scores"]["similarity"], v1_pair["test"]["scores"]["label"]),
            "test_at_validation_threshold": {
                "far": v1_pair["test"]["far"], "frr": v1_pair["test"]["frr"],
                "accuracy": v1_pair["test"]["accuracy"],
                "skilled_forgery_far": v1_pair["test"]["skilled_forgery_far"],
                "random_impostor_far": v1_pair["test"]["random_impostor_far"]}}),
        "v2_recomputed_pair_based": None if v2_pair is None else _clean({
            "test_metrics": M.full_metrics(v2_pair["test"]["scores"]["similarity"], v2_pair["test"]["scores"]["label"]),
            "test_at_validation_threshold": {
                "far": v2_pair["test"]["far"], "frr": v2_pair["test"]["frr"],
                "accuracy": v2_pair["test"]["accuracy"],
                "skilled_forgery_far": v2_pair["test"]["skilled_forgery_far"],
                "random_impostor_far": v2_pair["test"]["random_impostor_far"]}}),
        "config": cfg.to_dict(),
    }
    (rdir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ROC curve (test, selected candidate)
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = M.roc_curve(ts["score"], ts["label"])
    ax.plot(fpr, tpr, label=f"V3 test ROC (AUC={results['test_A']['metrics']['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FAR"); ax.set_ylabel("TPR"); ax.set_title(f"V3 test ROC — {cfg.model_version}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rdir / "roc_curve.png", dpi=150); plt.close(fig)

    # raw vs calibrated distributions on TEST (re-encode once)
    k, a, c = results["selection"]["best_candidate"]
    dev = detect_device()
    model = build_model(cfg).to(dev)
    ck = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    index = load_writer_index(cfg.dataset_root)
    test_emb = E.encode_all_writers(model, cfg, dev, cfg.test_writers, index)
    raw_sg, raw_sf = [], []
    for w in cfg.test_writers:
        g, f = test_emb[w]
        r = score_writer(g, f, k, a, "raw", cfg)
        raw_sg.append(r["genuine"].numpy()); raw_sf.append(r["forgery"].numpy())
    raw_sg = np.concatenate(raw_sg); raw_sf = np.concatenate(raw_sf)

    thr = results["test_A"]["threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(raw_sg, bins=50, alpha=0.5, color="tab:green", density=True, label="genuine")
    axes[0].hist(raw_sf, bins=50, alpha=0.5, color="tab:red", density=True, label="forgery")
    axes[0].set_title("RAW scores (test)")
    axes[1].hist(ts["score"][ts["label"] == 1], bins=50, alpha=0.5, color="tab:green", density=True, label="genuine")
    axes[1].hist(ts["score"][ts["label"] == 0], bins=50, alpha=0.5, color="tab:red", density=True, label="forgery")
    axes[1].set_title(f"CALIBRATED ({c}) scores (test)")
    for ax in axes:
        ax.axvline(thr, color="black", ls="--", label=f"threshold {thr:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rdir / "score_distribution_raw.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ts["score"][ts["label"] == 1], bins=50, alpha=0.5, color="tab:green", density=True, label="genuine")
    ax.hist(ts["score"][ts["label"] == 0], bins=50, alpha=0.5, color="tab:red", density=True, label="forgery")
    ax.axvline(thr, color="black", ls="--", label=f"threshold {thr:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rdir / "score_distribution_calibrated.png", dpi=150); plt.close(fig)

    print(f"\nReports written to {rdir}")


def main() -> None:
    args = parse_args()
    cfg = _cfg(args)
    ensure_dirs(cfg)
    device = detect_device() if args.device == "auto" else torch.device(args.device)
    print_environment(cfg, device)
    if cfg.cuda_deterministic and device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # preservation hashes
    pres = {
        "v1": {"path": str(cfg.v1_checkpoint_path), "sha256": sha256(cfg.v1_checkpoint_path)},
        "v2": {"path": str(cfg.v2_checkpoint_path), "sha256": sha256(cfg.v2_checkpoint_path)},
    }
    (cfg.reports_dir / "preservation.json").write_text(json.dumps(pres, indent=2), encoding="utf-8")
    print("V1 SHA256:", pres["v1"]["sha256"])
    print("V2 SHA256:", pres["v2"]["sha256"])

    index = load_writer_index(cfg.dataset_root)
    from ai.v2.dataset import WriterSplitMismatch
    try:
        assert_exact_v1_split(cfg)
    except WriterSplitMismatch as e:
        raise SystemExit(f"STOP: split mismatch -> {e}")
    print("exact V1/V2 split asserted")

    val_pool = generate_pairs(index, cfg.val_writers, "validation", cfg.seed)
    test_pool = generate_pairs(index, cfg.test_writers, "test", cfg.seed)
    validate_no_writer_leakage(val_pool, test_pool)

    if args.phase == "study":
        run_study(cfg, device, index, val_pool, study_epochs=args.study_epochs)
    elif args.phase == "final":
        if args.margin is not None:
            cfg = dataclasses.replace(cfg, margin=args.margin)
        study = json.loads((cfg.reports_dir / "study" / "study_results.json").read_text(encoding="utf-8"))
        sel_margin = args.margin if args.margin is not None else float(study["_selected_margin"])
        print(f"Selected margin: {sel_margin}")
        run_final(cfg, device, index, val_pool, test_pool, sel_margin, args)
    elif args.phase == "eval":
        study = json.loads((cfg.reports_dir / "study" / "study_results.json").read_text(encoding="utf-8"))
        sel_margin = args.margin if args.margin is not None else float(study["_selected_margin"])
        print(f"eval-only with selected margin {sel_margin}")
        run_final(cfg, device, index, val_pool, test_pool, sel_margin, args)


if __name__ == "__main__":
    main()