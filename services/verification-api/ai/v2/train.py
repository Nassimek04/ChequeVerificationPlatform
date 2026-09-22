"""V2 training + evaluation entry point (python -m ai.v2.train).

Triplet loss + hard/semi-hard negative mining (skilled-forgery focus) with the
UNCHANGED V1 backbone and EXACT V1 writer split.

Pipeline:
  split assertion -> training (mining, AMP, early stop, best by val EER) ->
  pair-based V1-comparable evaluation (val threshold -> test) ->
  multi-reference K=1/3/5 + prototype + calibration selection on VALIDATION ->
  frozen application to unseen TEST writers -> signer-7 -> reports.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ai import metrics as M
from ai.config import ExperimentConfig
from ai.dataset import generate_pairs, load_writer_index, validate_no_writer_leakage

from . import evaluate as E
from .config import V2Config, detect_device, ensure_dirs
from .dataset import WriterSplitMismatch, assert_exact_v1_split, build_samples
from .losses import TripletLoss
from .mining import build_batch, select_triplets
from .model import build_model


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI V2 metric-learning (triplet + hard mining)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--writers-per-batch", type=int, default=None)
    p.add_argument("--genuines-per-writer", type=int, default=None)
    p.add_argument("--forgeries-per-writer", type=int, default=None)
    p.add_argument("--negatives-per-anchor", type=int, default=None)
    p.add_argument("--skilled-fraction", type=float, default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--eval-only", action="store_true", help="load V2 checkpoint, skip training")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--skip-v1-recompute", action="store_true")
    return p.parse_args()


def _resolve_cfg(args: argparse.Namespace) -> V2Config:
    kw = {}
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.lr is not None:
        kw["lr"] = args.lr
    if args.margin is not None:
        kw["margin"] = args.margin
    if args.writers_per_batch is not None:
        kw["writers_per_batch"] = args.writers_per_batch
    if args.genuines_per_writer is not None:
        kw["genuines_per_writer"] = args.genuines_per_writer
    if args.forgeries_per_writer is not None:
        kw["forgeries_per_writer"] = args.forgeries_per_writer
    if args.negatives_per_anchor is not None:
        kw["negatives_per_anchor"] = args.negatives_per_anchor
    if args.skilled_fraction is not None:
        kw["skilled_negative_fraction"] = args.skilled_fraction
    if args.batches_per_epoch is not None:
        kw["batches_per_epoch"] = args.batches_per_epoch
    return V2Config(**kw)


def print_environment(cfg: V2Config, device: torch.device) -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print(f"  torch version         : {torch.__version__}")
    print(f"  CUDA available        : {torch.cuda.is_available()}")
    print(f"  CUDA build            : {torch.version.cuda or 'n/a'}")
    if torch.cuda.is_available():
        print(f"  GPU name              : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM (GB)             : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)}")
    print(f"  device                : {device}")
    print("=" * 70)


def train_one_epoch(
    model, cfg, device, optimizer, loss_fn, samples, epoch
) -> dict:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    use_amp = cfg.mixed_precision and device.type == "cuda"
    total_loss, n_loss = 0.0, 0
    agg_stats = {k: 0.0 for k in
                 ("triplets", "skilled_negatives", "random_negatives", "hard",
                  "semi_hard", "easy", "anchors")}

    for b in range(cfg.batches_per_epoch):
        images, writers_arr, is_genuine = build_batch(samples, cfg, epoch, b)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            emb = model.encode(images)
            mined = select_triplets(emb, writers_arr, is_genuine, cfg, epoch)
            if len(mined["negative"]) == 0:
                continue
            d_ap = mined["dist"][mined["anchor"], mined["positive"]]
            d_an = mined["dist"][mined["anchor"], mined["negative"]]
            loss = loss_fn(d_ap.float(), d_an.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_loss += 1
        s = mined["stats"]
        for k in agg_stats:
            agg_stats[k] += s[k]

    n = max(n_loss, 1)
    return {
        "train_loss": total_loss / n,
        "mining": {k: v / cfg.batches_per_epoch for k, v in agg_stats.items()},
        "avg_hard_ratio": agg_stats["hard"] / max(agg_stats["triplets"], 1),
        "avg_semi_ratio": agg_stats["semi_hard"] / max(agg_stats["triplets"], 1),
    }


def validate_pair_based(model, cfg, device, val_pool) -> dict:
    res = E.pair_based_metrics(model, cfg, device, val_pool.all())
    return res["metrics"]


def main() -> None:
    args = parse_args()
    cfg = _resolve_cfg(args)
    ensure_dirs(cfg)
    set_seeds(cfg.seed)
    device = detect_device() if args.device == "auto" else torch.device(args.device)
    print_environment(cfg, device)
    if cfg.cuda_deterministic and device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ---- exact V1 split + data
    print("\nLoading CEDAR index...")
    index = load_writer_index(cfg.dataset_root)
    try:
        assert_exact_v1_split(cfg)
    except WriterSplitMismatch as e:
        raise SystemExit(f"STOP: split mismatch with V1 -> {e}")
    print("  exact V1 split asserted: train/val/test disjoint, writer 7 test-only")
    train_samples = build_samples(index, cfg.train_writers)
    print(f"  train writers: {cfg.train_writers}")

    val_pool = generate_pairs(index, cfg.val_writers, "validation", cfg.seed)
    test_pool = generate_pairs(index, cfg.test_writers, "test", cfg.seed)
    validate_no_writer_leakage(val_pool, test_pool)
    print(f"  val pool: {val_pool.counts()}")
    print(f"  test pool: {test_pool.counts()}")

    # ---- model
    print("\nBuilding V2 model (V1 backbone unchanged)...")
    model = build_model(cfg).to(device)
    print(f"  backbone={cfg.backbone} embed={cfg.embedding_dim} "
          f"pretrained={model.pretrained_loaded}")
    loss_fn = TripletLoss(margin=cfg.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"  triplet margin={cfg.margin} | batch={cfg.writers_per_batch} writers x "
          f"({cfg.genuines_per_writer} gen + {cfg.forgeries_per_writer} forg) = "
          f"{cfg.images_per_batch} img | neg/anchor={cfg.negatives_per_anchor} "
          f"skilled_frac={cfg.skilled_negative_fraction} | batches/epoch={cfg.batches_per_epoch}")

    history = []
    best_eer, best_epoch, patience = float("inf"), -1, 0
    best_state = None
    optimizer_state = None
    t_start = time.time()

    if not args.eval_only:
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            set_seeds(cfg.seed + epoch)
            tr = train_one_epoch(model, cfg, device, optimizer, loss_fn, train_samples, epoch)
            val_met = validate_pair_based(model, cfg, device, val_pool)
            dt = time.time() - t0

            improved = val_met["eer"] < best_eer
            if improved:
                best_eer, best_epoch, patience = val_met["eer"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            row = {
                "epoch": epoch,
                "train_loss": round(tr["train_loss"], 6),
                "val_eer": round(val_met["eer"], 6),
                "val_roc_auc": round(val_met["roc_auc"], 6),
                "val_far_at_eer": round(val_met["far_at_eer"], 6),
                "val_frr_at_eer": round(val_met["frr_at_eer"], 6),
                "val_threshold_eer": round(val_met["eer_threshold"], 6),
                "mined_hard_ratio": round(tr["avg_hard_ratio"], 4),
                "mined_semi_ratio": round(tr["avg_semi_ratio"], 4),
                "time_s": round(dt, 2),
            }
            history.append(row)
            print(
                f"  epoch {epoch:>2}/{cfg.epochs}  train_loss={tr['train_loss']:.4f}  "
                f"val_EER={val_met['eer']:.4f}  val_AUC={val_met['roc_auc']:.4f}  "
                f"hard={tr['avg_hard_ratio']:.2f} semi={tr['avg_semi_ratio']:.2f}  "
                f"[{dt:.0f}s]  {('*best*' if improved else f'patience {patience}')}"
            )
            if patience >= cfg.early_stopping_patience:
                print(f"  early stopping at epoch {epoch}")
                break

        train_seconds = time.time() - t_start
        print(f"\nTraining finished in {train_seconds/60:.1f} min; best epoch = {best_epoch}")

        model.load_state_dict(best_state if best_state is not None else model.state_dict())
        model.to(device)
        optimizer_state = optimizer.state_dict()
    else:
        ckpt_load = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_load["model_state"])
        model.to(device)
        history = ckpt_load.get("train_history", [])
        best_epoch = ckpt_load.get("training_epoch", -1)
        optimizer_state = ckpt_load.get("optimizer_state")
        print(f"\nEVAL-ONLY: loaded checkpoint {cfg.checkpoint_path} (best epoch {best_epoch})")

    # ---- pair-based validation threshold (V1-comparable protocol)
    val_pair = E.pair_based_metrics(model, cfg, device, val_pool.all())
    val_threshold = float(val_pair["metrics"]["eer_threshold"])
    print(f"  pair-based val EER={val_pair['metrics']['eer']:.4f} "
          f"threshold={val_threshold:.4f}")

    # ---- V1 recompute on identical pools (comparison + reproducibility)
    v1_pair = None
    if not args.skip_v1_recompute:
        v1_ckpt = torch.load(cfg.v1_checkpoint_path, map_location="cpu", weights_only=False)
        v1_cfg = ExperimentConfig()
        v1_model = build_model(v1_cfg).to(device)
        v1_model.load_state_dict(v1_ckpt["model_state"])
        v1_val = E.pair_based_metrics(v1_model, v1_cfg, device, val_pool.all())
        v1_test = E.pair_based_apply_frozen(v1_model, v1_cfg, device, test_pool.all(), v1_val["metrics"]["eer_threshold"])
        v1_pair = {"val": v1_val, "test": v1_test}
        v1_test_met = M.full_metrics(v1_test["scores"]["similarity"], v1_test["scores"]["label"])
        print(f"  V1 recomputed (same pools): test AUC={v1_test_met['roc_auc']:.4f}")

    # ---- V2 pair-based test with the pair-based val threshold
    v2_pair_test = E.pair_based_apply_frozen(model, cfg, device, test_pool.all(), val_threshold)
    v2_pair_test_met = M.full_metrics(v2_pair_test["scores"]["similarity"], v2_pair_test["scores"]["label"])

    # ---- multi-reference selection on VALIDATION
    print("\nMulti-reference evaluation: selecting strategy on VALIDATION...")
    val_emb = E.encode_all_writers(model, cfg, device, cfg.val_writers, index)
    selection = E.select_on_validation(val_emb, cfg.val_writers, cfg)
    k_star, s_star, n_star = selection["best_candidate"]
    print(f"  best candidate on validation: K={k_star} strategy={s_star} norm={n_star} "
          f"val EER={selection['validation_metrics']['eer']:.4f} "
          f"skilled FAR@EER={selection['validation_metrics']['skilled_forgery_far_at_eer']:.4f} "
          f"threshold={selection['threshold']:.4f}")

    # ---- freeze + evaluate TEST
    test_emb = E.encode_all_writers(model, cfg, device, cfg.test_writers, index)
    frozen = E.evaluate_test_frozen(test_emb, cfg.test_writers, selection)
    print(f"  TEST (frozen K={k_star}/{s_star}/{n_star}): "
          f"AUC={frozen['metrics']['roc_auc']:.4f} EER={frozen['metrics']['eer']:.4f} "
          f"FAR={frozen['far_at_frozen_threshold']:.4f} FRR={frozen['frr_at_frozen_threshold']:.4f} "
          f"skilled FAR={frozen['skilled_forgery_far_at_frozen_threshold']:.4f}")

    # ---- signer 7
    g7, f7 = test_emb[7]
    s7 = E.signer7_report(g7, f7, val_threshold, frozen)
    g = s7["pair_based_original_7_5"]["genuine_similarity"]
    fo = s7["pair_based_original_7_5"]["forgery_similarity"]
    print("\nSigner 7 (pair-based, original_7_5 reference):")
    print(f"  genuine  sim: mean={g['mean']:.4f} std={g['std']:.4f} [{g['min']:.4f},{g['max']:.4f}]")
    print(f"  forgery  sim: mean={fo['mean']:.4f} std={fo['std']:.4f} [{fo['min']:.4f},{fo['max']:.4f}]")
    print(f"  AUC={s7['pair_based_original_7_5']['metrics']['roc_auc']:.4f} "
          f"EER={s7['pair_based_original_7_5']['metrics']['eer']:.4f} "
          f"FAR@globalVT={s7['pair_based_original_7_5']['far_at_global_val_threshold']:.4f}")

    # ---- checkpoint
    ckpt = {
        "model_version": cfg.model_version,
        "architecture": {"backbone": cfg.backbone, "embedding_dim": cfg.embedding_dim},
        "pretrained": cfg.pretrained,
        "canvas_size": {"width": cfg.canvas_width, "height": cfg.canvas_height},
        "loss": {"type": "triplet", "margin": cfg.margin, "distance": "euclidean on L2 embeddings"},
        "mining": {
            "strategy": "in-batch hard/semi-hard, skilled-forgery focused",
            "writers_per_batch": cfg.writers_per_batch,
            "genuines_per_writer": cfg.genuines_per_writer,
            "forgeries_per_writer": cfg.forgeries_per_writer,
            "negatives_per_anchor": cfg.negatives_per_anchor,
            "skilled_negative_fraction": cfg.skilled_negative_fraction,
        },
        "preprocessing": {"version": cfg.preprocessing_version, "config": cfg.to_dict()},
        "augmentation": {"rotation_deg": cfg.aug_rotation_deg, "translation_px": cfg.aug_translation_px,
                         "scale": (cfg.aug_scale_min, cfg.aug_scale_max)},
        "writer_split": {"train": list(cfg.train_writers), "validation": list(cfg.val_writers),
                         "test": list(cfg.test_writers)},
        "training_epoch": best_epoch,
        "validation_metric_pair_based": {"eer": val_pair["metrics"]["eer"],
                                         "roc_auc": val_pair["metrics"]["roc_auc"],
                                         "threshold": val_threshold},
        "validation_multi_reference": {
            "best_candidate": [k_star, s_star, n_star],
            "eer": selection["validation_metrics"]["eer"],
            "threshold": selection["threshold"],
        },
        "config": cfg.to_dict(),
        "model_state": best_state if best_state is not None else model.state_dict(),
        "optimizer_state": optimizer_state,
        "train_history": history,
        "seed": cfg.seed,
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.checkpoint_path)
    print(f"\nCheckpoint saved: {cfg.checkpoint_path}")

    # ---- reports
    write_reports(cfg, history, val_pair, v2_pair_test, v2_pair_test_met, val_threshold,
                  selection, frozen, s7, v1_pair, val_pool, test_pool)
    print("\n=== FINAL SUMMARY (V2) ===")
    t = frozen["metrics"]
    print(f"  PAIR-BASED test AUC            = {v2_pair_test_met['roc_auc']:.4f}")
    print(f"  PAIR-BASED test EER            = {v2_pair_test_met['eer']:.4f}")
    print(f"  PAIR-BASED FAR/FRR @ val thr   = {v2_pair_test['far']:.4f} / {v2_pair_test['frr']:.4f}")
    print(f"  PAIR-BASED skilled FAR         = {v2_pair_test['skilled_forgery_far']:.4f}")
    print(f"  PAIR-BASED impostor FAR        = {v2_pair_test['random_impostor_far']:.4f}")
    print(f"  MULTI-REF (frozen K={k_star} {s_star}/{n_star}) test AUC = {t['roc_auc']:.4f}")
    print(f"  MULTI-REF test EER             = {t['eer']:.4f}")
    print(f"  MULTI-REF FAR/FRR @ frozen thr = {frozen['far_at_frozen_threshold']:.4f} / {frozen['frr_at_frozen_threshold']:.4f}")
    print(f"  MULTI-REF skilled FAR          = {frozen['skilled_forgery_far_at_frozen_threshold']:.4f}")
    print(f"  signer-7 pair-based AUC        = {s7['pair_based_original_7_5']['metrics']['roc_auc']:.4f}")
    print(f"  signer-7 pair-based EER        = {s7['pair_based_original_7_5']['metrics']['eer']:.4f}")
    print(f"  checkpoint                     = {cfg.checkpoint_path}")


def write_reports(cfg, history, val_pair, v2_pair_test, v2_test_met, val_threshold,
                  selection, frozen, signer7, v1_pair, val_pool, test_pool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rdir = Path(cfg.reports_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    # training_history.csv
    with open(rdir / "training_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    # pair-based per-split CSVs
    for name, scores in (("validation_pairs", val_pair["scores"]), ("test_pairs", v2_pair_test["scores"])):
        with open(rdir / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["writer", "pair_type", "label", "similarity", "distance"])
            pairs = val_pool.all() if name.startswith("validation") else test_pool.all()
            for p, s, d in zip(pairs, scores["similarity"], scores["distance"]):
                w.writerow([p.writer, p.pair_type, p.label, f"{s:.6f}", f"{d:.6f}"])

    # per-writer test table
    with open(rdir / "per_writer_test.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frozen["per_writer"][0].keys()))
        w.writeheader()
        w.writerows(frozen["per_writer"])

    # multi-reference table (all candidates, validation)
    with open(rdir / "multi_reference_validation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["K", "strategy", "norm", "auc", "eer", "eer_threshold", "skilled_far_at_eer", "accuracy_at_eer"])
        for cand_key, met in selection["table"].items():
            k, s, n = cand_key.split("-")
            w.writerow([k, s, n, f"{met['roc_auc']:.6f}", f"{met['eer']:.6f}",
                        f"{met['eer_threshold']:.6f}", f"{met['skilled_forgery_far_at_eer']:.6f}",
                        f"{met['accuracy_at_eer']:.6f}"])

    # metrics.json
    payload = {
        "model_version": cfg.model_version,
        "config": cfg.to_dict(),
        "pair_based": {
            "validation": {"eer": val_pair["metrics"]["eer"], "auc": val_pair["metrics"]["roc_auc"],
                           "threshold": val_threshold},
            "test": v2_test_met,
            "test_at_validation_threshold": {
                "far": v2_pair_test["far"], "frr": v2_pair_test["frr"],
                "accuracy": v2_pair_test["accuracy"],
                "skilled_forgery_far": v2_pair_test["skilled_forgery_far"],
                "random_impostor_far": v2_pair_test["random_impostor_far"],
            },
        },
        "multi_reference": {
            "best_candidate": [frozen["best_candidate"][0], frozen["best_candidate"][1], frozen["best_candidate"][2]],
            "validation_eer": selection["validation_metrics"]["eer"],
            "validation_threshold": selection["threshold"],
            "test": frozen["metrics"],
            "test_at_frozen_threshold": {
                "far": frozen["far_at_frozen_threshold"],
                "frr": frozen["frr_at_frozen_threshold"],
                "accuracy": frozen["accuracy_at_frozen_threshold"],
                "skilled_forgery_far": frozen["skilled_forgery_far_at_frozen_threshold"],
            },
            "per_writer": frozen["per_writer"],
            "validation_candidates": selection["table"],
        },
        "signer7": signer7,
        "v1_recomputed_pair_based": None if v1_pair is None else {
            "test_metrics": M.full_metrics(v1_pair["test"]["scores"]["similarity"], v1_pair["test"]["scores"]["label"]),
            "test_at_validation_threshold": {
                "far": v1_pair["test"]["far"], "frr": v1_pair["test"]["frr"],
                "accuracy": v1_pair["test"]["accuracy"],
                "skilled_forgery_far": v1_pair["test"]["skilled_forgery_far"],
                "random_impostor_far": v1_pair["test"]["random_impostor_far"],
            },
        },
    }
    (rdir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # plots
    fig, ax = plt.subplots(figsize=(6, 6))
    ts = v2_pair_test["scores"]
    fpr, tpr, _ = M.roc_curve(ts["similarity"], ts["label"])
    ax.plot(fpr, tpr, label=f"V2 pair-based test ROC (AUC={v2_test_met['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FAR"); ax.set_ylabel("TPR")
    ax.set_title(f"V2 pair-based test ROC — {cfg.model_version}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rdir / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    fs = frozen["all_scores"]
    ax.hist(fs["score"][fs["label"] == 1], bins=50, alpha=0.5, color="tab:green", label="genuine queries", density=True)
    ax.hist(fs["score"][fs["label"] == 0], bins=50, alpha=0.5, color="tab:red", label="skilled forgery queries", density=True)
    ax.axvline(frozen["threshold"], color="black", ls="--",
               label=f"frozen validation threshold = {frozen['threshold']:.3f}")
    ax.set_xlabel("aggregated score")
    ax.set_title(f"V2 multi-reference test scores — {cfg.model_version}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(rdir / "multi_reference_scores.png", dpi=150); plt.close(fig)

    print(f"\nReports written to {rdir}")


if __name__ == "__main__":
    main()