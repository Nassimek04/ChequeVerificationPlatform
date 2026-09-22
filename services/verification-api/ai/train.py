"""Training entry point for the writer-independent Siamese V1 experiment.

Run (from services/verification-api):

    ..\\ai\\.venv\\Scripts\\python.exe -m ai.train            # or whatever venv python

Pipeline: dataset scan -> writer split (assert disjoint) -> pair pools ->
training (AdamW + contrastive + AMP, best-checkpoint by validation EER) ->
unseen-writer evaluation -> signer-7 diagnostic -> reports.

EXPERIMENTAL: nothing here touches the FastAPI service (app/).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .config import ExperimentConfig, detect_device, ensure_dirs
from .dataset import (
    PairDataset,
    compute_writer_split,
    default_collate,
    generate_pairs,
    load_writer_index,
    sample_balanced_epoch,
    validate_no_identical_positive_pairs,
    validate_no_writer_leakage,
)
from .evaluate import (
    run_evaluation,
    run_signer7_diagnostic,
    score_pairs,
    write_reports,
)
from .losses import ContrastiveLoss
from .model import build_model


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI signature verification V1 (writer-independent)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train-pairs", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--eval-only", action="store_true",
                   help="load the saved checkpoint and re-run evaluation + reports only")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="checkpoint path for --eval-only (default: config default)")
    return p.parse_args()


def _resolve_cfg(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    kw = {}
    if args.epochs is not None:
        kw["epochs"] = args.epochs
    if args.batch_size is not None:
        kw["batch_size"] = args.batch_size
    if args.lr is not None:
        kw["lr"] = args.lr
    if args.train_pairs is not None:
        kw["train_pairs_per_epoch"] = args.train_pairs
    if args.no_pretrained:
        kw["pretrained"] = False
    if args.tag:
        kw["model_version"] = args.tag
    cfg = ExperimentConfig(**kw)
    return cfg


def print_environment(cfg: ExperimentConfig, device: torch.device) -> None:
    print("=" * 70)
    print("ENVIRONMENT")
    print(f"  python torch          : {torch.__version__}")
    print(f"  CUDA available        : {torch.cuda.is_available()}")
    print(f"  CUDA build            : {torch.version.cuda or 'n/a'}")
    if torch.cuda.is_available():
        print(f"  GPU name              : {torch.cuda.get_device_name(0)}")
        print(f"  GPU capability        : {torch.cuda.get_device_capability(0)}")
        print(f"  VRAM (GB)             : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 2)}")
    print(f"  selected device       : {device}")
    print("=" * 70)


def train_one_epoch(
    model,
    cfg: ExperimentConfig,
    device: torch.device,
    optimizer,
    loss_fn,
    epoch: int,
    train_pool,
) -> float:
    pairs = sample_balanced_epoch(train_pool, cfg.train_pairs_per_epoch, cfg.seed, epoch)
    validate_no_identical_positive_pairs(pairs)
    ds = PairDataset(pairs, cfg, split="train", epoch=epoch)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=default_collate
    )
    model.train()
    total, n = 0.0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    use_amp = cfg.mixed_precision and device.type == "cuda"
    for img_a, img_b, y in loader:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            e1, e2 = model(img_a, img_b)
            loss = loss_fn(e1, e2, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.inference_mode()
def validate(model, cfg, device, val_pairs, loss_fn) -> dict:
    """Validation: EER/ROC metrics + actual contrastive loss on the val pool."""
    scores = score_pairs(model, val_pairs.all(), device, cfg)
    metrics = M.full_metrics(scores["similarity"], scores["label"])

    from .dataset import PairDataset, default_collate

    ds = PairDataset(val_pairs.all(), cfg, split="eval")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=default_collate)
    model.eval()
    total, n = 0.0, 0
    for img_a, img_b, y in loader:
        e1, e2 = model(img_a.to(device), img_b.to(device))
        loss = loss_fn(e1, e2, y.to(device))
        total += loss.item() * len(y)
        n += len(y)
    metrics["loss"] = total / n
    return metrics


def print_final_summary(result: dict, signer7: dict, val_threshold: float) -> None:
    print("\n=== FINAL SUMMARY ===")
    t = result["test"]["metrics"]
    print(f"  unseen-writer test ROC-AUC       = {t['roc_auc']:.4f}")
    print(f"  unseen-writer test EER           = {t['eer']:.4f}")
    print(f"  test EER threshold (descriptive) = {t['eer_threshold']:.4f}")
    print(f"  OPERATIONAL validation threshold = {val_threshold:.4f}")
    print(f"  test FAR @ val threshold         = {result['test']['far_at_val_threshold']:.4f}")
    print(f"  test FRR @ val threshold         = {result['test']['frr_at_val_threshold']:.4f}")
    print(f"  test accuracy @ val threshold    = {result['test']['accuracy_at_val_threshold']:.4f}")
    print(f"  skilled-forgery FAR @ val thr    = {result['test']['skilled_forgery_far_at_val_threshold']:.4f}")
    print(f"  random-impostor FAR @ val thr    = {result['test']['random_impostor_far_at_val_threshold']:.4f}")
    print(f"  signer7 EER (diagnostic)         = {signer7.get('metrics', {}).get('eer', 'n/a')}")
    print(f"  checkpoint                       = {ExperimentConfig().checkpoint_path}")


def _prepare_data(cfg: ExperimentConfig):
    """Index + split + pools (shared by train and eval-only flows)."""
    print("\nLoading CEDAR index...")
    index = load_writer_index(cfg.dataset_root)
    writer_ids = sorted(index.keys())
    print(f"  writers found: {len(writer_ids)}")

    train_writers, val_writers, test_writers = compute_writer_split(
        writer_ids, cfg.seed, cfg.train_ratio, cfg.val_ratio, cfg.guarantee_test_writer
    )
    s_train, s_val, s_test = set(train_writers), set(val_writers), set(test_writers)
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test)
    assert cfg.guarantee_test_writer in s_test

    train_pool = generate_pairs(
        index, train_writers, "train", cfg.seed,
        impostor_cap_per_writer=192, impostor_writer_pairs=40,
    )
    val_pool = generate_pairs(index, val_writers, "validation", cfg.seed)
    test_pool = generate_pairs(index, test_writers, "test", cfg.seed)
    validate_no_writer_leakage(train_pool, test_pool)
    validate_no_identical_positive_pairs(train_pool.positive)
    validate_no_identical_positive_pairs(val_pool.positive)
    validate_no_identical_positive_pairs(test_pool.positive)
    return index, train_writers, val_writers, test_writers, train_pool, val_pool, test_pool


def main_eval_only(args: argparse.Namespace) -> None:
    """Load the saved checkpoint and re-run the full evaluation + reports."""
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ExperimentConfig().checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_kwargs = {
        k: v
        for k, v in ckpt["config"].items()
        if k in ExperimentConfig.__dataclass_fields__
    }
    cfg = ExperimentConfig(**cfg_kwargs)
    cfg = dataclasses.replace(
        cfg,
        dataset_root=Path(cfg.dataset_root),
        checkpoint_path=Path(cfg.checkpoint_path),
        reports_dir=Path(cfg.reports_dir),
    )
    ensure_dirs(cfg)
    set_seeds(cfg.seed)
    device = detect_device() if args.device == "auto" else torch.device(args.device)
    print_environment(cfg, device)

    index, tw, vw, tst, train_pool, val_pool, test_pool = _prepare_data(cfg)
    print("  split / pools rebuilt (same deterministic protocol as training)")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  checkpoint loaded: {ckpt_path} (best epoch {ckpt.get('training_epoch')})")

    result = run_evaluation(model, cfg, device, val_pool, test_pool)
    val_threshold = float(result["validation"]["threshold"])
    signer7 = run_signer7_diagnostic(model, cfg, device, index, val_threshold)
    reports = write_reports(cfg, result, signer7, ckpt.get("train_history", []), test_pool, val_pool)
    print("\n=== REPORTS WRITTEN ===")
    for k, p in reports.items():
        print(f"  {k}: {p}")
    print_final_summary(result, signer7, val_threshold)


def main() -> None:
    args = parse_args()
    if args.eval_only:
        main_eval_only(args)
        return
    cfg = _resolve_cfg(args)
    ensure_dirs(cfg)
    set_seeds(cfg.seed)
    device = detect_device() if args.device == "auto" else torch.device(args.device)
    print_environment(cfg, device)

    if cfg.cuda_deterministic and device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # ------------------------------------------------------------------ data
    print("\nLoading CEDAR index...")
    index = load_writer_index(cfg.dataset_root)
    writer_ids = sorted(index.keys())
    print(f"  writers found: {len(writer_ids)}")
    for w in sorted(index):
        wi = index[w]
        print(f"    writer {w:>3}: {wi.n_originals} originals, {wi.n_forgeries} forgeries")

    train_writers, val_writers, test_writers = compute_writer_split(
        writer_ids, cfg.seed, cfg.train_ratio, cfg.val_ratio, cfg.guarantee_test_writer
    )
    print("\nWRITER SPLIT (whole writers, deterministic seed=%d)" % cfg.seed)
    print(f"  train      ({len(train_writers)}): {train_writers}")
    print(f"  validation ({len(val_writers)}): {val_writers}")
    print(f"  test       ({len(test_writers)}): {test_writers}")

    s_train, s_val, s_test = set(train_writers), set(val_writers), set(test_writers)
    assert s_train.isdisjoint(s_val) and s_train.isdisjoint(s_test) and s_val.isdisjoint(s_test)
    assert cfg.guarantee_test_writer in s_test
    print("  ASSERT disjoint splits: OK; writer 7 in test: OK")

    print("\nGenerating pair pools...")
    train_pool = generate_pairs(
        index, train_writers, "train", cfg.seed,
        impostor_cap_per_writer=192, impostor_writer_pairs=40,
    )
    val_pool = generate_pairs(index, val_writers, "validation", cfg.seed)
    test_pool = generate_pairs(index, test_writers, "test", cfg.seed)
    validate_no_writer_leakage(train_pool, test_pool)
    validate_no_identical_positive_pairs(train_pool.positive)
    validate_no_identical_positive_pairs(val_pool.positive)
    validate_no_identical_positive_pairs(test_pool.positive)

    for name, pool in (("TRAIN", train_pool), ("VALIDATION", val_pool), ("TEST", test_pool)):
        c = pool.counts()
        print(
            f"  {name:<10} positive={c['positive']:>6} "
            f"skilled_forgery={c['skilled_forgery']:>6} "
            f"random_impostor={c['random_impostor']:>6} total={c['total']:>6}"
        )

    # ------------------------------------------------------------------ model
    print("\nBuilding model...")
    model = build_model(cfg).to(device)
    print(f"  backbone: {cfg.backbone} | embedding dim: {cfg.embedding_dim}")
    print(f"  pretrained ImageNet weights loaded: {model.pretrained_loaded}")

    loss_fn = ContrastiveLoss(margin=cfg.margin)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    print(
        f"  optimizer AdamW lr={cfg.lr} wd={cfg.weight_decay} | "
        f"loss=contrastive margin={cfg.margin} | epochs={cfg.epochs} "
        f"batch={cfg.batch_size} pairs/epoch={cfg.train_pairs_per_epoch}"
    )

    # ------------------------------------------------------------------ train
    history: list = []
    best_eer, best_epoch, patience = float("inf"), -1, 0
    best_state = None
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        set_seeds(cfg.seed + epoch)
        tr_loss = train_one_epoch(model, cfg, device, optimizer, loss_fn, epoch, train_pool)
        val_metrics = validate(model, cfg, device, val_pool, loss_fn)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_eer": round(val_metrics["eer"], 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6),
            "val_far_at_eer": round(val_metrics["far_at_eer"], 6),
            "val_frr_at_eer": round(val_metrics["frr_at_eer"], 6),
            "val_threshold_eer": round(val_metrics["eer_threshold"], 6),
            "time_s": round(dt, 2),
        }
        history.append(row)

        improved = val_metrics["eer"] < best_eer
        if improved:
            best_eer = val_metrics["eer"]
            best_epoch = epoch
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        print(
            f"  epoch {epoch:>2}/{cfg.epochs}  train_loss={tr_loss:.4f}  "
            f"val_EER={val_metrics['eer']:.4f}  val_AUC={val_metrics['roc_auc']:.4f}  "
            f"val_thr={val_metrics['eer_threshold']:.4f}  [{dt:.1f}s]  "
            f"{'*best*' if improved else f'patience {patience}'}"
        )

        if patience >= cfg.early_stopping_patience:
            print(f"  early stopping triggered at epoch {epoch}")
            break

    train_seconds = time.time() - t_start
    print(f"\nTraining finished in {train_seconds/60:.1f} min; best epoch = {best_epoch}")

    # Restore best weights and compute the validation threshold.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    val_scores = score_pairs(model, val_pool.all(), device, cfg)
    val_metrics = M.full_metrics(val_scores["similarity"], val_scores["label"])
    val_threshold = float(val_metrics["eer_threshold"])
    print(f"  best validation EER = {val_metrics['eer']:.4f} (epoch {best_epoch})")
    print(f"  VALIDATION-DERIVED THRESHOLD (similarity) = {val_threshold:.4f}")

    # ------------------------------------------------------------------ checkpoint
    checkpoint = {
        "model_version": cfg.model_version,
        "architecture": {"backbone": cfg.backbone, "embedding_dim": cfg.embedding_dim},
        "canvas_size": {"width": cfg.canvas_width, "height": cfg.canvas_height},
        "training_epoch": best_epoch,
        "validation_metric": {
            "eer": val_metrics["eer"],
            "roc_auc": val_metrics["roc_auc"],
            "threshold": val_threshold,
        },
        "threshold_derived_from_validation": val_threshold,
        "preprocessing": {"version": cfg.preprocessing_version, "config": cfg.to_dict()},
        "config": cfg.to_dict(),
        "model_state": best_state if best_state is not None else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_history": history,
        "train_writers": train_writers,
        "val_writers": val_writers,
        "test_writers": test_writers,
        "seed": cfg.seed,
    }
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, cfg.checkpoint_path)
    print(f"\nCheckpoint saved: {cfg.checkpoint_path}")

    # training_history.csv
    hist_path = cfg.reports_dir / "training_history.csv"
    with open(hist_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()) if history else [])
        w.writeheader()
        w.writerows(history)
    print(f"training_history.csv -> {hist_path}")

    # ------------------------------------------------------------------ evaluate
    result = run_evaluation(model, cfg, device, val_pool, test_pool)
    signer7 = run_signer7_diagnostic(
        model, cfg, device, index, val_threshold
    )

    print("\n=== SIGNER #7 SECONDARY BENCHMARK ===")
    if "skipped" in signer7:
        print("  " + signer7["skipped"])
    else:
        g = signer7["genuine_similarity"]
        f = signer7["forgery_similarity"]
        print(
            f"  genuine  similarity: mean={g['mean']:.4f} std={g['std']:.4f} "
            f"[{g['min']:.4f}, {g['max']:.4f}] n={signer7['n_genuine']}"
        )
        print(
            f"  forgery  similarity: mean={f['mean']:.4f} std={f['std']:.4f} "
            f"[{f['min']:.4f}, {f['max']:.4f}] n={signer7['n_forgery']}"
        )
        m = signer7["metrics"]
        print(
            f"  signer7 AUC={m['roc_auc']:.4f} EER={m['eer']:.4f} "
            f"FAR@vt={signer7['far_at_val_threshold']:.4f} "
            f"FRR@vt={signer7['frr_at_val_threshold']:.4f}"
        )
        print("  interpretation:", signer7["interpretation"])

    reports = write_reports(cfg, result, signer7, history, test_pool, val_pool)
    print("\n=== REPORTS WRITTEN ===")
    for k, p in reports.items():
        print(f"  {k}: {p}")

    # ------------------------------------------------------------------ summary
    print("\n=== FINAL SUMMARY ===")
    t = result["test"]["metrics"]
    print(f"  unseen-writer test ROC-AUC       = {t['roc_auc']:.4f}")
    print(f"  unseen-writer test EER           = {t['eer']:.4f}")
    print(f"  test EER threshold (descriptive) = {t['eer_threshold']:.4f}")
    print(f"  OPERATIONAL validation threshold = {val_threshold:.4f}")
    print(f"  test FAR @ val threshold         = {result['test']['far_at_val_threshold']:.4f}")
    print(f"  test FRR @ val threshold         = {result['test']['frr_at_val_threshold']:.4f}")
    print(f"  test accuracy @ val threshold    = {result['test']['accuracy_at_val_threshold']:.4f}")
    print(f"  skilled-forgery FAR @ val thr    = {result['test']['skilled_forgery_far_at_val_threshold']:.4f}")
    print(f"  random-impostor FAR @ val thr    = {result['test']['random_impostor_far_at_val_threshold']:.4f}")
    print(f"  signer7 EER (diagnostic)         = {signer7.get('metrics', {}).get('eer', 'n/a')}")
    print(f"  checkpoint                       = {cfg.checkpoint_path}")
    print(f"  total wall time                  = {train_seconds/60:.1f} min (training only)")


if __name__ == "__main__":
    main()