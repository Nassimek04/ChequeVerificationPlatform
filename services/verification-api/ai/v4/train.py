"""V4 training — domain-aware (A: augmentation, B: synthetic).

Usage:
  python -m ai.v4.train --variant a
  python -m ai.v4.train --variant b
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ai import metrics as M
from ai.dataset import load_writer_index, validate_no_writer_leakage, generate_pairs
from ai.v4.config import get_v4_config, ensure_dirs
from ai.v4.dataset import assert_exact_v1_split, build_samples, preprocess_with_v4_augment, get_synthetic_extracted_gray, stable_seed
from ai.v4.model import build_model
from ai.v4.losses import TripletLoss
from ai.v2.mining import select_triplets
from ai.v2.dataset import rng_for

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def detect_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_batch_v4(samples, cfg, epoch, batch_idx):
    rng = rng_for(cfg.seed, "batch", epoch, batch_idx)
    train_writers = list(cfg.train_writers)
    writers = rng.sample(train_writers, cfg.writers_per_batch)
    entries = []  # (writer, is_genuine, path, clean_gray)
    for w in writers:
        ws = samples[w]
        g_idx = rng.sample(range(ws.n_genuine), cfg.genuines_per_writer)
        f_idx = rng.sample(range(ws.n_forgery), cfg.forgeries_per_writer)
        for i in g_idx:
            entries.append((w, True, ws.genuine_paths[i]))
        for i in f_idx:
            entries.append((w, False, ws.forgery_paths[i]))
    rng.shuffle(entries)
    images = []
    writers_arr = []
    is_genuine = []
    for pos, (w, gen, path) in enumerate(entries):
        # Decide synthetic for V4-B
        use_synthetic = False
        if cfg.variant == "b":
            # 50% synthetic for train writers only
            # Use deterministic per-image per-epoch seed
            if rng.random() < cfg.synthetic_ratio:
                use_synthetic = True
        # Load gray
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot decode {path}")
        if use_synthetic:
            # Generate synthetic extracted gray deterministically
            seed_key = f"synth-{path.name}-{epoch}-{batch_idx}-{pos}"
            gray = get_synthetic_extracted_gray(gray, seed_key)
        # Preprocess with V4 augmentation
        seed_img = (epoch * 1_000_003 + batch_idx * 1000 + pos) % (2**31)
        # For V4-A, augment True will apply domain augmentations; for V4-B also
        # For clean without synthetic, still apply augmentation
        tensor = preprocess_with_v4_augment(gray, cfg, seed_img, augment=True)
        images.append(tensor)
        writers_arr.append(w)
        is_genuine.append(gen)
    images = torch.stack(images)
    return images, np.asarray(writers_arr, dtype=np.int64), np.asarray(is_genuine, dtype=bool)

def train_one_epoch(model, cfg, device, optimizer, loss_fn, samples, epoch):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device.type == "cuda")
    use_amp = cfg.mixed_precision and device.type == "cuda"
    total_loss, n_loss = 0.0, 0
    agg = {k:0.0 for k in ("triplets","skilled_negatives","random_negatives","hard","semi_hard","easy","anchors")}
    for b in range(cfg.batches_per_epoch):
        images, writers_arr, is_genuine = build_batch_v4(samples, cfg, epoch, b)
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            emb = model.encode(images)
            mined = select_triplets(emb, writers_arr, is_genuine, cfg, epoch)
            if len(mined["negative"])==0:
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
        for k in agg:
            agg[k] += s.get(k,0)
    n = max(n_loss,1)
    return {"train_loss": total_loss/n, "mining": {k: v/cfg.batches_per_epoch for k,v in agg.items()}, "avg_hard_ratio": agg["hard"]/max(agg["triplets"],1), "avg_semi_ratio": agg["semi_hard"]/max(agg["triplets"],1)}

# Validation helpers: we need to encode validation writers in both clean and extracted domains
# For efficiency, pre-generate extracted validation sets? Instead generate on the fly per epoch with same seed (deterministic) - but we want consistent validation across epochs, so use fixed seed per validation image (epoch-independent)
# We'll create helper that encodes validation writers with clean and extracted

def encode_validation_sets(model, cfg, device, index, writers, use_extracted=False):
    # Encode all genuines and forgeries for writers in given domain
    # For extracted, generate synthetic cheque extraction deterministically with fixed seed (not epoch-dependent)
    model.eval()
    emb_map = {}  # writer -> (gen_emb Tensor, forg_emb Tensor)
    with torch.inference_mode():
        for w in writers:
            # Get sorted paths
            import re
            g_paths = sorted(index[w].originals, key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)))
            f_paths = sorted(index[w].forgeries, key=lambda p: int(re.search(r"_(\d+)\.png$", p.name).group(1)))
            g_embs=[]
            for p in g_paths:
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if use_extracted:
                    gray = get_synthetic_extracted_gray(gray, f"val-extracted-{w}-{p.name}-fixed")
                tensor = preprocess_with_v4_augment(gray, cfg, seed=0, augment=False)  # no augment for eval
                emb = model.encode(tensor.unsqueeze(0).to(device)).cpu()
                g_embs.append(emb[0])
            f_embs=[]
            for p in f_paths:
                gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if use_extracted:
                    gray = get_synthetic_extracted_gray(gray, f"val-extracted-{w}-{p.name}-fixed")
                tensor = preprocess_with_v4_augment(gray, cfg, seed=0, augment=False)
                emb = model.encode(tensor.unsqueeze(0).to(device)).cpu()
                f_embs.append(emb[0])
            emb_map[w] = (torch.stack(g_embs), torch.stack(f_embs))
    return emb_map

def evaluate_extracted_validation(emb_clean_ref_map, emb_ext_map, writers, k=1, agg="max", norm="raw"):
    # Evaluate clean ref -> extracted candidate (primary extracted-domain metric)
    # Use same logic as domain alignment: refs clean, queries extracted
    scores_gen=[]; scores_forg=[]
    for w in writers:
        refs = emb_clean_ref_map[w][0][:k]  # clean genuines refs
        q_gen = emb_ext_map[w][0][k:]  # extracted genuines queries
        q_forg = emb_ext_map[w][1]  # extracted forgeries
        # Compute mu/sigma for z
        if norm=="z" and k>=3:
            sims_ref = refs @ refs.T
            idx=torch.triu_indices(k,k,offset=1)
            vals=sims_ref[idx[0],idx[1]]
            mu=float(vals.mean()); sigma=float(max(vals.std(unbiased=False),1e-6))
        else:
            mu,sigma=0.0,1.0
        def apply(q):
            sims=q @ refs.T
            if agg=="prototype":
                proto=F.normalize(refs.mean(dim=0),p=2,dim=0)
                raw=q @ proto
            else:
                # Use same aggregate as V2
                if agg=="max": raw=sims.max(dim=1).values
                elif agg=="mean": raw=sims.mean(dim=1)
                elif agg=="median": raw=sims.median(dim=1).values
                elif agg=="top2_mean": raw=sims.topk(2,dim=1).values.mean(dim=1)
                else: raise ValueError(agg)
            if norm=="z":
                return (raw - mu)/sigma
            return raw
        g_scores=apply(q_gen)
        f_scores=apply(q_forg)
        scores_gen.append(g_scores.numpy())
        scores_forg.append(f_scores.numpy())
    sg=np.concatenate(scores_gen) if scores_gen else np.array([])
    sf=np.concatenate(scores_forg) if scores_forg else np.array([])
    scores=np.concatenate([sg,sf]) if sg.size else np.array([])
    labels=np.concatenate([np.ones_like(sg), np.zeros_like(sf)]) if sg.size else np.array([])
    met=M.full_metrics(scores, labels) if scores.size else {"roc_auc":0.5,"eer":1.0}
    return met, sg, sf, scores, labels

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a","b"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    cfg=get_v4_config(args.variant)
    if args.epochs is not None:
        cfg=cfg.__class__(**{**cfg.__dict__, "epochs":args.epochs})
    ensure_dirs(cfg)
    set_seeds(cfg.seed)
    device=torch.device("cuda") if (args.device=="auto" and torch.cuda.is_available()) else torch.device(args.device if args.device!="auto" else "cpu")
    print("="*70)
    print(f"V4-{cfg.variant.upper()} training")
    print(f" device {device} checkpoint {cfg.checkpoint_path}")
    print("="*70)
    index=load_writer_index(cfg.dataset_root)
    assert_exact_v1_split(cfg)
    train_samples=build_samples(index, cfg.train_writers)
    print(f" train writers {cfg.train_writers}")

    model=build_model(cfg).to(device)
    loss_fn=TripletLoss(margin=cfg.margin)
    optimizer=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history=[]
    best_auc=-1
    best_eer=float("inf")
    best_epoch=-1
    best_state=None
    patience=0

    t_start=time.time()
    for epoch in range(1, cfg.epochs+1):
        t0=time.time()
        set_seeds(cfg.seed+epoch)
        tr=train_one_epoch(model, cfg, device, optimizer, loss_fn, train_samples, epoch)
        # Validation: extracted-domain AUC (primary)
        emb_clean=encode_validation_sets(model, cfg, device, index, cfg.val_writers, use_extracted=False)
        emb_ext=encode_validation_sets(model, cfg, device, index, cfg.val_writers, use_extracted=True)
        # For validation metric, use clean_ref -> extracted candidate, K=1 max raw (primary)
        met,_,_,_,_=evaluate_extracted_validation(emb_clean, emb_ext, list(cfg.val_writers), k=1, agg="max", norm="raw")
        met_clean2,_,_,_,_=evaluate_extracted_validation(emb_clean, emb_clean, list(cfg.val_writers), k=1, agg="max", norm="raw")

        dt=time.time()-t0
        auc=met["roc_auc"]; eer=met["eer"]
        improved = (auc > best_auc) or (auc==best_auc and eer < best_eer)
        if improved:
            best_auc, best_eer, best_epoch = auc, eer, epoch
            best_state={k:v.clone() for k,v in model.state_dict().items()}
            patience=0
        else:
            patience+=1

        row={"epoch":epoch,"train_loss":round(tr["train_loss"],6),"val_ext_auc":round(auc,6),"val_ext_eer":round(eer,6),"val_clean_auc":round(met_clean2["roc_auc"],6),"val_clean_eer":round(met_clean2["eer"],6),"hard_ratio":round(tr["avg_hard_ratio"],4),"semi_ratio":round(tr["avg_semi_ratio"],4),"time_s":round(dt,2)}
        history.append(row)
        print(f" epoch {epoch:>2}/{cfg.epochs} loss={tr['train_loss']:.4f} val_ext AUC={auc:.4f} EER={eer:.4f} clean AUC={met_clean2['roc_auc']:.4f} {'*best*' if improved else f'patience {patience}'} [{dt:.0f}s]")
        if patience >= cfg.early_stopping_patience:
            print(f" early stopping at {epoch}")
            break

    train_seconds=time.time()-t_start
    print(f"Training finished {train_seconds/60:.1f} min best epoch {best_epoch} AUC {best_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt={
        "model_version":cfg.model_version,
        "architecture":{"backbone":cfg.backbone,"embedding_dim":cfg.embedding_dim},
        "pretrained":cfg.pretrained,
        "canvas_size":{"width":cfg.canvas_width,"height":cfg.canvas_height},
        "loss":{"type":"triplet","margin":cfg.margin},
        "mining":{"writers_per_batch":cfg.writers_per_batch,"genuines_per_writer":cfg.genuines_per_writer,"forgeries_per_writer":cfg.forgeries_per_writer,"negatives_per_anchor":cfg.negatives_per_anchor,"skilled_negative_fraction":cfg.skilled_negative_fraction},
        "augmentation":{"scale":(cfg.aug_scale_min,cfg.aug_scale_max),"aspect":(cfg.aug_aspect_min,cfg.aug_aspect_max),"blur":(cfg.aug_blur_min,cfg.aug_blur_max,cfg.aug_blur_prob),"darken":(cfg.aug_darken_min,cfg.aug_darken_max,cfg.aug_darken_prob)},
        "variant":cfg.variant,
        "synthetic_ratio":cfg.synthetic_ratio if cfg.variant=="b" else 0,
        "writer_split":{"train":list(cfg.train_writers),"validation":list(cfg.val_writers),"test":list(cfg.test_writers)},
        "training_epoch":best_epoch,
        "validation_metric":{"ext_auc":best_auc,"ext_eer":best_eer},
        "config":cfg.to_dict(),
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "train_history":history,
        "seed":cfg.seed,
    }
    torch.save(ckpt, cfg.checkpoint_path)
    print(f"Checkpoint saved {cfg.checkpoint_path}")

    # Save training history csv
    import os
    os.makedirs(cfg.reports_dir, exist_ok=True)
    with open(Path(cfg.reports_dir)/f"v4{cfg.variant}_training_history.csv","w",newline="") as fh:
        w=csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

if __name__=="__main__":
    main()
