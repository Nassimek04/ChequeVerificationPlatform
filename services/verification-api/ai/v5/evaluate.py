"""V5 REAL evaluation — K1/3/5 mean raw cosine, genuine vs skilled vs random, locked signer 7."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
import numpy as np
import torch
from .config import V5Config
from ai import metrics as M

def load_model(cfg: V5Config, device):
    from ai.model import SiameseResNet18
    import dataclasses
    @dataclasses.dataclass
    class Cfg2:
        embedding_dim=cfg.embedding_dim
        pretrained=False
        canvas_width=cfg.canvas_width
        canvas_height=cfg.canvas_height
    m = SiameseResNet18(Cfg2())
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(device); m.eval()
    return m

def load_v2_model(device):
    from ai.v2.config import V2Config
    from ai.model import SiameseResNet18
    import dataclasses
    cfg2=V2Config()
    @dataclasses.dataclass
    class Cfg2:
        embedding_dim=cfg2.embedding_dim
        pretrained=False
        canvas_width=cfg2.canvas_width
        canvas_height=cfg2.canvas_height
    m=SiameseResNet18(Cfg2())
    ckpt=torch.load(cfg2.checkpoint_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(device); m.eval()
    return m, cfg2

def encode_cedar_image(path: Path, cfg, device, model):
    from ai.preprocessing import preprocess
    # Use V5's canvas size (256x128) same as V1/V2
    # For CEDAR, we can use the standard preprocess but with V5's canvas
    # Create a temporary config for preprocessing
    import dataclasses
    @dataclasses.dataclass
    class PCfg:
        canvas_width=cfg.canvas_width
        canvas_height=cfg.canvas_height
        input_channels=3
        aug_rotation_deg=0
        aug_translation_px=0
        aug_scale_min=1.0
        aug_scale_max=1.0
        aug_blur_prob=0.0
    # Use the model's preprocess via ai.preprocessing
    # For simplicity, call preprocess with a dummy ExperimentConfig
    from ai.config import ExperimentConfig
    pcfg = ExperimentConfig(canvas_width=cfg.canvas_width, canvas_height=cfg.canvas_height)
    tensor = preprocess(path, pcfg, augment=False)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode(tensor).cpu().numpy()[0]
    return emb

def encode_ssbi_crop(sheet_path: Path, bbox, cfg, device, model):
    import cv2
    from ai.preprocessing import binarize_ink, crop_to_ink, fit_to_canvas
    img = cv2.imread(str(sheet_path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(sheet_path)
    x,y,w,h = [int(round(v)) for v in bbox]
    crop = img[max(0,y):y+h, max(0,x):x+w]
    blurred = cv2.GaussianBlur(crop, (3,3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ys, xs = np.where(binary>0)
    if ys.size>0:
        y0,y1 = int(ys.min()), int(ys.max())+1
        x0,x1 = int(xs.min()), int(xs.max())+1
        ink = binary[y0:y1, x0:x1]
    else:
        ink = binary
    canvas = fit_to_canvas(ink, cfg.canvas_width, cfg.canvas_height)
    # to tensor
    canvas = np.clip(canvas,0,1)
    gray_t = torch.from_numpy(canvas).float().unsqueeze(0)
    rgb = gray_t.repeat(3,1,1)
    mean = torch.tensor((0.485,0.456,0.406)).view(-1,1,1)
    std = torch.tensor((0.229,0.224,0.225)).view(-1,1,1)
    tensor = ((rgb - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode(tensor).cpu().numpy()[0]
    return emb

def evaluate_for_k(model, cfg, device, writers, k, is_ssbi, gen_by_person, forg_by_person, cedar_index):
    # For each writer, enrollment = first k genuine, queries = remaining genuine + all forgeries + random impostor
    # Compute mean raw cosine for each query vs k refs
    genuine_scores=[]
    forgery_scores=[]
    random_scores=[]
    for w in writers:
        if not is_ssbi:
            # CEDAR
            idx = cedar_index[w]
            genuines = sorted(idx.originals, key=lambda p: int(p.stem.split("_")[-1]))
            forgeries = sorted(idx.forgeries, key=lambda p: int(p.stem.split("_")[-1]))
            if len(genuines) < k+1: continue
            refs = genuines[:k]
            queries_gen = genuines[k:]
            # Encode refs
            ref_embs = [encode_cedar_image(p, cfg, device, model) for p in refs]
            ref_embs = np.stack(ref_embs)
            # Queries genuine
            for q in queries_gen:
                q_emb = encode_cedar_image(q, cfg, device, model)
                sims = ref_embs @ q_emb  # cosine since L2
                mean = float(np.mean(sims))
                genuine_scores.append(mean)
            # Queries forgery
            for q in forgeries:
                q_emb = encode_cedar_image(q, cfg, device, model)
                sims = ref_embs @ q_emb
                mean = float(np.mean(sims))
                forgery_scores.append(mean)
            # Random impostor: genuine of other writers in same split
            other_writers = [ow for ow in writers if ow != w]
            if other_writers:
                other = random.choice(other_writers)
                other_idx = cedar_index[other]
                other_gen = random.choice(other_idx.originals)
                q_emb = encode_cedar_image(other_gen, cfg, device, model)
                sims = ref_embs @ q_emb
                mean = float(np.mean(sims))
                random_scores.append(mean)
        else:
            # SSBI
            gen_crops = gen_by_person.get(w, [])
            forg_crops = forg_by_person.get(w, [])
            if len(gen_crops) < k+1: continue
            refs = gen_crops[:k]
            queries_gen = gen_crops[k:]
            # Encode refs
            ref_embs=[]
            for sheet, bbox in refs:
                emb = encode_ssbi_crop(sheet, bbox, cfg, device, model)
                ref_embs.append(emb)
            ref_embs = np.stack(ref_embs)
            for sheet, bbox in queries_gen:
                q_emb = encode_ssbi_crop(sheet, bbox, cfg, device, model)
                sims = ref_embs @ q_emb
                mean = float(np.mean(sims))
                genuine_scores.append(mean)
            for sheet, bbox in forg_crops:
                q_emb = encode_ssbi_crop(sheet, bbox, cfg, device, model)
                sims = ref_embs @ q_emb
                mean = float(np.mean(sims))
                forgery_scores.append(mean)
            # Random impostor: other SSBI writer in same split
            other_writers = [ow for ow in writers if ow != w and len(gen_by_person.get(ow, []))>0]
            if other_writers:
                other = random.choice(other_writers)
                sheet, bbox = random.choice(gen_by_person[other])
                q_emb = encode_ssbi_crop(sheet, bbox, cfg, device, model)
                sims = ref_embs @ q_emb
                mean = float(np.mean(sims))
                random_scores.append(mean)
    # Compute metrics
    def compute_metrics(genuine, forged):
        scores = genuine + forged
        labels = [1]*len(genuine) + [0]*len(forged)
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, scores)
        except:
            pairs = sum(1 for g in genuine for f in forged if g>f) + 0.5*sum(1 for g in genuine for f in forged if g==f)
            auc = pairs / (len(genuine)*len(forged)) if genuine and forged else 0.5
        # EER via threshold sweep
        thresholds = sorted(set(scores))
        best_eer=1; best_thr=0; best_far=0; best_frr=0
        for thr in thresholds:
            far = sum(1 for f in forged if f>=thr)/len(forged) if forged else 0
            frr = sum(1 for g in genuine if g<thr)/len(genuine) if genuine else 0
            eer = (far+frr)/2
            if abs(far-frr) < 0.1 and eer < best_eer:
                best_eer=eer; best_thr=thr; best_far=far; best_frr=frr
        if best_eer==1:
            # fallback
            best_eer=0.5; best_thr=0
        return {"roc_auc": float(auc), "eer": float(best_eer), "eer_threshold": float(best_thr), "far_at_eer": float(best_far), "frr_at_eer": float(best_frr)}

    metrics = compute_metrics(genuine_scores, forgery_scores)
    # Random impostor AUC
    if random_scores:
        scores2 = genuine_scores + random_scores
        labels2 = [1]*len(genuine_scores) + [0]*len(random_scores)
        try:
            from sklearn.metrics import roc_auc_score
            random_auc = roc_auc_score(labels2, scores2)
        except:
            pairs = sum(1 for g in genuine_scores for f in random_scores if g>f) + 0.5*sum(1 for g in genuine_scores for f in random_scores if g==f)
            random_auc = pairs / (len(genuine_scores)*len(random_scores))
    else:
        random_auc=0.5

    return {
        "genuine_scores": genuine_scores,
        "forgery_scores": forgery_scores,
        "random_scores": random_scores,
        "genuine_mean": float(np.mean(genuine_scores)) if genuine_scores else 0,
        "genuine_median": float(np.median(genuine_scores)) if genuine_scores else 0,
        "genuine_std": float(np.std(genuine_scores)) if genuine_scores else 0,
        "forgery_mean": float(np.mean(forgery_scores)) if forgery_scores else 0,
        "forgery_median": float(np.median(forgery_scores)) if forgery_scores else 0,
        "forgery_std": float(np.std(forgery_scores)) if forgery_scores else 0,
        "random_mean": float(np.mean(random_scores)) if random_scores else 0,
        "random_median": float(np.median(random_scores)) if random_scores else 0,
        "random_std": float(np.std(random_scores)) if random_scores else 0,
        "random_auc": float(random_auc),
        **metrics
    }

def main():
    import hashlib, json
    cfg=V5Config()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Evaluating V5-A {cfg.checkpoint_path} vs V2")
    v2_sha = hashlib.sha256(Path(cfg.v2_checkpoint).read_bytes()).hexdigest() if Path(cfg.v2_checkpoint).exists() else "missing"
    v5_sha = hashlib.sha256(Path(cfg.checkpoint_path).read_bytes()).hexdigest() if Path(cfg.checkpoint_path).exists() else "missing"
    print(f"V2 SHA {v2_sha[:16]}... V5 SHA {v5_sha[:16]}...")
    # Load indices
    from ai.dataset import load_writer_index
    cedar_index = load_writer_index(cfg.cedar_root)
    # SSBI
    import json
    from collections import defaultdict
    gen_by_person, forg_by_person = {}, {}
    # Use same loader as train.py
    from ai.v5.train import load_ssbi_crops
    gen_by_person, forg_by_person = load_ssbi_crops(cfg.ssbi_root)
    # Evaluate for each K
    results={}
    for k in cfg.eval_ks:
        print(f"\nK={k} CEDAR TEST")
        res = evaluate_for_k(load_model(cfg, device), cfg, device, cfg.cedar_test, k, is_ssbi=False, gen_by_person=gen_by_person, forg_by_person=forg_by_person, cedar_index=cedar_index)
        print(f"  AUC {res['roc_auc']:.4f} EER {res['eer']:.4f} genuine {res['genuine_mean']:.3f} forgery {res['forgery_mean']:.3f} random AUC {res['random_auc']:.3f}")
        results[f"cedar_{k}"]=res
        print(f"K={k} SSBI TEST")
        res2 = evaluate_for_k(load_model(cfg, device), cfg, device, cfg.ssbi_test, k, is_ssbi=True, gen_by_person=gen_by_person, forg_by_person=forg_by_person, cedar_index=cedar_index)
        print(f"  AUC {res2['roc_auc']:.4f} EER {res2['eer']:.4f} genuine {res2['genuine_mean']:.3f} forgery {res2['forgery_mean']:.3f} random AUC {res2['random_auc']:.3f}")
        results[f"ssbi_{k}"]=res2
    # Locked signer 7
    print("\nLocked SSBI signer 7 (10+10, K5 mean)")
    # Use the controlled pack's extraction and K5 mean via V5 model
    # For this real evaluation, we will use the actual SSBI 7 crops from source
    # The controlled pack's REF_S7_01..05 are the first 5 genuine of person 7, and the 10+10 are from the same person
    # We will evaluate with K5 mean using V5
    model_v5 = load_model(cfg, device)
    # Get SSBI 7 genuine crops
    gen_7 = gen_by_person[7]
    forg_7 = forg_by_person[7]
    # Use first 5 as refs, remaining 11 genuines (16-5=11) but we only have 10 in controlled pack, so use first 5 refs and next 10 as queries to match controlled
    refs = gen_7[:5]
    queries_gen = gen_7[5:15]  # 10
    queries_forg = forg_7[:8]  # 8, but controlled has 10 with 2 reused, so use 8 + 2 duplicates
    # For 10 forged, duplicate 2
    if len(queries_forg)==8:
        queries_forg = queries_forg + queries_forg[:2]
    ref_embs = []
    for sheet, bbox in refs:
        emb = encode_ssbi_crop(sheet, bbox, cfg, device, model_v5)
        ref_embs.append(emb)
    ref_embs = np.stack(ref_embs)
    v5_genuine=[]
    for sheet, bbox in queries_gen[:10]:
        q_emb = encode_ssbi_crop(sheet, bbox, cfg, device, model_v5)
        sims = ref_embs @ q_emb
        v5_genuine.append(float(np.mean(sims)))
    v5_forged=[]
    for sheet, bbox in queries_forg[:10]:
        q_emb = encode_ssbi_crop(sheet, bbox, cfg, device, model_v5)
        sims = ref_embs @ q_emb
        v5_forged.append(float(np.mean(sims)))
    v5_genuine_mean = float(np.mean(v5_genuine))
    v5_forged_mean = float(np.mean(v5_forged))
    pairs = sum(1 for g in v5_genuine for f in v5_forged if g>f)
    auc = pairs / 100
    print(f"V5 genuine {v5_genuine} mean {v5_genuine_mean:.4f}")
    print(f"V5 forged {v5_forged} mean {v5_forged_mean:.4f} AUC {auc:.4f} G>F {pairs}/100")
    print(f"V2 genuine mean 0.1645 forged 0.2277 AUC 0.09 inverted: V5 corrected {v5_genuine_mean > v5_forged_mean}")
    # Save artifacts
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    real_dir = cfg.reports_dir / "real_run"
    real_dir.mkdir(parents=True, exist_ok=True)
    # Also write to main reports_dir for compatibility
    for d in [cfg.reports_dir, real_dir]:
        with open(d / "test_cedar_scores.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["K","genuine_score","forgery_score","random_score"])
            for k in cfg.eval_ks:
                r=results[f"cedar_{k}"]
                for g,fo,ra in zip(r["genuine_scores"], r["forgery_scores"], r["random_scores"]):
                    w.writerow([k, f"{g:.6f}", f"{fo:.6f}", f"{ra:.6f}"])
        with open(d / "test_ssbi_scores.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["K","genuine_score","forgery_score","random_score"])
            for k in cfg.eval_ks:
                r=results[f"ssbi_{k}"]
                for g,fo,ra in zip(r["genuine_scores"], r["forgery_scores"], r["random_scores"]):
                    w.writerow([k, f"{g:.6f}", f"{fo:.6f}", f"{ra:.6f}"])
        with open(d / "locked_ssbi7_scores.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["sample","ground_truth","score"])
            for i,g in enumerate(v5_genuine):
                w.writerow([f"S7_GENUINE_{i+1:02d}.png","GENUINE", f"{g:.6f}"])
            for i,f_ in enumerate(v5_forged):
                w.writerow([f"S7_FORGED_{i+1:02d}.png","FORGED", f"{f_:.6f}"])
        # v2 vs v5
        # Load V2 metrics for same protocol (we need to evaluate V2 similarly)
        model_v2, _ = load_v2_model(device)
        v2_results={}
        for k in cfg.eval_ks:
            r = evaluate_for_k(model_v2, cfg, device, cfg.cedar_test, k, is_ssbi=False, gen_by_person=gen_by_person, forg_by_person=forg_by_person, cedar_index=cedar_index)
            v2_results[f"cedar_{k}"]=r
            r2 = evaluate_for_k(model_v2, cfg, device, cfg.ssbi_test, k, is_ssbi=True, gen_by_person=gen_by_person, forg_by_person=forg_by_person, cedar_index=cedar_index)
            v2_results[f"ssbi_{k}"]=r2
        with open(d / "v2_vs_v5_comparison.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["domain","K","metric","V2","V5-A","delta"])
            for k in cfg.eval_ks:
                w.writerow(["CEDAR",k,"skilled_AUC", f"{v2_results[f'cedar_{k}']['roc_auc']:.4f}", f"{results[f'cedar_{k}']['roc_auc']:.4f}", f"{results[f'cedar_{k}']['roc_auc']-v2_results[f'cedar_{k}']['roc_auc']:.4f}"])
                w.writerow(["CEDAR",k,"EER", f"{v2_results[f'cedar_{k}']['eer']:.4f}", f"{results[f'cedar_{k}']['eer']:.4f}", f"{v2_results[f'cedar_{k}']['eer']-results[f'cedar_{k}']['eer']:.4f}"])
                w.writerow(["SSBI",k,"skilled_AUC", f"{v2_results[f'ssbi_{k}']['roc_auc']:.4f}", f"{results[f'ssbi_{k}']['roc_auc']:.4f}", f"{results[f'ssbi_{k}']['roc_auc']-v2_results[f'ssbi_{k}']['roc_auc']:.4f}"])
                w.writerow(["SSBI",k,"EER", f"{v2_results[f'ssbi_{k}']['eer']:.4f}", f"{results[f'ssbi_{k}']['eer']:.4f}", f"{v2_results[f'ssbi_{k}']['eer']-results[f'ssbi_{k}']['eer']:.4f}"])
        # ROC plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for domain, is_ssbi in [("cedar", False), ("ssbi", True)]:
            plt.figure()
            for k in cfg.eval_ks:
                r=results[f"{domain}_{k}"]
                genuine=r["genuine_scores"]; forged=r["forgery_scores"]
                scores=genuine+forged; labels=[1]*len(genuine)+[0]*len(forged)
                pairs=sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
                P=len(genuine); N=len(forged); TP=0; FP=0; fpr=[0]; tpr=[0]
                for s,l in pairs:
                    if l==1: TP+=1
                    else: FP+=1
                    fpr.append(FP/N); tpr.append(TP/P)
                fpr.append(1); tpr.append(1)
                auc_val=r["roc_auc"]
                plt.plot(fpr,tpr, label=f"K{k} AUC={auc_val:.3f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"V5-A {domain} ROC")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(d / f"roc_{domain}.png", dpi=150); plt.close()
        # locked roc
        plt.figure()
        scores=v5_genuine+v5_forged; labels=[1]*10+[0]*10
        pairs=sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        P=10;N=10;TP=0;FP=0;fpr=[0];tpr=[0]
        for s,l in pairs:
            if l==1: TP+=1
            else: FP+=1
            fpr.append(FP/N); tpr.append(TP/P)
        fpr.append(1); tpr.append(1)
        plt.plot(fpr,tpr, label=f"Locked AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("V5-A locked signer7 ROC")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(d / "roc_locked_ssbi7.png", dpi=150); plt.close()
        # Final report
        with open(d / "V5A_FINAL_REPORT.md","w") as f:
            f.write(f"# V5-A Final Report (REAL)\n\n")
            f.write(f"V2 SHA {v2_sha[:16]} V5 SHA {hashlib.sha256(Path(cfg.checkpoint_path).read_bytes()).hexdigest()[:16] if Path(cfg.checkpoint_path).exists() else 'missing'}\n\n")
            for k in cfg.eval_ks:
                f.write(f"## K{k} CEDAR AUC {results[f'cedar_{k}']['roc_auc']:.4f} EER {results[f'cedar_{k}']['eer']:.4f} SSBI AUC {results[f'ssbi_{k}']['roc_auc']:.4f}\n")
            f.write(f"\nLocked signer7 V5 genuine mean {v5_genuine_mean:.4f} forged {v5_forged_mean:.4f} AUC {auc:.4f} G>F {pairs}/100 vs V2 0.09 inverted corrected {v5_genuine_mean>v5_forged_mean}\n")
    print(f"Artifacts written to {cfg.reports_dir} and {real_dir}")

if __name__=="__main__":
    main()
