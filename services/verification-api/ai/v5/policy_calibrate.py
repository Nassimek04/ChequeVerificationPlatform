"""Phase 8 decision-policy calibration — VAL ONLY, frozen V5 checkpoint."""
from __future__ import annotations
import hashlib, json, csv, time, pathlib, random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import V5Config
from .train import load_ssbi_crops, compute_real_validation_k5
from ai.dataset import load_writer_index
from ai.model import SiameseResNet18
import dataclasses

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_model(cfg: V5Config, device):
    @dataclasses.dataclass
    class Cfg2:
        embedding_dim=cfg.embedding_dim
        pretrained=False
        canvas_width=cfg.canvas_width
        canvas_height=cfg.canvas_height
    m=SiameseResNet18(Cfg2())
    ckpt=torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(device); m.eval()
    return m, ckpt

def policy_metrics(genuine, skilled, L, U):
    # L < U required
    assert L < U
    g_con = sum(1 for s in genuine if s >= U)
    g_man = sum(1 for s in genuine if L < s < U)
    g_non = sum(1 for s in genuine if s <= L)
    s_con = sum(1 for s in skilled if s >= U)  # false accept
    s_man = sum(1 for s in skilled if L < s < U)
    s_non = sum(1 for s in skilled if s <= L)
    total = len(genuine)+len(skilled)
    auto = g_con+g_non+s_con+s_non
    manual = g_man+s_man
    coverage = auto/total if total else 0
    manual_rate = manual/total if total else 0
    auto_far = s_con/len(skilled) if skilled else 0
    auto_frr = g_non/len(genuine) if genuine else 0
    auto_correct = g_con+s_non
    auto_acc = auto_correct/auto if auto else 0
    # also compute auto FAR over auto-forgeries? but we report per total
    return {
        "L": L, "U": U,
        "g_con": g_con, "g_man": g_man, "g_non": g_non,
        "s_con": s_con, "s_man": s_man, "s_non": s_non,
        "total": total, "auto": auto, "manual": manual,
        "coverage": coverage, "manual_rate": manual_rate,
        "auto_far": auto_far, "auto_frr": auto_frr,
        "auto_acc": auto_acc, "auto_correct": auto_correct,
    }

def search_frontier(genuine, skilled):
    # candidate thresholds from unique scores
    scores = sorted(set(genuine+skilled))
    # add epsilon outside
    candidates=[]
    # exhaustive L<U pairs from scores
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            L=scores[i]; U=scores[j]
            if L>=U: continue
            m=policy_metrics(genuine, skilled, L, U)
            candidates.append(m)
    # also consider L below min and U above max for completeness
    # but we have enough candidates
    return candidates

def main():
    cfg=V5Config()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    v2_path = cfg.v2_checkpoint
    ckpt_path = cfg.checkpoint_path
    sha_before = sha256(ckpt_path)
    print(f"Checkpoint SHA before {sha_before}")
    assert sha_before == "5593242d5e846bef481e9e07f0217c99abf8365d251ff645b910a7e4355ac0a2", "checkpoint changed before calibration!"
    model, ckpt = load_model(cfg, device)
    cedar_index = load_writer_index(cfg.cedar_root)
    gen_by_person, forg_by_person = load_ssbi_crops(cfg.ssbi_root)

    # === VAL scores ===
    val_res = compute_real_validation_k5(model, cfg, device, cedar_index, gen_by_person, forg_by_person)
    genuine = val_res["genuine_scores"]
    skilled = val_res["forgery_scores"]
    random_scores = val_res["random_scores"]
    print(f"VAL genuine {len(genuine)} skilled {len(skilled)} random {len(random_scores)}")
    print(f"VAL genuine mean {np.mean(genuine):.4f} forg mean {np.mean(skilled):.4f}")
    # save validation_scores.csv
    out_dir = cfg.reports_dir / "phase8_policy"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"validation_scores.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["score","label","type"])  # label 1 genuine 0 skilled
        for s in genuine: w.writerow([f"{s:.6f}",1,"genuine"])
        for s in skilled: w.writerow([f"{s:.6f}",0,"skilled"])
        for s in random_scores: w.writerow([f"{s:.6f}",0,"random"])
    # === search frontier ===
    cands = search_frontier(genuine, skilled)
    print(f"candidates {len(cands)}")
    # Filter FAR=0
    far0 = [c for c in cands if c["auto_far"]==0]
    print(f"FAR0 candidates {len(far0)}")
    far0_sorted = sorted(far0, key=lambda c: (c["auto_frr"], -c["coverage"]))
    # For meaningful frontier, consider only low-FRR candidates (FRR<0.2) to keep genuine rejects low
    # This is the safety-first but usable region
    far0_low = [c for c in far0 if c["auto_frr"] <= 0.20 and 0.2 <= c["coverage"] <= 0.9]
    if len(far0_low) < 3:
        far0_low = [c for c in far0 if c["auto_frr"] <= 0.30 and 0.1 <= c["coverage"] <= 0.95]
    if len(far0_low) < 3:
        far0_low = far0  # fallback
    # Sort low group by coverage for frontier
    low_sorted_cov = sorted(far0_low, key=lambda c: c["coverage"])
    # Frontier points: very (lowest coverage among low), conservative (median), balanced (highest coverage)
    very = low_sorted_cov[0] if low_sorted_cov else None
    balanced = low_sorted_cov[-1] if low_sorted_cov else None
    conservative = low_sorted_cov[len(low_sorted_cov)//2] if low_sorted_cov else None
    frontier=[]
    for name, cand in [("VERY CONSERVATIVE", very), ("CONSERVATIVE", conservative), ("BALANCED", balanced)]:
        if cand:
            frontier.append((name, cand))
    # Deduplicate if same
    uniq={}
    for n,c in frontier:
        key=(c["L"],c["U"])
        uniq[key]=(n,c)  # keep last
    frontier=list(uniq.values())
    # Ensure sorted order VERY->CONS->BALANCED by coverage
    frontier=sorted(frontier, key=lambda x: x[1]["coverage"])
    # Re-label after dedup to keep 3 names ordered
    labels=["VERY CONSERVATIVE","CONSERVATIVE","BALANCED"]
    frontier=[(labels[i], c) for i,(n,c) in enumerate(frontier[:3])]
    print("Frontier:")
    for name,c in frontier:
        print(f"{name}: L={c['L']:.4f} U={c['U']:.4f} cov={c['coverage']:.3f} man={c['manual_rate']:.3f} FAR={c['auto_far']:.4f} FRR={c['auto_frr']:.4f} acc={c['auto_acc']:.3f} g {c['g_con']}/{c['g_man']}/{c['g_non']} s {c['s_con']}/{c['s_man']}/{c['s_non']}")

    # Selection rule: skilled FAR=0, lowest FRR, highest coverage
    if far0_sorted:
        primary = far0_sorted[0]  # lowest FRR, highest coverage tie-break already sorted
        reason = "Primary safety constraint: skilled FAR=0 on VAL; lowest genuine FRR; highest coverage among ties"
    else:
        all_sorted = sorted(cands, key=lambda c: (c["auto_far"], c["auto_frr"], -c["coverage"]))
        primary = all_sorted[0]
        reason = "No FAR=0 candidate; fallback to lowest FAR then FRR"

    L = float(primary["L"]); U = float(primary["U"])
    print(f"Selected L={L:.6f} U={U:.6f} reason {reason}")

    # Save policy.json
    policy={
        "model_sha": sha_before,
        "model_version": cfg.model_version,
        "K": 5,
        "aggregation": "mean raw cosine",
        "L": L,
        "U": U,
        "calibration_identities": {"cedar_val": list(cfg.cedar_val), "ssbi_val": list(cfg.ssbi_val), "ssbi_skilled_val": [1,8]},
        "calibration_counts": {"genuine": len(genuine), "skilled": len(skilled), "random": len(random_scores)},
        "selection_rule": reason,
        "primary_safety_constraint": "skilled FAR=0 on VAL, then lowest FRR, then highest coverage",
        "frontier": [{"name": n, "L": float(c["L"]), "U": float(c["U"]), "coverage": float(c["coverage"]), "manual_rate": float(c["manual_rate"]), "auto_far": float(c["auto_far"]), "auto_frr": float(c["auto_frr"]), "auto_acc": float(c["auto_acc"]), "g_con": int(c["g_con"]), "g_man": int(c["g_man"]), "g_non": int(c["g_non"]), "s_con": int(c["s_con"]), "s_man": int(c["s_man"]), "s_non": int(c["s_non"])} for n,c in frontier],
        "validation_metrics_at_primary": {"coverage": float(primary["coverage"]), "manual_rate": float(primary["manual_rate"]), "auto_far": float(primary["auto_far"]), "auto_frr": float(primary["auto_frr"]), "auto_acc": float(primary["auto_acc"]), "g_con": int(primary["g_con"]), "g_man": int(primary["g_man"]), "g_non": int(primary["g_non"]), "s_con": int(primary["s_con"]), "s_man": int(primary["s_man"]), "s_non": int(primary["s_non"])},
        "threshold_calibration": "VAL ONLY, deterministic K5, no TEST, no signer7",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg.to_dict(),
        "aggregation_details": "raw cosine per reference, mean over K=5",
        "real_validation": True,
        "frozen": True,
    }
    with open(out_dir/"policy.json","w") as f: json.dump(policy, f, indent=2)
    # policy_frontier.csv
    with open(out_dir/"policy_frontier.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["name","L","U","coverage","manual_rate","auto_far","auto_frr","auto_acc","g_con","g_man","g_non","s_con","s_man","s_non"])
        for name,c in frontier:
            w.writerow([name, f"{c['L']:.6f}", f"{c['U']:.6f}", f"{c['coverage']:.4f}", f"{c['manual_rate']:.4f}", f"{c['auto_far']:.4f}", f"{c['auto_frr']:.4f}", f"{c['auto_acc']:.4f}", c["g_con"], c["g_man"], c["g_non"], c["s_con"], c["s_man"], c["s_non"]])
        # also primary
        w.writerow(["PRIMARY", f"{L:.6f}", f"{U:.6f}", f"{primary['coverage']:.4f}", f"{primary['manual_rate']:.4f}", f"{primary['auto_far']:.4f}", f"{primary['auto_frr']:.4f}", f"{primary['auto_acc']:.4f}", primary["g_con"], primary["g_man"], primary["g_non"], primary["s_con"], primary["s_man"], primary["s_non"]])

    # === Evaluate on benchmarks with frozen L/U ===
    # Helper to compute policy results for a given test set
    def evaluate_test(writers, is_ssbi, label):
        from ai.v5.train import _encode_cedar_noaug, _encode_ssbi_noaug, _batch_encode
        import re
        genuine_scores=[]; skilled_scores=[]; random_scores=[]
        # For per-sample csv we need individual scores with decisions
        per_sample=[]  # (writer, score, type, decision)
        # We'll compute same as validation but for TEST
        for w in writers:
            if not is_ssbi:
                idx=cedar_index[w]
                def ckey(p):
                    import re
                    m=re.search(r"_(\d+)\.png$", p.name)
                    return int(m.group(1)) if m else p.name
                genuines=sorted(idx.originals, key=ckey)
                forgeries=sorted(idx.forgeries, key=ckey)
                if len(genuines)<6: continue
                refs=genuines[:5]
                queries_gen=genuines[5:]
                ref_tensors=[_encode_cedar_noaug(p,cfg) for p in refs]
                ref_embs=_batch_encode(ref_tensors, model, device)
                for q in queries_gen:
                    q_t=_encode_cedar_noaug(q,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    genuine_scores.append(score)
                    decision="CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"
                    per_sample.append((w, q.name, score, "genuine", decision))
                for q in forgeries:
                    q_t=_encode_cedar_noaug(q,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    skilled_scores.append(score)
                    decision="CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"
                    per_sample.append((w, q.name, score, "skilled", decision))
                # random impostor: one other writer's genuine
                other=[ow for ow in writers if ow!=w]
                if other:
                    ow=sorted(other)[0]
                    other_gen=sorted(cedar_index[ow].originals, key=ckey)[0]
                    q_t=_encode_cedar_noaug(other_gen,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    random_scores.append(score)
                    per_sample.append((w, f"random_from_{ow}", score, "random", "CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"))
            else:
                gen_crops=gen_by_person.get(w, [])
                forg_crops=forg_by_person.get(w, [])
                if len(gen_crops)<6: continue
                gen_sorted=sorted(gen_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                forg_sorted=sorted(forg_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                refs=gen_sorted[:5]
                queries_gen=gen_sorted[5:]
                ref_tensors=[_encode_ssbi_noaug(s,b,cfg) for s,b in refs]
                ref_embs=_batch_encode(ref_tensors, model, device)
                for s,b in queries_gen:
                    q_t=_encode_ssbi_noaug(s,b,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    genuine_scores.append(score)
                    per_sample.append((w, str(s.name), score, "genuine", "CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"))
                for s,b in forg_sorted:
                    q_t=_encode_ssbi_noaug(s,b,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    skilled_scores.append(score)
                    per_sample.append((w, str(s.name), score, "skilled", "CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"))
                other=[ow for ow in writers if ow!=w and len(gen_by_person.get(ow,[]))>0]
                if other:
                    ow=sorted(other)[0]
                    other_sorted=sorted(gen_by_person[ow], key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                    s,b=other_sorted[0]
                    q_t=_encode_ssbi_noaug(s,b,cfg)
                    q_emb=_batch_encode([q_t], model, device)[0]
                    score=float(np.mean(ref_embs @ q_emb))
                    random_scores.append(score)
                    per_sample.append((w, f"random_from_{ow}", score, "random", "CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"))
        # compute metrics via policy_metrics
        pm=policy_metrics(genuine_scores, skilled_scores, L, U)
        # add random breakdown
        r_con=sum(1 for s in random_scores if s>=U)
        r_man=sum(1 for s in random_scores if L < s < U)
        r_non=sum(1 for s in random_scores if s<=L)
        return pm, genuine_scores, skilled_scores, random_scores, per_sample, (r_con,r_man,r_non)

    # CEDAR TEST
    cedar_pm, cedar_gen, cedar_sk, cedar_rand, cedar_samples, cedar_r = evaluate_test(cfg.cedar_test, False, "cedar")
    with open(out_dir/"cedar_policy_results.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","sample","score","type","decision"])
        for row in cedar_samples: w.writerow([row[0], row[1], f"{row[2]:.6f}", row[3], row[4]])
        w.writerow([])
        w.writerow(["summary","genuine_con","genuine_man","genuine_non","skilled_con","skilled_man","skilled_non","coverage","manual_rate","auto_far","auto_frr","auto_acc"])
        w.writerow(["CEDAR", cedar_pm["g_con"], cedar_pm["g_man"], cedar_pm["g_non"], cedar_pm["s_con"], cedar_pm["s_man"], cedar_pm["s_non"], f"{cedar_pm['coverage']:.4f}", f"{cedar_pm['manual_rate']:.4f}", f"{cedar_pm['auto_far']:.4f}", f"{cedar_pm['auto_frr']:.4f}", f"{cedar_pm['auto_acc']:.4f}"])

    # SSBI TEST (for skilled, use 3,14 only but we evaluate 0,3,14 total; genuine includes 0, skilled only 3,14)
    # For accurate per spec, we should run for all 0,3,14 but count skilled only 3,14
    # Our evaluate_test for SSBI will include 0's genuine but 0 has no forged, so skilled only from 3,14
    ssbi_pm, ssbi_gen, ssbi_sk, ssbi_rand, ssbi_samples, ssbi_r = evaluate_test(cfg.ssbi_test, True, "ssbi")
    # Need to separate counts for genuine (includes 0) vs skilled (3,14)
    with open(out_dir/"ssbi_policy_results.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","sample","score","type","decision"])
        for row in ssbi_samples: w.writerow([row[0], row[1], f"{row[2]:.6f}", row[3], row[4]])
        w.writerow([])
        w.writerow(["summary","genuine_con","genuine_man","genuine_non","skilled_con","skilled_man","skilled_non","coverage","manual_rate","auto_far","auto_frr","auto_acc"])
        w.writerow(["SSBI", ssbi_pm["g_con"], ssbi_pm["g_man"], ssbi_pm["g_non"], ssbi_pm["s_con"], ssbi_pm["s_man"], ssbi_pm["s_non"], f"{ssbi_pm['coverage']:.4f}", f"{ssbi_pm['manual_rate']:.4f}", f"{ssbi_pm['auto_far']:.4f}", f"{ssbi_pm['auto_frr']:.4f}", f"{ssbi_pm['auto_acc']:.4f}"])

    # SIGNER 7
    # Use same SSBI 7 crops, K5 refs, 10 genuine +10 forged as in evaluate.py
    gen7=gen_by_person[7]
    forg7=forg_by_person[7]
    gen_sorted=sorted(gen7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    forg_sorted=sorted(forg7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    refs=gen_sorted[:5]
    queries_gen=gen_sorted[5:15]
    queries_forg=forg_sorted[:8]+forg_sorted[:2]  # 10
    from ai.v5.train import _encode_ssbi_noaug, _batch_encode
    ref_tensors=[_encode_ssbi_noaug(s,b,cfg) for s,b in refs]
    ref_embs=_batch_encode(ref_tensors, model, device)
    s7_gen_scores=[]
    s7_gen_dec=[]
    for s,b in queries_gen[:10]:
        q_t=_encode_ssbi_noaug(s,b,cfg)
        q_emb=_batch_encode([q_t], model, device)[0]
        score=float(np.mean(ref_embs @ q_emb))
        s7_gen_scores.append(score)
        dec="CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"
        s7_gen_dec.append(dec)
    s7_forg_scores=[]
    s7_forg_dec=[]
    for s,b in queries_forg[:10]:
        q_t=_encode_ssbi_noaug(s,b,cfg)
        q_emb=_batch_encode([q_t], model, device)[0]
        score=float(np.mean(ref_embs @ q_emb))
        s7_forg_scores.append(score)
        dec="CONFORME" if score>=U else "NON CONFORME" if score<=L else "MANUAL"
        s7_forg_dec.append(dec)
    s7_pm=policy_metrics(s7_gen_scores, s7_forg_scores, L, U)
    with open(out_dir/"signer7_policy_results.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["sample","score","type","decision"])
        for i,(s,d) in enumerate(zip(s7_gen_scores, s7_gen_dec)):
            w.writerow([f"S7_GENUINE_{i+1:02d}", f"{s:.6f}", "genuine", d])
        for i,(s,d) in enumerate(zip(s7_forg_scores, s7_forg_dec)):
            w.writerow([f"S7_FORGED_{i+1:02d}", f"{s:.6f}", "forged", d])
        w.writerow([])
        w.writerow(["summary","genuine_con","genuine_man","genuine_non","forged_con","forged_man","forged_non","coverage","auto_far","auto_frr"])
        w.writerow(["SIGNER7", s7_pm["g_con"], s7_pm["g_man"], s7_pm["g_non"], s7_pm["s_con"], s7_pm["s_man"], s7_pm["s_non"], f"{s7_pm['coverage']:.4f}", f"{s7_pm['auto_far']:.4f}", f"{s7_pm['auto_frr']:.4f}"])

    # RANDOM IMPOSTOR separate (already have random scores from cedar/ssbi)
    # Combine cedar+ssbi random
    all_random = cedar_rand+ssbi_rand
    # For V5, random scores are from validation random and test random; we should compute test random
    # Use cedar_rand and ssbi_rand from above test evaluations
    random_all = cedar_rand+ssbi_rand  # wait cedar_rand is list of scores, not counts
    # Actually cedar_rand list is scores
    # Let's compute random policy breakdown for test sets combined
    combined_random_scores = cedar_rand + ssbi_rand
    r_con=sum(1 for s in combined_random_scores if s>=U)
    r_man=sum(1 for s in combined_random_scores if L < s < U)
    r_non=sum(1 for s in combined_random_scores if s<=L)
    with open(out_dir/"random_impostor_policy_results.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["score","decision"])
        for s in combined_random_scores:
            dec="CONFORME" if s>=U else "NON CONFORME" if s<=L else "MANUAL"
            w.writerow([f"{s:.6f}", dec])
        w.writerow([])
        w.writerow(["random_con","random_man","random_non","total"])
        w.writerow([r_con, r_man, r_non, len(combined_random_scores)])

    # V2 vs V5 policy comparison
    # V2 thresholds
    V2_L=0.0895; V2_U=0.6898
    # Need V2 model scoring on same test sets
    # Load V2
    from ai.v2.config import V2Config
    from ai.model import SiameseResNet18 as Siamese
    v2cfg=V2Config()
    @dataclasses.dataclass
    class V2C:
        embedding_dim=v2cfg.embedding_dim
        pretrained=False
        canvas_width=v2cfg.canvas_width
        canvas_height=v2cfg.canvas_height
    v2m=Siamese(V2C())
    ckpt2=torch.load(v2cfg.checkpoint_path, map_location="cpu", weights_only=False)
    v2m.load_state_dict(ckpt2["model_state"])
    v2m.to(device); v2m.eval()
    # Evaluate V2 using same K5 logic but with V2's canvas (256x128 same) and its own encode
    # We'll reuse same helper but with V2 model - need to handle canvas size same
    # For simplicity, use same _encode functions but with cfg (same canvas) - V2 canvas is same 256x128
    # So we can evaluate V2 similarly by swapping model
    def eval_v2(writers, is_ssbi):
        gen=[]; skil=[]; rand=[]
        for w in writers:
            if not is_ssbi:
                idx=cedar_index[w]
                def ckey(p):
                    import re
                    m=re.search(r"_(\d+)\.png$", p.name)
                    return int(m.group(1)) if m else p.name
                genuines=sorted(idx.originals, key=ckey)
                forgeries=sorted(idx.forgeries, key=ckey)
                if len(genuines)<6: continue
                refs=genuines[:5]
                queries_gen=genuines[5:]
                # encode with V2 model using same preprocess (no aug)
                from ai.preprocessing import read_gray, binarize_ink, crop_to_ink, fit_to_canvas
                def enc_cedar(p):
                    gray=read_gray(p); bin_img=binarize_ink(gray); ink=crop_to_ink(bin_img); canvas=fit_to_canvas(ink, v2cfg.canvas_width, v2cfg.canvas_height)
                    import numpy as np2
                    canvas=np2.clip(canvas,0,1)
                    t=torch.from_numpy(canvas).float().unsqueeze(0).repeat(3,1,1)
                    mean=torch.tensor((0.485,0.456,0.406)).view(-1,1,1)
                    std=torch.tensor((0.229,0.224,0.225)).view(-1,1,1)
                    tt=(t-mean)/std
                    return tt
                ref_tensors=[enc_cedar(p) for p in refs]
                # batch encode V2
                with torch.no_grad():
                    batch=torch.stack(ref_tensors).to(device)
                    ref_embs=v2m.encode(batch).cpu().numpy()
                for q in queries_gen:
                    qt=enc_cedar(q)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    gen.append(score)
                for q in forgeries:
                    qt=enc_cedar(q)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    skil.append(score)
                other=[ow for ow in writers if ow!=w]
                if other:
                    ow=sorted(other)[0]
                    other_gen=sorted(cedar_index[ow].originals, key=ckey)[0]
                    qt=enc_cedar(other_gen)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    rand.append(score)
            else:
                gen_crops=gen_by_person.get(w, [])
                forg_crops=forg_by_person.get(w, [])
                if len(gen_crops)<6: continue
                gen_sorted=sorted(gen_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                forg_sorted=sorted(forg_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                refs=gen_sorted[:5]
                # encode refs with V2 using same SSBI crop preprocessing
                def enc_ssbi(sheet,bbox):
                    import cv2
                    from ai.preprocessing import fit_to_canvas
                    img=cv2.imread(str(sheet), cv2.IMREAD_GRAYSCALE)
                    x,y,ww,hh=[int(round(v)) for v in bbox]
                    crop=img[max(0,y):y+hh, max(0,x):x+ww]
                    blurred=cv2.GaussianBlur(crop,(3,3),0)
                    _,binary=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    ys,xs=np.where(binary>0)
                    if ys.size>0:
                        y0,y1=int(ys.min()), int(ys.max())+1
                        x0,x1=int(xs.min()), int(xs.max())+1
                        ink=binary[y0:y1, x0:x1]
                    else:
                        ink=binary
                    canvas=fit_to_canvas(ink, v2cfg.canvas_width, v2cfg.canvas_height)
                    canvas=np.clip(canvas,0,1)
                    t=torch.from_numpy(canvas).float().unsqueeze(0).repeat(3,1,1)
                    mean=torch.tensor((0.485,0.456,0.406)).view(-1,1,1)
                    std=torch.tensor((0.229,0.224,0.225)).view(-1,1,1)
                    return (t-mean)/std
                ref_tensors=[enc_ssbi(s,b) for s,b in refs]
                with torch.no_grad():
                    batch=torch.stack(ref_tensors).to(device)
                    ref_embs=v2m.encode(batch).cpu().numpy()
                for s,b in gen_sorted[5:]:
                    qt=enc_ssbi(s,b)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    gen.append(score)
                for s,b in forg_sorted:
                    qt=enc_ssbi(s,b)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    skil.append(score)
                other=[ow for ow in writers if ow!=w and len(gen_by_person.get(ow,[]))>0]
                if other:
                    ow=sorted(other)[0]
                    other_sorted=sorted(gen_by_person[ow], key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
                    s,b=other_sorted[0]
                    qt=enc_ssbi(s,b)
                    with torch.no_grad():
                        qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
                    score=float(np.mean(ref_embs @ qe))
                    rand.append(score)
        return gen, skil, rand

    v2_cedar_gen,v2_cedar_sk,v2_cedar_rand=eval_v2(cfg.cedar_test, False)
    v2_ssbi_gen,v2_ssbi_sk,v2_ssbi_rand=eval_v2(cfg.ssbi_test, True)
    # V2 signer7 similarly (use V2 model)
    gen7=gen_by_person[7]
    forg7=forg_by_person[7]
    gen_sorted=sorted(gen7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    forg_sorted=sorted(forg7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    refs=gen_sorted[:5]
    from ai.preprocessing import fit_to_canvas as ftc
    def enc_ssbi_v2(sheet,bbox):
        import cv2
        img=cv2.imread(str(sheet), cv2.IMREAD_GRAYSCALE)
        x,y,ww,hh=[int(round(v)) for v in bbox]
        crop=img[max(0,y):y+hh, max(0,x):x+ww]
        blurred=cv2.GaussianBlur(crop,(3,3),0)
        _,binary=cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ys,xs=np.where(binary>0)
        if ys.size>0:
            y0,y1=int(ys.min()), int(ys.max())+1
            x0,x1=int(xs.min()), int(xs.max())+1
            ink=binary[y0:y1, x0:x1]
        else:
            ink=binary
        canvas=ftc(ink, v2cfg.canvas_width, v2cfg.canvas_height)
        canvas=np.clip(canvas,0,1)
        t=torch.from_numpy(canvas).float().unsqueeze(0).repeat(3,1,1)
        mean=torch.tensor((0.485,0.456,0.406)).view(-1,1,1)
        std=torch.tensor((0.229,0.224,0.225)).view(-1,1,1)
        return (t-mean)/std
    ref_tensors=[enc_ssbi_v2(s,b) for s,b in refs]
    with torch.no_grad():
        batch=torch.stack(ref_tensors).to(device)
        ref_embs=v2m.encode(batch).cpu().numpy()
    v2_s7_gen=[]
    for s,b in gen_sorted[5:15]:
        qt=enc_ssbi_v2(s,b)
        with torch.no_grad():
            qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
        v2_s7_gen.append(float(np.mean(ref_embs @ qe)))
    v2_s7_forg=[]
    for s,b in (forg_sorted[:8]+forg_sorted[:2]):
        qt=enc_ssbi_v2(s,b)
        with torch.no_grad():
            qe=v2m.encode(qt.unsqueeze(0).to(device)).cpu().numpy()[0]
        v2_s7_forg.append(float(np.mean(ref_embs @ qe)))

    # Compute V2 policy metrics
    def v2_pm(gen,sk, L=V2_L, U=V2_U): return policy_metrics(gen,sk,L,U)
    v2_cedar_pm=v2_pm(v2_cedar_gen, v2_cedar_sk)
    v2_ssbi_pm=v2_pm(v2_ssbi_gen, v2_ssbi_sk)
    v2_s7_pm=v2_pm(v2_s7_gen, v2_s7_forg)

    v5_cedar_pm=cedar_pm
    v5_ssbi_pm=ssbi_pm
    v5_s7_pm=s7_pm

    with open(out_dir/"v2_vs_v5_policy.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","V2","V5"])
        w.writerow(["model","V2 ResNet18 128-D","V5-A ResNet18 128-D"])
        w.writerow(["L", f"{V2_L:.4f}", f"{L:.4f}"])
        w.writerow(["U", f"{V2_U:.4f}", f"{U:.4f}"])
        w.writerow([])
        w.writerow(["CEDAR auto coverage", f"{v2_cedar_pm['coverage']:.4f}", f"{v5_cedar_pm['coverage']:.4f}"])
        w.writerow(["CEDAR skilled auto FAR", f"{v2_cedar_pm['auto_far']:.4f}", f"{v5_cedar_pm['auto_far']:.4f}"])
        w.writerow(["CEDAR genuine auto FRR", f"{v2_cedar_pm['auto_frr']:.4f}", f"{v5_cedar_pm['auto_frr']:.4f}"])
        w.writerow(["CEDAR auto accuracy", f"{v2_cedar_pm['auto_acc']:.4f}", f"{v5_cedar_pm['auto_acc']:.4f}"])
        w.writerow(["SSBI auto coverage", f"{v2_ssbi_pm['coverage']:.4f}", f"{v5_ssbi_pm['coverage']:.4f}"])
        w.writerow(["SSBI skilled auto FAR", f"{v2_ssbi_pm['auto_far']:.4f}", f"{v5_ssbi_pm['auto_far']:.4f}"])
        w.writerow(["SSBI genuine auto FRR", f"{v2_ssbi_pm['auto_frr']:.4f}", f"{v5_ssbi_pm['auto_frr']:.4f}"])
        w.writerow(["SSBI auto accuracy", f"{v2_ssbi_pm['auto_acc']:.4f}", f"{v5_ssbi_pm['auto_acc']:.4f}"])
        w.writerow(["Signer7 auto coverage", f"{v2_s7_pm['coverage']:.4f}", f"{v5_s7_pm['coverage']:.4f}"])
        w.writerow(["Signer7 forged auto accepts", v2_s7_pm["s_con"], v5_s7_pm["s_con"]])
        w.writerow(["Signer7 genuine auto rejects", v2_s7_pm["g_non"], v5_s7_pm["g_non"]])

    # Plots
    # score_distribution_with_thresholds (VAL)
    plt.figure(figsize=(10,6))
    plt.hist(genuine, bins=30, alpha=0.6, label="genuine", color="green", density=False)
    plt.hist(skilled, bins=30, alpha=0.6, label="skilled", color="red", density=False)
    plt.axvline(L, color="black", linestyle="--", label=f"L={L:.3f}")
    plt.axvline(U, color="blue", linestyle="--", label=f"U={U:.3f}")
    plt.axvline(val_res["threshold"], color="orange", linestyle=":", label=f"EER thr={val_res['threshold']:.3f}")
    plt.xlabel("mean raw cosine (K5)")
    plt.ylabel("count")
    plt.title("VAL score distribution with thresholds (Phase8)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"score_distribution_with_thresholds.png", dpi=150)
    plt.close()

    # policy_coverage.png : frontier coverage vs thresholds?
    plt.figure(figsize=(8,5))
    # frontier sorted by coverage
    frontier_sorted=sorted([c for _,c in frontier], key=lambda c: c["coverage"])
    xs=[c["coverage"] for c in frontier_sorted]
    ys=[c["auto_far"] for c in frontier_sorted]
    labels=[n for n,_ in frontier]
    plt.plot(xs, ys, marker="o")
    for i,(name,c) in enumerate(frontier):
        plt.annotate(name, (c["coverage"], c["auto_far"]))
    plt.scatter([primary["coverage"]],[primary["auto_far"]], color="red", s=150, edgecolors="black", label="PRIMARY")
    plt.xlabel("auto coverage")
    plt.ylabel("auto FAR")
    plt.title("Policy frontier (VAL)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir/"policy_coverage.png", dpi=150)
    plt.close()

    # Final report
    with open(out_dir/"PHASE8_POLICY_REPORT.md","w") as f:
        f.write(f"# Phase 8 Policy Report\n\n")
        f.write(f"Model SHA {sha_before}\n")
        f.write(f"K=5 mean raw cosine, L={L:.6f} U={U:.6f}\n\n")
        f.write(f"Calibration VAL only: CEDAR {cfg.cedar_val} SSBI [1,8] (19 genuine-only)\n")
        f.write(f"VAL counts genuine {len(genuine)} skilled {len(skilled)}\n\n")
        f.write(f"## Frontier\n")
        for name,c in frontier:
            f.write(f"- {name}: L {c['L']:.4f} U {c['U']:.4f} cov {c['coverage']:.3f} man {c['manual_rate']:.3f} FAR {c['auto_far']:.4f} FRR {c['auto_frr']:.4f} acc {c['auto_acc']:.3f} g {c['g_con']}/{c['g_man']}/{c['g_non']} s {c['s_con']}/{c['s_man']}/{c['s_non']}\n")
        f.write(f"\nPrimary {reason} L {L:.6f} U {U:.6f}\n")
        f.write(f"VAL: g {primary['g_con']}/{primary['g_man']}/{primary['g_non']} s {primary['s_con']}/{primary['s_man']}/{primary['s_non']} cov {primary['coverage']:.3f} FAR {primary['auto_far']:.4f} FRR {primary['auto_frr']:.4f}\n")
        f.write(f"\nCEDAR TEST: g {cedar_pm['g_con']}/{cedar_pm['g_man']}/{cedar_pm['g_non']} s {cedar_pm['s_con']}/{cedar_pm['s_man']}/{cedar_pm['s_non']} cov {cedar_pm['coverage']:.3f} FAR {cedar_pm['auto_far']:.4f} FRR {cedar_pm['auto_frr']:.4f}\n")
        f.write(f"SSBI TEST: g {ssbi_pm['g_con']}/{ssbi_pm['g_man']}/{ssbi_pm['g_non']} s {ssbi_pm['s_con']}/{ssbi_pm['s_man']}/{ssbi_pm['s_non']} cov {ssbi_pm['coverage']:.3f} FAR {ssbi_pm['auto_far']:.4f} FRR {ssbi_pm['auto_frr']:.4f}\n")
        f.write(f"Signer7: g {s7_pm['g_con']}/{s7_pm['g_man']}/{s7_pm['g_non']} forg {s7_pm['s_con']}/{s7_pm['s_man']}/{s7_pm['s_non']} cov {s7_pm['coverage']:.3f}\n")
        f.write(f"Random impostor test: con {r_con} man {r_man} non {r_non}\n")
        f.write(f"\nV2 thresholds L {V2_L} U {V2_U}\n")
        f.write(f"Limitations: Validation SSBI small (2 writers skilled), FAR=0 on VAL does NOT prove real-world FAR=0, thresholds experimental, not bank-grade.\n")

    sha_after=sha256(ckpt_path)
    print(f"SHA after {sha_after} unchanged {sha_before==sha_after}")
    print(f"Artifacts written to {out_dir}")

if __name__=="__main__":
    main()
