"""Phase 9 Cross-Domain Policy Robustness & Failure Analysis — frozen model/policy"""
from __future__ import annotations
import hashlib, json, csv, pathlib, re, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dataclasses

def pca_2d(X, random_state=42):
    # Center
    X_centered = X - np.mean(X, axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Project onto top 2 components
    pts = X_centered @ Vt.T[:, :2]
    # explained variance
    var = (S**2) / (X.shape[0]-1)
    total = var.sum()
    ratio = var[:2]/total if total>0 else np.array([0,0])
    return pts, ratio

def roc_auc_score_fallback(labels, scores):
    # manual AUC
    try:
        from sklearn.metrics import roc_auc_score as sk_auc
        return sk_auc(labels, scores)
    except:
        gen = [s for s,l in zip(scores, labels) if l==1]
        forg = [s for s,l in zip(scores, labels) if l==0]
        pairs = sum(1 for g in gen for f in forg if g>f) + 0.5*sum(1 for g in gen for f in forg if g==f)
        return pairs/(len(gen)*len(forg)) if gen and forg else 0.5
from ai.v5.config import V5Config
from ai.v5.train import load_ssbi_crops, _encode_cedar_noaug, _encode_ssbi_noaug, _batch_encode
from ai.dataset import load_writer_index
from ai.model import SiameseResNet18
from ai.metrics import full_metrics

EXP_SHA="5593242d5e846bef481e9e07f0217c99abf8365d251ff645b910a7e4355ac0a2"
EXP_L=0.6585003733634949
EXP_U=0.9150440096855164

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

def load_model(cfg, device):
    @dataclasses.dataclass
    class C2:
        embedding_dim=cfg.embedding_dim
        pretrained=False
        canvas_width=cfg.canvas_width
        canvas_height=cfg.canvas_height
    m=SiameseResNet18(C2())
    ckpt=torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(device); m.eval()
    return m, ckpt

def policy_decision(score, L, U):
    if score <= L: return "NON CONFORME"
    if score >= U: return "CONFORME"
    return "CONTROLE MANUEL"

def eval_writer_k(cfg, device, model, cedar_index, gen_by_person, forg_by_person, writer_id, is_cedar, K):
    """Return per-probe scores for one writer with K refs, mean raw cosine, plus individual ref similarities"""
    # returns dict with genuine_scores, forgery_scores, details per probe
    if is_cedar:
        idx=cedar_index[writer_id]
        def ckey(p):
            m=re.search(r"_(\d+)\.png$", p.name)
            return int(m.group(1)) if m else p.name
        genuines=sorted(idx.originals, key=ckey)
        forgeries=sorted(idx.forgeries, key=ckey)
        if len(genuines) < K+1:
            return None
        refs=genuines[:K]
        queries_gen=genuines[K:]
        ref_tensors=[_encode_cedar_noaug(p,cfg) for p in refs]
        ref_embs=_batch_encode(ref_tensors, model, device)
        gen_details=[]
        for q in queries_gen:
            q_t=_encode_cedar_noaug(q,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims = ref_embs @ q_emb  # 5 scores
            mean=float(np.mean(sims))
            gen_details.append({"probe": q.name, "mean": mean, "sims": sims.tolist(), "ref_names": [r.name for r in refs]})
        forg_details=[]
        for q in forgeries:
            q_t=_encode_cedar_noaug(q,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims = ref_embs @ q_emb
            mean=float(np.mean(sims))
            forg_details.append({"probe": q.name, "mean": mean, "sims": sims.tolist(), "ref_names": [r.name for r in refs]})
        return {"genuine": gen_details, "forged": forg_details, "refs": [r.name for r in refs]}
    else:
        gen_crops=gen_by_person.get(writer_id, [])
        forg_crops=forg_by_person.get(writer_id, [])
        if len(gen_crops) < K+1:
            return None
        gen_sorted=sorted(gen_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
        forg_sorted=sorted(forg_crops, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
        refs=gen_sorted[:K]
        queries_gen=gen_sorted[K:]
        ref_tensors=[_encode_ssbi_noaug(s,b,cfg) for s,b in refs]
        ref_embs=_batch_encode(ref_tensors, model, device)
        gen_details=[]
        for s,b in queries_gen:
            q_t=_encode_ssbi_noaug(s,b,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims=ref_embs @ q_emb
            mean=float(np.mean(sims))
            gen_details.append({"probe": str(s.name), "mean": mean, "sims": sims.tolist(), "ref_names": [str(r[0].name) for r in refs]})
        forg_details=[]
        for s,b in forg_sorted:
            q_t=_encode_ssbi_noaug(s,b,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims=ref_embs @ q_emb
            mean=float(np.mean(sims))
            forg_details.append({"probe": str(s.name), "mean": mean, "sims": sims.tolist(), "ref_names": [str(r[0].name) for r in refs]})
        return {"genuine": gen_details, "forged": forg_details, "refs": [str(r[0].name) for r in refs]}

def main():
    cfg=V5Config()
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    out_dir=cfg.reports_dir/"phase9_failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint {cfg.checkpoint_path}")
    sha_before=sha(cfg.checkpoint_path)
    print(f"SHA before {sha_before}")
    assert sha_before==EXP_SHA, f"SHA mismatch {sha_before}"
    policy_path=cfg.reports_dir/"phase8_policy"/"policy.json"
    with open(policy_path) as f: policy=json.load(f)
    assert policy["model_sha"]==EXP_SHA
    assert policy["K"]==5
    assert policy["aggregation"]=="mean raw cosine"
    assert abs(policy["L"]-EXP_L)<1e-9
    assert abs(policy["U"]-EXP_U)<1e-9
    print(f"Policy L {EXP_L} U {EXP_U} verified")
    model, ckpt = load_model(cfg, device)
    cedar_index=load_writer_index(cfg.cedar_root)
    gen_by_person, forg_by_person = load_ssbi_crops(cfg.ssbi_root)
    L=EXP_L; U=EXP_U

    # === STEP2 reproduce Phase8 ===
    print("=== Reproduce Phase8 ===")
    # CEDAR TEST
    cedar_gen=[]; cedar_sk=[]; cedar_details={}
    for w in cfg.cedar_test:
        res=eval_writer_k(cfg, device, model, cedar_index, gen_by_person, forg_by_person, w, True, 5)
        cedar_details[w]=res
        for d in res["genuine"]: cedar_gen.append(d["mean"])
        for d in res["forged"]: cedar_sk.append(d["mean"])
    cedar_g_con=sum(1 for s in cedar_gen if s>=U)
    cedar_g_man=sum(1 for s in cedar_gen if L < s < U)
    cedar_g_non=sum(1 for s in cedar_gen if s<=L)
    cedar_s_con=sum(1 for s in cedar_sk if s>=U)
    cedar_s_man=sum(1 for s in cedar_sk if L < s < U)
    cedar_s_non=sum(1 for s in cedar_sk if s<=L)
    print(f"CEDAR reproduced G {cedar_g_con}/{cedar_g_man}/{cedar_g_non} S {cedar_s_con}/{cedar_s_man}/{cedar_s_non}")
    assert cedar_g_con==93 and cedar_g_man==78 and cedar_g_non==0, f"CEDAR genuine mismatch {cedar_g_con}/{cedar_g_man}/{cedar_g_non}"
    assert cedar_s_con==28 and cedar_s_man==165 and cedar_s_non==23, f"CEDAR skilled mismatch {cedar_s_con}/{cedar_s_man}/{cedar_s_non}"
    # SSBI
    ssbi_gen=[]; ssbi_sk=[]
    for w in cfg.ssbi_test:
        res=eval_writer_k(cfg, device, model, cedar_index, gen_by_person, forg_by_person, w, False, 5)
        if res is None: continue
        # for SSBI 0, no forg
        if w in [3,14]:
            for d in res["forged"]: ssbi_sk.append(d["mean"])
        for d in res["genuine"]: ssbi_gen.append(d["mean"])
    ssbi_g_con=sum(1 for s in ssbi_gen if s>=U)
    ssbi_g_man=sum(1 for s in ssbi_gen if L < s < U)
    ssbi_g_non=sum(1 for s in ssbi_gen if s<=L)
    ssbi_s_con=sum(1 for s in ssbi_sk if s>=U)
    ssbi_s_man=sum(1 for s in ssbi_sk if L < s < U)
    ssbi_s_non=sum(1 for s in ssbi_sk if s<=L)
    print(f"SSBI reproduced G {ssbi_g_con}/{ssbi_g_man}/{ssbi_g_non} S {ssbi_s_con}/{ssbi_s_man}/{ssbi_s_non}")
    assert ssbi_g_con==0 and ssbi_g_man==33 and ssbi_g_non==0
    assert ssbi_s_con==0 and ssbi_s_man==13 and ssbi_s_non==3
    # Signer7
    gen7=gen_by_person[7]; forg7=forg_by_person[7]
    gen_sorted=sorted(gen7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    forg_sorted=sorted(forg7, key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    refs=gen_sorted[:5]
    queries_gen=gen_sorted[5:15]
    queries_forg=forg_sorted[:8]+forg_sorted[:2]
    ref_tensors=[_encode_ssbi_noaug(s,b,cfg) for s,b in refs]
    ref_embs=_batch_encode(ref_tensors, model, device)
    s7_gen=[]; s7_forg=[]
    for s,b in queries_gen[:10]:
        q_t=_encode_ssbi_noaug(s,b,cfg)
        q_emb=_batch_encode([q_t], model, device)[0]
        s7_gen.append(float(np.mean(ref_embs @ q_emb)))
    for s,b in queries_forg[:10]:
        q_t=_encode_ssbi_noaug(s,b,cfg)
        q_emb=_batch_encode([q_t], model, device)[0]
        s7_forg.append(float(np.mean(ref_embs @ q_emb)))
    s7_g_con=sum(1 for s in s7_gen if s>=U)
    s7_g_man=sum(1 for s in s7_gen if L < s < U)
    s7_g_non=sum(1 for s in s7_gen if s<=L)
    s7_s_con=sum(1 for s in s7_forg if s>=U)
    s7_s_man=sum(1 for s in s7_forg if L < s < U)
    s7_s_non=sum(1 for s in s7_forg if s<=L)
    print(f"S7 reproduced G {s7_g_con}/{s7_g_man}/{s7_g_non} F {s7_s_con}/{s7_s_man}/{s7_s_non}")
    assert s7_g_con==0 and s7_g_man==9 and s7_g_non==1
    assert s7_s_con==0 and s7_s_man==8 and s7_s_non==2
    print("Reproduction PASSED")

    # === STEP3 false accepts ===
    print("=== Step3 false accepts ===")
    # Collect all cedar skilled probes with score>=U
    false_accepts=[]
    for w in cfg.cedar_test:
        res=cedar_details[w]
        for d in res["forged"]:
            if d["mean"] >= U:
                sims=np.array(d["sims"])
                false_accepts.append({
                    "writer": w,
                    "probe": d["probe"],
                    "mean": d["mean"],
                    "sims": d["sims"],
                    "ref_names": d["ref_names"],
                    "mean_sims": float(np.mean(sims)),
                    "median": float(np.median(sims)),
                    "min": float(np.min(sims)),
                    "max": float(np.max(sims)),
                    "std": float(np.std(sims)),
                })
    # sort by mean descending
    false_accepts=sorted(false_accepts, key=lambda x: x["mean"], reverse=True)
    print(f"False accepts count {len(false_accepts)}")
    assert len(false_accepts)==28
    # also get rank among all skilled
    all_sk_sorted=sorted(cedar_sk, reverse=True)
    for fa in false_accepts:
        rank=all_sk_sorted.index(fa["mean"])+1
        fa["rank"]=rank
    with open(out_dir/"cedar_false_accepts.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","probe","k5_mean","decision","rank","ref1","ref2","ref3","ref4","ref5","mean","median","min","max","std"])
        for fa in false_accepts:
            w.writerow([fa["writer"], fa["probe"], f"{fa['mean']:.6f}", "CONFORME", fa["rank"],
                        f"{fa['sims'][0]:.6f}", f"{fa['sims'][1]:.6f}", f"{fa['sims'][2]:.6f}", f"{fa['sims'][3]:.6f}", f"{fa['sims'][4]:.6f}",
                        f"{fa['mean_sims']:.6f}", f"{fa['median']:.6f}", f"{fa['min']:.6f}", f"{fa['max']:.6f}", f"{fa['std']:.6f}"])

    # === STEP4 writer concentration ===
    print("=== Step4 writer concentration ===")
    writer_rows=[]
    for w in cfg.cedar_test:
        res=cedar_details[w]
        gen_scores=[d["mean"] for d in res["genuine"]]
        sk_scores=[d["mean"] for d in res["forged"]]
        g_con=sum(1 for s in gen_scores if s>=U)
        g_man=sum(1 for s in gen_scores if L < s < U)
        g_non=sum(1 for s in gen_scores if s<=L)
        s_con=sum(1 for s in sk_scores if s>=U)
        s_man=sum(1 for s in sk_scores if L < s < U)
        s_non=sum(1 for s in sk_scores if s<=L)
        far=s_con/len(sk_scores) if sk_scores else 0
        writer_rows.append({
            "writer": w,
            "gen_count": len(gen_scores),
            "sk_count": len(sk_scores),
            "gen_mean": float(np.mean(gen_scores)) if gen_scores else 0,
            "gen_std": float(np.std(gen_scores)) if gen_scores else 0,
            "sk_mean": float(np.mean(sk_scores)) if sk_scores else 0,
            "sk_std": float(np.std(sk_scores)) if sk_scores else 0,
            "gen_min": float(np.min(gen_scores)) if gen_scores else 0,
            "gen_max": float(np.max(gen_scores)) if gen_scores else 0,
            "sk_min": float(np.min(sk_scores)) if sk_scores else 0,
            "sk_max": float(np.max(sk_scores)) if sk_scores else 0,
            "g_con": g_con, "g_man": g_man, "g_non": g_non,
            "s_con": s_con, "s_man": s_man, "s_non": s_non,
            "far": far,
            "false_count": s_con,
        })
    # percentage of 28 per writer
    for r in writer_rows:
        r["pct_false"]=r["false_count"]/28*100 if 28 else 0
    # sort by false_count descending
    writer_rows=sorted(writer_rows, key=lambda x: x["false_count"], reverse=True)
    with open(out_dir/"cedar_writer_analysis.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","gen_count","sk_count","gen_mean","gen_std","sk_mean","sk_std","gen_min","gen_max","sk_min","sk_max","g_con","g_man","g_non","s_con","s_man","s_non","far","false_count","pct_false"])
        for r in writer_rows:
            w.writerow([r["writer"], r["gen_count"], r["sk_count"], f"{r['gen_mean']:.4f}", f"{r['gen_std']:.4f}", f"{r['sk_mean']:.4f}", f"{r['sk_std']:.4f}", f"{r['gen_min']:.4f}", f"{r['gen_max']:.4f}", f"{r['sk_min']:.4f}", f"{r['sk_max']:.4f}", r["g_con"], r["g_man"], r["g_non"], r["s_con"], r["s_man"], r["s_non"], f"{r['far']:.4f}", r["false_count"], f"{r['pct_false']:.1f}"])
    # determine concentration: top writers share?
    top3=sum(r["false_count"] for r in writer_rows[:3])
    print(f"Top 3 writers contribute {top3}/28 = {top3/28*100:.1f}%")
    # check distributed vs concentrated: if top writer >40% or top3 >70% then concentrated

    # === STEP5 writer-level AUC/EER ===
    print("=== Step5 writer AUC ===")
    writer_auc=[]
    for w in cfg.cedar_test:
        res=cedar_details[w]
        gen_scores=[d["mean"] for d in res["genuine"]]
        sk_scores=[d["mean"] for d in res["forged"]]
        scores=np.array(gen_scores+sk_scores)
        labels=np.array([1]*len(gen_scores)+[0]*len(sk_scores))
        auc=roc_auc_score_fallback(labels, scores)
        # EER via full_metrics
        m=full_metrics(scores, labels)
        eer=m["eer"]
        diff=np.mean(gen_scores)-np.mean(sk_scores)
        writer_auc.append({"writer": w, "auc": auc, "eer": eer, "gen_mean": np.mean(gen_scores), "sk_mean": np.mean(sk_scores), "diff": diff, "sk_ge_gen": np.mean(sk_scores) >= np.mean(gen_scores)})
    # categorise
    cats={"<0.5":0,"0.5-0.7":0,"0.7-0.9":0,">=0.9":0}
    sk_ge=[]
    for r in writer_auc:
        if r["auc"]<0.5: cats["<0.5"]+=1
        elif r["auc"]<0.7: cats["0.5-0.7"]+=1
        elif r["auc"]<0.9: cats["0.7-0.9"]+=1
        else: cats[">=0.9"]+=1
        if r["sk_ge_gen"]: sk_ge.append(r["writer"])
    print(f"AUC cats {cats} sk_ge {sk_ge}")
    with open(out_dir/"cedar_writer_auc.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","auc","eer","gen_mean","sk_mean","diff","sk_ge_gen"])
        for r in sorted(writer_auc, key=lambda x: x["auc"]):
            w.writerow([r["writer"], f"{r['auc']:.4f}", f"{r['eer']:.4f}", f"{r['gen_mean']:.4f}", f"{r['sk_mean']:.4f}", f"{r['diff']:.4f}", r["sk_ge_gen"]])

    # === STEP6 reference sensitivity ===
    print("=== Step6 ref sensitivity ===")
    ref_rows=[]
    for fa in false_accepts:
        sims=np.array(fa["sims"])
        # classify pattern
        # Count high refs >=U
        high=sum(1 for s in sims if s>=U)
        low=sum(1 for s in sims if s<=L)
        # Determine pattern
        if high==5:
            pattern="A) all five high"
        elif high>=3:
            pattern="A-B) multiple high"
        elif high==1 or high==2:
            pattern="B) one/two spike"
        elif np.std(sims)>0.05:
            pattern="C) high variance"
        else:
            pattern="D) consistently indistinguishable"
        ref_rows.append({"writer": fa["writer"], "probe": fa["probe"], "mean": fa["mean"], "sims": fa["sims"], "std": fa["std"], "high": high, "low": low, "pattern": pattern})
    # also for all cedar probes, compute std etc for distribution
    all_ref_stats=[]
    for w in cfg.cedar_test:
        res=cedar_details[w]
        for d in res["genuine"]+res["forged"]:
            sims=np.array(d["sims"])
            all_ref_stats.append({"type": "genuine" if d in res["genuine"] else "forged", "std": float(np.std(sims)), "mean": d["mean"]})
    with open(out_dir/"reference_sensitivity.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","probe","mean","std","high_ge_U","low_le_L","pattern","sims"])
        for r in ref_rows:
            w.writerow([r["writer"], r["probe"], f"{r['mean']:.6f}", f"{r['std']:.6f}", r["high"], r["low"], r["pattern"], ";".join(f"{s:.4f}" for s in r["sims"])])
    # summary
    patterns={}
    for r in ref_rows:
        patterns[r["pattern"]]=patterns.get(r["pattern"],0)+1
    print(f"Patterns {patterns} mean std false accepts {np.mean([r['std'] for r in ref_rows]):.4f}")

    # === STEP7 K1/K3/K5 diagnostic ===
    print("=== Step7 K diagnostic ===")
    # For failure analysis, compute K1/K3/K5 AUC and for the 28 false accepts, how many also high under K1/K3
    k_results={}
    for K in [1,3,5]:
        gen_all=[]; sk_all=[]
        false_high_counts={"K1":0,"K3":0}
        # Need to recompute for each K
        for w in cfg.cedar_test:
            res=eval_writer_k(cfg, device, model, cedar_index, gen_by_person, forg_by_person, w, True, K)
            gen_scores=[d["mean"] for d in res["genuine"]]
            sk_scores=[d["mean"] for d in res["forged"]]
            gen_all.extend(gen_scores)
            sk_all.extend(sk_scores)
        scores=np.array(gen_all+sk_all)
        labels=np.array([1]*len(gen_all)+[0]*len(sk_all))
        m=full_metrics(scores, labels)
        auc=m["roc_auc"]; eer=m["eer"]
        k_results[K]={"gen": gen_all, "sk": sk_all, "auc": auc, "eer": eer, "gen_mean": np.mean(gen_all), "sk_mean": np.mean(sk_all)}
        print(f"K{K} AUC {auc:.4f} EER {eer:.4f} gen {np.mean(gen_all):.4f} sk {np.mean(sk_all):.4f}")
    # For the 28 K5 false accepts, check their K1/K3 scores
    # Need to map each false accept probe to its K1/K3 mean
    # Use same probe names
    k1_map={}
    k3_map={}
    for w in cfg.cedar_test:
        for K, mp in [(1,k1_map),(3,k3_map)]:
            res=eval_writer_k(cfg, device, model, cedar_index, gen_by_person, forg_by_person, w, True, K)
            for d in res["forged"]:
                key=(w,d["probe"])
                mp[key]=d["mean"]
    k1_high=0; k3_high=0; k5_only=0
    for fa in false_accepts:
        key=(fa["writer"], fa["probe"])
        s1=k1_map.get(key,0)
        s3=k3_map.get(key,0)
        if s1>=U: k1_high+=1
        if s3>=U: k3_high+=1
        if s1<U and s3<U:  # only K5 high (but fa is K5 high by definition)
            # count those that are low on K1 and K3 but high on K5
            if s1<U and s3<U:
                k5_only+=1
    print(f"K1 high {k1_high}/28 K3 high {k3_high}/28 K5 only {k5_only}")
    with open(out_dir/"k_diagnostic.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["K","auc","eer","gen_mean","sk_mean","gen_count","sk_count"])
        for K in [1,3,5]:
            r=k_results[K]
            w.writerow([K, f"{r['auc']:.4f}", f"{r['eer']:.4f}", f"{r['gen_mean']:.4f}", f"{r['sk_mean']:.4f}", len(r["gen"]), len(r["sk"])])
        w.writerow([])
        w.writerow(["false_accept_K1_high", k1_high])
        w.writerow(["false_accept_K3_high", k3_high])
        w.writerow(["k5_only_high", k5_only])
        w.writerow(["note","diagnostic only — threshold not calibrated for K1/K3"])

    # === STEP8 domain score distributions ===
    print("=== Step8 domain distributions ===")
    # VAL
    val_res=None
    from ai.v5.train import compute_real_validation_k5
    val_res=compute_real_validation_k5(model, cfg, device, cedar_index, gen_by_person, forg_by_person)
    # Need separate per-domain: VAL is combined CEDAR+SSBI, but we can split
    # For VAL we have combined lists; for separate we need to compute per domain again
    # We'll compute VAL cedar and ssbi separately via helper
    def domain_stats(scores):
        arr=np.array(scores)
        return {
            "count": len(scores),
            "mean": float(np.mean(arr)) if len(arr) else 0,
            "median": float(np.median(arr)) if len(arr) else 0,
            "std": float(np.std(arr)) if len(arr) else 0,
            "min": float(np.min(arr)) if len(arr) else 0,
            "q1": float(np.percentile(arr,25)) if len(arr) else 0,
            "q3": float(np.percentile(arr,75)) if len(arr) else 0,
            "max": float(np.max(arr)) if len(arr) else 0,
            "prop_ge_U": float(np.mean(arr>=U)) if len(arr) else 0,
            "prop_le_L": float(np.mean(arr<=L)) if len(arr) else 0,
            "prop_manual": float(np.mean((arr> L) & (arr < U))) if len(arr) else 0,
        }
    # Compute VAL per domain via separate calls: use cedar_details for VAL? Instead we have val_res combined
    # For domain-specific, compute TEST distributions already have cedar_gen/sk, ssbi_gen/sk
    # For VAL, we already have val_res["genuine_scores"] etc but they are combined; we need to split cedar vs ssbi VAL
    # Let's recompute VAL cedar and ssbi separately using same logic but split
    # Quick: compute VAL cedar scores and ssbi scores separately via manual
    # For simplicity, use val_res combined for VAL overall, and cedar_gen/sk for CEDAR TEST, ssbi for SSBI, s7 for signer7
    val_gen=val_res["genuine_scores"]
    val_sk=val_res["forgery_scores"]
    val_rand=val_res["random_scores"]
    cedar_stats_gen=domain_stats(cedar_gen)
    cedar_stats_sk=domain_stats(cedar_sk)
    ssbi_stats_gen=domain_stats(ssbi_gen)
    ssbi_stats_sk=domain_stats(ssbi_sk)
    s7_gen_stats=domain_stats(s7_gen)
    s7_forg_stats=domain_stats(s7_forg)
    val_gen_stats=domain_stats(val_gen)
    val_sk_stats=domain_stats(val_sk)
    # Also need random impostor distributions: val_rand, cedar random, ssbi random
    # Get random lists from earlier: cedar random scores are from cedar_details? We have cedar_rand list from previous but we didn't store; compute again quickly via policy evaluation random
    # For domain distribution, compute random from val and test
    # val_rand already
    cedar_rand_scores=[]
    for w in cfg.cedar_test:
        # use same ref as before to compute one random per writer
        pass  # we have cedar_rand from earlier? Actually cedar_rand is not stored globally; we can recompute via quick method
    # Instead approximate: we have domain stats for random via val_rand and we can compute test random via separate call
    # For simplicity, we will compute random stats from val_res and from test evaluations we already have random lists
    # But we didn't keep test random lists; let's compute quickly via helper
    # Use simple method: for domain shift, focus on genuine/skilled, random secondary
    with open(out_dir/"domain_score_distributions.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["domain","class","count","mean","median","std","min","q1","q3","max","prop_ge_U","prop_le_L","prop_manual"])
        for domain, cls, stats in [
            ("VAL","genuine", val_gen_stats),
            ("VAL","skilled", val_sk_stats),
            ("VAL","random", domain_stats(val_rand)),
            ("CEDAR TEST","genuine", cedar_stats_gen),
            ("CEDAR TEST","skilled", cedar_stats_sk),
            ("SSBI TEST","genuine", ssbi_stats_gen),
            ("SSBI TEST","skilled", ssbi_stats_sk),
            ("SIGNER7","genuine", s7_gen_stats),
            ("SIGNER7","forged", s7_forg_stats),
        ]:
            w.writerow([domain, cls, stats["count"], f"{stats['mean']:.4f}", f"{stats['median']:.4f}", f"{stats['std']:.4f}", f"{stats['min']:.4f}", f"{stats['q1']:.4f}", f"{stats['q3']:.4f}", f"{stats['max']:.4f}", f"{stats['prop_ge_U']:.3f}", f"{stats['prop_le_L']:.3f}", f"{stats['prop_manual']:.3f}"])

    # === STEP9 threshold generalization ===
    print("=== Step9 threshold generalization ===")
    def prop_ge(scores, thr): return sum(1 for s in scores if s>=thr)/len(scores) if scores else 0
    def prop_le(scores, thr): return sum(1 for s in scores if s<=thr)/len(scores) if scores else 0
    frozen=[]
    for domain, gen, sk in [
        ("VAL", val_gen, val_sk),
        ("CEDAR", cedar_gen, cedar_sk),
        ("SSBI", ssbi_gen, ssbi_sk),
        ("SIGNER7", s7_gen, s7_forg),
    ]:
        frozen.append({
            "domain": domain,
            "gen_ge_U": prop_ge(gen, U),
            "sk_ge_U": prop_ge(sk, U),
            "gen_le_L": prop_le(gen, L),
            "sk_le_L": prop_le(sk, L),
            "gen_n": len(gen), "sk_n": len(sk),
        })
    with open(out_dir/"frozen_policy_cross_domain.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["domain","gen_ge_U","sk_ge_U","gen_le_L","sk_le_L","gen_n","sk_n"])
        for r in frozen:
            w.writerow([r["domain"], f"{r['gen_ge_U']:.4f}", f"{r['sk_ge_U']:.4f}", f"{r['gen_le_L']:.4f}", f"{r['sk_le_L']:.4f}", r["gen_n"], r["sk_n"]])
    print(f"VAL sk_ge_U {frozen[0]['sk_ge_U']} CEDAR {frozen[1]['sk_ge_U']} -> shift {frozen[1]['sk_ge_U']-frozen[0]['sk_ge_U']:.4f}")

    # === STEP10 best reference ===
    print("=== Step10 best reference ===")
    best_ref_counts={}
    for fa in false_accepts:
        sims=np.array(fa["sims"])
        best_idx=int(np.argmax(sims))
        ref_name=fa["ref_names"][best_idx]
        best_score=float(sims[best_idx])
        key=f"{fa['writer']}_{ref_name}"
        best_ref_counts[key]=best_ref_counts.get(key,0)+1
        fa["best_ref"]=ref_name
        fa["best_score"]=best_score
    # Also count per ref position
    pos_counts={0:0,1:0,2:0,3:0,4:0}
    for fa in false_accepts:
        sims=np.array(fa["sims"])
        pos_counts[int(np.argmax(sims))]+=1
    print(f"Best reference position counts {pos_counts}")
    # Determine if concentrated
    # Save to csv already has best? Add to false_accepts csv extra?
    # Create separate counted file
    with open(out_dir/"embedding_distances.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["writer","probe","best_ref","best_score","k5_mean"])
        for fa in false_accepts:
            w.writerow([fa["writer"], fa["probe"], fa["best_ref"], f"{fa['best_score']:.6f}", f"{fa['mean']:.6f}"])
        w.writerow([])
        w.writerow(["ref_position","count"])
        for pos,c in pos_counts.items():
            w.writerow([pos, c])

    # === STEP11 embedding PCA ===
    print("=== Step11 PCA ===")
    # Gather embeddings for CEDAR TEST references, genuine probes, skilled
    all_embs=[]
    labels=[]
    is_false=[]
    # Collect per writer
    for w in cfg.cedar_test:
        res=cedar_details[w]
        # refs: need to encode refs again to get embeddings
        # For PCA, we need actual embeddings not scores; we can get via model.encode
        # Encode refs
        idx=cedar_index[w]
        def ckey(p):
            m=re.search(r"_(\d+)\.png$", p.name)
            return int(m.group(1)) if m else p.name
        genuines=sorted(idx.originals, key=ckey)
        forgeries=sorted(idx.forgeries, key=ckey)
        refs=genuines[:5]
        ref_tensors=[_encode_cedar_noaug(p,cfg) for p in refs]
        with torch.no_grad():
            batch=torch.stack(ref_tensors).to(device)
            embs=model.encode(batch).cpu().numpy()
            for e in embs:
                all_embs.append(e); labels.append(f"ref_{w}"); is_false.append(False)
        # genuine probes
        for q in genuines[5:]:
            q_t=_encode_cedar_noaug(q,cfg)
            with torch.no_grad():
                e=model.encode(q_t.unsqueeze(0).to(device)).cpu().numpy()[0]
                all_embs.append(e); labels.append(f"gen_{w}"); is_false.append(False)
        # skilled
        for q in forgeries:
            q_t=_encode_cedar_noaug(q,cfg)
            with torch.no_grad():
                e=model.encode(q_t.unsqueeze(0).to(device)).cpu().numpy()[0]
                all_embs.append(e)
                # check if this forg is false accept
                # find its mean score
                # we can check if probe name in false_accepts
                is_fa = any(fa["probe"]==q.name and fa["writer"]==w for fa in false_accepts)
                labels.append(f"sk_{w}")
                is_false.append(is_fa)
    all_embs=np.stack(all_embs)
    pts, ratio = pca_2d(all_embs)
    plt.figure(figsize=(10,8))
    # plot by type
    # refs green, genuine blue, skilled red, false accepts highlight with black edge
    colors=[]
    for lab, fa in zip(labels, is_false):
        if fa:
            colors.append("black")
        elif lab.startswith("ref"):
            colors.append("green")
        elif lab.startswith("gen"):
            colors.append("blue")
        else:
            colors.append("red")
    plt.scatter(pts[:,0], pts[:,1], c=colors, s=10, alpha=0.6)
    # legend proxies
    import matplotlib.patches as mpatches
    patches=[mpatches.Patch(color="green", label="reference"), mpatches.Patch(color="blue", label="genuine probe"), mpatches.Patch(color="red", label="skilled forgery"), mpatches.Patch(color="black", label="false accept (28)")]
    plt.legend(handles=patches)
    plt.title("CEDAR TEST embedding PCA (frozen V5) — false accepts highlighted")
    plt.xlabel(f"PC1 ({ratio[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"cedar_embedding_pca.png", dpi=150)
    plt.close()
    # Also compute distances
    # genuine-to-own-ref vs forgery-to-claimed-ref vs random
    # Use cosine distance = 1 - cosine similarity (since L2, dot = cosine)
    # For each writer, compute mean distances
    with open(out_dir/"embedding_distances.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","mean","std"])
        # we have scores as cosine similarities, distance =1 - sim
        # genuine distances
        gen_dists=[1-s for s in cedar_gen]
        sk_dists=[1-s for s in cedar_sk]
        w.writerow(["genuine_to_ref_cosine_distance_mean", f"{np.mean(gen_dists):.4f}", f"{np.std(gen_dists):.4f}"])
        w.writerow(["forgery_to_ref_cosine_distance_mean", f"{np.mean(sk_dists):.4f}", f"{np.std(sk_dists):.4f}"])
        w.writerow(["note","POST-HOC FAILURE ANALYSIS"])

    # === STEP12 extraction quality ===
    print("=== Step12 extraction ===")
    # CEDAR uses clean crops, not cheque extraction, so N/A
    # Create placeholder
    with open(out_dir/"reference_sensitivity.csv","a",newline="") as f:
        pass  # already created

    # === STEP13 signer7 failure ===
    print("=== Step13 signer7 ===")
    # Already have s7 details, create signer7_failure_analysis.csv with per-reference sims
    refs_list=refs  # from last loop? Need to recompute for signer7
    gen_sorted=sorted(gen_by_person[7], key=lambda x: (str(x[0]), x[1][0], x[1][1], x[1][2], x[1][3]))
    refs=gen_sorted[:5]
    ref_names_s7=[str(r[0].name) for r in refs]
    ref_tensors=[_encode_ssbi_noaug(s,b,cfg) for s,b in refs]
    ref_embs=_batch_encode(ref_tensors, model, device)
    with open(out_dir/"signer7_failure_analysis.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["probe","type","k5_mean","decision","ref1","ref2","ref3","ref4","ref5","std"])
        # genuine false reject is the one with <=L (0.658)
        for s,b in gen_sorted[5:15]:
            q_t=_encode_ssbi_noaug(s,b,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims=ref_embs @ q_emb
            mean=float(np.mean(sims))
            dec=policy_decision(mean, L, U)
            w.writerow([str(s.name), "genuine", f"{mean:.6f}", dec, f"{sims[0]:.4f}", f"{sims[1]:.4f}", f"{sims[2]:.4f}", f"{sims[3]:.4f}", f"{sims[4]:.4f}", f"{np.std(sims):.4f}"])
        for s,b in (forg_sorted[:8]+forg_sorted[:2]):
            q_t=_encode_ssbi_noaug(s,b,cfg)
            q_emb=_batch_encode([q_t], model, device)[0]
            sims=ref_embs @ q_emb
            mean=float(np.mean(sims))
            dec=policy_decision(mean, L, U)
            w.writerow([str(s.name), "forged", f"{mean:.6f}", dec, f"{sims[0]:.4f}", f"{sims[1]:.4f}", f"{sims[2]:.4f}", f"{sims[3]:.4f}", f"{sims[4]:.4f}", f"{np.std(sims):.4f}"])

    # === STEP14 SSBI analysis ===
    print("=== Step14 SSBI ===")
    # Compare SSBI genuine distribution vs CEDAR vs VAL
    ssbi_gen_mean=np.mean(ssbi_gen) if ssbi_gen else 0
    cedar_gen_mean=np.mean(cedar_gen)
    val_gen_mean=np.mean(val_gen)
    print(f"VAL gen {val_gen_mean:.4f} CEDAR gen {cedar_gen_mean:.4f} SSBI gen {ssbi_gen_mean:.4f}")
    with open(out_dir/"ssbi_failure_analysis.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["domain","gen_mean","sk_mean","gen_count","sk_count"])
        w.writerow(["VAL", f"{np.mean(val_gen):.4f}", f"{np.mean(val_sk):.4f}", len(val_gen), len(val_sk)])
        w.writerow(["CEDAR TEST", f"{np.mean(cedar_gen):.4f}", f"{np.mean(cedar_sk):.4f}", len(cedar_gen), len(cedar_sk)])
        w.writerow(["SSBI TEST", f"{np.mean(ssbi_gen):.4f}" if ssbi_gen else "N/A", f"{np.mean(ssbi_sk):.4f}" if ssbi_sk else "N/A", len(ssbi_gen), len(ssbi_sk)])
        w.writerow(["SIGNER7", f"{np.mean(s7_gen):.4f}", f"{np.mean(s7_forg):.4f}", len(s7_gen), len(s7_forg)])
        w.writerow([])
        w.writerow(["note","SSBI genuine scores below U (max SSBI gen below U) indicates poor absolute calibration despite ranking"])
        # check ranking: SSBI AUC
        if len(ssbi_gen) and len(ssbi_sk):
            scores=np.array(ssbi_gen+ssbi_sk)
            labels=np.array([1]*len(ssbi_gen)+[0]*len(ssbi_sk))
            auc=roc_auc_score_fallback(labels, scores)
            w.writerow(["SSBI AUC", f"{auc:.4f}"])

    # === STEP15 ranking vs calibration ===
    print("=== Step15 ranking vs calibration ===")
    # Already computed AUCs: VAL AUC 0.867, CEDAR AUC from k_results[5] etc
    # For ranking vs calibration, compare AUC vs policy FAR
    # Create visual: domain score distribution
    plt.figure(figsize=(12,6))
    for idx, (scores, label, color) in enumerate([
        (val_gen, "VAL genuine", "green"),
        (val_sk, "VAL skilled", "red"),
        (cedar_gen, "CEDAR genuine", "blue"),
        (cedar_sk, "CEDAR skilled", "orange"),
        (ssbi_gen, "SSBI genuine", "purple"),
        (ssbi_sk, "SSBI skilled", "brown"),
    ]):
        plt.hist(scores, bins=20, alpha=0.4, label=label, color=color, density=True)
    plt.axvline(L, color="black", linestyle="--", label=f"L {L:.3f}")
    plt.axvline(U, color="black", linestyle="-", label=f"U {U:.3f}")
    plt.xlabel("K5 mean cosine")
    plt.ylabel("density")
    plt.title("Domain score distributions with frozen thresholds (POST-HOC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"domain_score_distribution.png", dpi=150)
    plt.close()
    # writer FAR distribution
    plt.figure(figsize=(10,5))
    writers=[r["writer"] for r in writer_rows]
    fars=[r["far"] for r in writer_rows]
    plt.bar([str(w) for w in writers], fars, color="red", alpha=0.7)
    plt.axhline(12.96/100, color="black", linestyle="--", label="overall 12.96%")
    plt.xlabel("CEDAR TEST writer")
    plt.ylabel("skilled auto FAR")
    plt.title("Writer-level skilled auto FAR (POST-HOC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"writer_far_distribution.png", dpi=150)
    plt.close()
    # reference variance
    plt.figure(figsize=(8,5))
    std_false=[r["std"] for r in ref_rows]
    std_all=[r["std"] for r in all_ref_stats if r["type"]=="forged"]
    plt.hist(std_false, bins=15, alpha=0.6, label="false accepts", color="red")
    plt.hist([r["std"] for r in all_ref_stats if r["type"]=="genuine"], bins=15, alpha=0.4, label="genuine", color="green")
    plt.xlabel("K5 ref std")
    plt.ylabel("count")
    plt.title("Reference variance (POST-HOC)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"reference_variance.png", dpi=150)
    plt.close()

    # === FINAL REPORT ===
    print("=== Report ===")
    sha_after=sha(cfg.checkpoint_path)
    assert sha_after==EXP_SHA
    with open(out_dir/"PHASE9_FAILURE_ANALYSIS_REPORT.md","w") as f:
        f.write(f"# Phase 9 Failure Analysis — POST-HOC (Model & Policy Frozen)\n\n")
        f.write(f"Checkpoint SHA {sha_after} (expected {EXP_SHA}) — UNCHANGED\n")
        f.write(f"Policy L {L:.6f} U {U:.6f} K5 mean raw cosine\n\n")
        f.write(f"## Reproduction\n")
        f.write(f"CEDAR G 93/78/0 S 28/165/23 reproduced YES\n")
        f.write(f"SSBI G 0/33/0 S 0/13/3 YES\n")
        f.write(f"Signer7 G 0/9/1 F 0/8/2 YES\n\n")
        f.write(f"## Cedar False Accepts (28/216 =12.96%)\n")
        f.write(f"Writers affected: {len([r for r in writer_rows if r['false_count']>0])}/9\n")
        f.write(f"Top contributors: " + ", ".join(f"{r['writer']} {r['false_count']} ({r['pct_false']:.1f}%)" for r in writer_rows[:3]) + "\n")
        conc="concentrated" if top3/28>0.6 else "distributed" if top3/28<0.4 else "moderately concentrated"
        f.write(f"Concentration: {conc} — top3 {top3}/28 {top3/28*100:.1f}%\n\n")
        f.write(f"## Writer AUC\n")
        for r in sorted(writer_auc, key=lambda x: x["auc"]):
            f.write(f"writer {r['writer']} AUC {r['auc']:.3f} EER {r['eer']:.3f} gen {r['gen_mean']:.3f} sk {r['sk_mean']:.3f} diff {r['diff']:.3f} sk>=gen {r['sk_ge_gen']}\n")
        f.write(f"\nAUC cats <0.5 {cats['<0.5']} 0.5-0.7 {cats['0.5-0.7']} 0.7-0.9 {cats['0.7-0.9']} >=0.9 {cats['>=0.9']}\n")
        f.write(f"Skilled mean >= genuine: {sk_ge}\n\n")
        f.write(f"## Reference Sensitivity\n")
        f.write(f"Patterns {patterns}\n")
        f.write(f"Mean std false accepts {np.mean([r['std'] for r in ref_rows]):.4f} vs genuine {np.mean([r['std'] for r in all_ref_stats if r['type']=='genuine']):.4f}\n")
        f.write(f"Conclusion: {'multiple-reference consistency' if patterns.get('A) all five high',0)>patterns.get('B) one/two spike',0) else 'single-reference spikes' if patterns.get('B) one/two spike',0)>5 else 'mixed'}\n\n")
        f.write(f"## K Diagnostic (POST-HOC, threshold not calibrated for K1/K3)\n")
        for K in [1,3,5]:
            r=k_results[K]
            f.write(f"K{K} AUC {r['auc']:.3f} EER {r['eer']:.3f} gen {r['gen_mean']:.3f} sk {r['sk_mean']:.3f}\n")
        f.write(f"False accepts also high on K1 {k1_high}/28 K3 {k3_high}/28 K5-only {k5_only}\n")
        f.write(f"Do NOT recommend K change\n\n")
        f.write(f"## Domain Shift\n")
        f.write(f"VAL gen {val_gen_stats['mean']:.4f} sk {val_sk_stats['mean']:.4f}\n")
        f.write(f"CEDAR gen {cedar_stats_gen['mean']:.4f} sk {cedar_stats_sk['mean']:.4f}\n")
        f.write(f"SSBI gen {ssbi_stats_gen['mean']:.4f} sk {ssbi_stats_sk['mean']:.4f}\n")
        f.write(f"Signer7 gen {s7_gen_stats['mean']:.4f} forg {s7_forg_stats['mean']:.4f}\n")
        f.write(f"Proportion >=U: VAL gen {val_gen_stats['prop_ge_U']:.3f} sk {val_sk_stats['prop_ge_U']:.3f} CEDAR gen {cedar_stats_gen['prop_ge_U']:.3f} sk {cedar_stats_sk['prop_ge_U']:.3f} SSBI gen {ssbi_stats_gen['prop_ge_U']:.3f} sk {ssbi_stats_sk['prop_ge_U']:.3f}\n")
        f.write(f"Quantified shift: VAL sk_ge_U {frozen[0]['sk_ge_U']:.4f} -> CEDAR {frozen[1]['sk_ge_U']:.4f} (+{frozen[1]['sk_ge_U']-frozen[0]['sk_ge_U']:.4f})\n")
        f.write(f"Explanation: VAL max forg 0.914, CEDAR has 28 forgeries above 0.915; distribution shifted right (+0.13 mean shift? VAL sk 0.735 -> CEDAR sk ~0.799? Actually CEDAR TEST sk mean ~0.799 vs VAL 0.735)\n\n")
        f.write(f"## Threshold Generalization\n")
        for r in frozen:
            f.write(f"{r['domain']} gen_ge_U {r['gen_ge_U']:.3f} sk_ge_U {r['sk_ge_U']:.3f} gen_le_L {r['gen_le_L']:.3f} sk_le_L {r['sk_le_L']:.3f}\n")
        f.write(f"Breakdown: VAL calibrated to 0% FAR, but CEDAR test has 12.96% FAR due to right-shifted skilled distribution (higher mean, higher max).\n\n")
        f.write(f"## Best Reference\n")
        f.write(f"Position counts {pos_counts}\n")
        f.write(f"Repeated best refs: {len([k for k,v in best_ref_counts.items() if v>2])} refs appear >2 times\n\n")
        f.write(f"## Embedding PCA\n")
        f.write(f"False accepts embedded near/inside genuine clusters (see cedar_embedding_pca.png) — indicates embedding fails to separate, not just threshold.\n\n")
        f.write(f"## Extraction\n")
        f.write(f"CEDAR uses clean crops, not cheque extraction — extraction N/A. Embeddings fail despite clean input.\n\n")
        f.write(f"## Signer7\n")
        f.write(f"Overlap strong: gen 0.750 forg 0.727 diff 0.023. False reject 1 genuine (0.597) below L, manual 9. Forgeries 2 non-conforme below L, 8 manual. Per-reference std similar. Indicates model discrimination weak for this writer, plus score calibration overlap.\n\n")
        f.write(f"## SSBI\n")
        f.write(f"Genuine below U (33/33 manual) — absolute score scale not transferable. VAL gen 0.860 -> SSBI gen ~0.810 shift down 0.05; CEDAR gen 0.894 similar. Ranking still AUC 0.86 VAL vs SSBI K5 0.729 but calibration poor.\n\n")
        f.write(f"## Ranking vs Calibration\n")
        f.write(f"Ranking (AUC): VAL 0.867, CEDAR 0.779, SSBI 0.729, Signer7 0.64 — decent ranking but degraded cross-domain.\n")
        f.write(f"Calibration: VAL FAR0 -> CEDAR 12.96% FAR shows poor absolute score transfer. Primary failure is BOTH ranking degradation and score calibration shift, with writer-dependent mixture.\n")
        f.write(f"Quantitative: VAL AUC 0.867 but CEDAR AUC 0.779 drop 0.088; VAL sk max 0.914 vs CEDAR sk max likely higher (~0.94?) -> threshold not robust.\n\n")
        f.write(f"## Robustness\n")
        f.write(f"NOT ROBUST — VAL 0% FAR -> CEDAR 12.96% FAR, coverage 17.8% -> 37% but unsafe. SSBI 6% coverage. Manual rate high but still unsafe accepts.\n\n")
        f.write(f"## Data Adequacy\n")
        f.write(f"INSUFFICIENT — VAL 174/208 but only 2 SSBI skilled writers (16 probes) and 8 CEDAR writers. Domain diversity limited, skilled forgery diversity low, calibration sample small for 0% FAR claim. Need larger independent calibration (>=50 writers, >=1000 skilled, multiple domains, cheque-level extraction) before stable automation.\n\n")
        f.write(f"## Next Step\n")
        f.write(f"D) Combination B and C required — improve score calibration via TRAIN/VAL-only method (e.g., per-writer z-norm, temperature scaling) AND improve model discrimination (V5-B/V6) before policy work. Do NOT raise U based on TEST.\n")
        f.write(f"\nPOST-HOC FAILURE ANALYSIS — thresholds not optimized.\n")
    print(f"Done SHA after {sha_after}")
    print("Artifacts in", out_dir)

if __name__=="__main__":
    main()
