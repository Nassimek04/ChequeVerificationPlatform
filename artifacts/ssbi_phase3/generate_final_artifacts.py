import pathlib, csv, pyodbc, requests, json, os, statistics, math
from pathlib import Path

base_pkg = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7")
web_root = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot")
api_base="http://localhost:8000"
conn_str="DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=ChequeVerificationDB;Trusted_Connection=yes;"
artifacts_dir = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase3")
csv_path = artifacts_dir / "ssbi_phase3_results.csv"
md_path = artifacts_dir / "ssbi_phase3_summary.md"

# Manifest
import csv as csvmod
manifest={}
with open(base_pkg/"manifest.csv", newline='', encoding='utf-8') as f:
    for row in csvmod.DictReader(f):
        manifest[row['file']]=row

def parse_bbox(s):
    inner=s.strip().strip('[]')
    parts=[int(p.strip()) for p in inner.split(',')]
    return (parts[0],parts[1],parts[2],parts[3])

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3])
    if x2<=x1 or y2<=y1: return 0.0
    inter=(x2-x1)*(y2-y1)
    union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union

def get_debug(file_path):
    with open(file_path,'rb') as f:
        r=requests.post(f"{api_base}/api/signatures/debug", files={"file": (Path(file_path).name, f, "image/png")}, timeout=30)
        if r.status_code!=200:
            return None
        j=r.json()
        return j

# Get all cheques for SSBI
conn=pyodbc.connect(conn_str)
cur=conn.cursor()
cur.execute("SELECT ChequeId, ChequeNumber FROM Cheque WHERE CustomerId=5 ORDER BY ChequeId")
cheques = cur.fetchall()  # list of (id, num)
# Map file to cheque? We need to know which file corresponds to which cheque.
# We don't have file name in Cheque table, but we can infer via order of creation:
# ChequeIds 9,30,31,... correspond to G01,G02,G03,G04,...G10,F01..F10 in order of campaign.
# Let's define order list
order_files = []
# G01 already
order_files.append(("genuine_cheques/S7_GENUINE_01.png", "GENUINE"))
for i in range(2,11):
    order_files.append((f"genuine_cheques/S7_GENUINE_{i:02d}.png", "GENUINE"))
for i in range(1,11):
    order_files.append((f"forged_cheques/S7_FORGED_{i:02d}.png", "FORGED"))
# cheques list should be same length 20 in order
print(f"Cheques {len(cheques)} order files {len(order_files)}")
for idx, (cheque_id, cheque_num) in enumerate(cheques):
    print(f"{cheque_id} {cheque_num} -> {order_files[idx][0]}")

# For each cheque, get extraction, VR, scores, debug
# We will regenerate CSV from scratch
header=['Sample','GroundTruth','SourceSignature','ChequeId','ChequeNumber','VerificationId','ExtractionSuccess','ExtractionClassification','ExtractionQuality','ExtractionCompleteness','Ref4Score','Ref5Score','Ref6Score','Ref7Score','Ref8Score','ComparedReferenceCount','MeanRawScore','BestReferenceId','BestReferenceScore','AutomaticDecision','Persisted','Notes','GlobalBbox','ManifestBbox','IoU']
rows=[]
for idx, (cheque_id, cheque_num) in enumerate(cheques):
    file_key, gt = order_files[idx]
    fname = Path(file_key).name
    subfolder = file_key.split('/')[0]
    file_path = base_pkg / file_key
    mrow = manifest[file_key]
    manifest_bbox = parse_bbox(mrow['placed_signature_bbox'])
    # Debug
    dbg = get_debug(file_path)
    if dbg and dbg.get('signature_bbox'):
        roi = dbg['candidate_roi']
        bbox = dbg['signature_bbox']
        qual = dbg['extraction_quality']
        comp = dbg['completeness_score']
        global_bbox = (roi['x']+bbox['x'], roi['y']+bbox['y'], bbox['width'], bbox['height'])
        iou_val = iou(global_bbox, manifest_bbox)
        if iou_val>=0.5 and qual>=0.5 and comp>=0.8:
            classification="CLEAN"
        elif iou_val>=0.30 and qual>=0.3:
            classification="ACCEPTABLE"
        elif bbox:
            classification="CONTAMINATED"
        else:
            classification="FAILED"
        extraction_success=True
        gb_str = f"{global_bbox[0]},{global_bbox[1]},{global_bbox[2]},{global_bbox[3]}"
        mb_str = f"{manifest_bbox[0]},{manifest_bbox[1]},{manifest_bbox[2]},{manifest_bbox[3]}"
        iou_str = f"{iou_val:.4f}"
        qual_str = f"{qual:.4f}"
        comp_str = f"{comp:.4f}"
    else:
        classification="FAILED"
        extraction_success=False
        qual_str=""; comp_str=""; gb_str=""; mb_str=f"{manifest_bbox[0]},{manifest_bbox[1]},{manifest_bbox[2]},{manifest_bbox[3]}"; iou_str=""
        qual=None; comp=None; global_bbox=None; iou_val=None

    # Get VR and scores from DB
    cur2=conn.cursor()
    cur2.execute("SELECT VerificationId, SimilarityScore, AutomaticDecision FROM VerificationResult WHERE ChequeId=?", (cheque_id,))
    vr_row = cur2.fetchone()
    if vr_row:
        vr_id, sim, auto_dec = vr_row
        persisted=True
        # Get per-ref scores
        cur2.execute("SELECT ReferenceSignatureId, SimilarityScore, IsBestMatch FROM SignatureComparison WHERE VerificationId=? ORDER BY ReferenceSignatureId", (vr_id,))
        comps = cur2.fetchall()
        ref_dict={r[0]: r[1] for r in comps}
        best_id = [r[0] for r in comps if r[2]==1][0] if any(r[2]==1 for r in comps) else None
        best_score = ref_dict.get(best_id) if best_id else None
        # For CSV, need all 5 ref scores
        r4=ref_dict.get(4, ""); r5=ref_dict.get(5, ""); r6=ref_dict.get(6, ""); r7=ref_dict.get(7, ""); r8=ref_dict.get(8, "")
        compared=5
        mean_raw = float(sim)  # persisted mean (rounded)
        # But for raw mean, we should use the actual mean before rounding? For G01, persisted 0.1844 vs raw 0.184384, but we will use persisted for CSV's MeanRawScore? Use raw from DB? We'll use persisted
        # For more precise, we could recompute via API, but use DB
        auto_map={1:'Conforme',2:'Non conforme',3:'Contrôle manuel'}
        auto_str=auto_map[auto_dec]
        notes=""
        # For G03 which has no VR, we need to handle separately
    else:
        # No VR - need to get AI scores via API (like retry did) - for G03
        # Forcheques without VR, we can call API to get scores
        # Get extracted crop path
        cur2.execute("SELECT ImagePath FROM ExtractedSignature WHERE ChequeId=?", (cheque_id,))
        ext_row=cur2.fetchone()
        if ext_row:
            rel_crop=ext_row[0]
            crop_path = web_root / rel_crop.lstrip('/').replace('/', os.sep)
            # Get refs
            cur2.execute("SELECT ReferenceSignatureId, ImagePath FROM ReferenceSignature WHERE CustomerId=5 AND IsActive=1 ORDER BY ReferenceSignatureId")
            refs = cur2.fetchall()
            ref_scores={}
            for ref_id, rel_ref in refs:
                ref_path = web_root / rel_ref.lstrip('/').replace('/', os.sep)
                with open(crop_path,'rb') as f1, open(ref_path,'rb') as f2:
                    resp = requests.post(f"{api_base}/api/signatures/compare-ai", files={"extracted_file": (crop_path.name, f1, "image/png"), "reference_file": (ref_path.name, f2, "image/png")}, timeout=30)
                    if resp.status_code==200:
                        j=resp.json()
                        if j.get('success'):
                            ref_scores[ref_id]=j['similarity_score']
            if ref_scores:
                r4=ref_scores.get(4, ""); r5=ref_scores.get(5, ""); r6=ref_scores.get(6, ""); r7=ref_scores.get(7, ""); r8=ref_scores.get(8, "")
                compared=len(ref_scores)
                mean_raw=sum(ref_scores.values())/len(ref_scores)
                best_id = max(ref_scores, key=lambda k: ref_scores[k])
                max_sc=max(ref_scores.values())
                cands=[k for k,v in ref_scores.items() if abs(v-max_sc)<1e-9]
                best_id=min(cands)
                best_score=ref_scores[best_id]
                if mean_raw <=0.0895:
                    auto_str="Non conforme"
                elif mean_raw>=0.6898:
                    auto_str="Conforme"
                else:
                    auto_str="Contrôle manuel"
                vr_id=""; persisted=False; notes="CHECK constraint violation (negative per-ref score) - not persisted"
                # Use mean_raw for CSV
                sim=mean_raw
            else:
                r4=r5=r6=r7=r8=""; compared=0; mean_raw=None; best_id=""; best_score=""; auto_str="N/A"; vr_id=""; persisted=False; notes="No AI scores"
                sim=None
        else:
            r4=r5=r6=r7=r8=""; compared=0; mean_raw=None; best_id=""; best_score=""; auto_str="N/A"; vr_id=""; persisted=False; notes="No extracted signature"
            sim=None
        # For CSV, use mean_raw
    # Prepare row
    # Need to ensure mean_raw formatting
    def fmt(v):
        if v=="" or v is None:
            return ""
        try:
            return f"{float(v):.6f}"
        except:
            return str(v)
    row = [fname, gt, f"{mrow['source_signature_sheet']}#{mrow['source_annotation_id']}", cheque_id, cheque_num, vr_id if vr_row else "", str(extraction_success).lower(), classification, qual_str, comp_str, fmt(r4), fmt(r5), fmt(r6), fmt(r7), fmt(r8), compared, fmt(mean_raw) if 'mean_raw' in locals() and mean_raw is not None else (fmt(sim) if 'sim' in locals() and sim is not None else ""), best_id if 'best_id' in locals() else "", fmt(best_score) if 'best_score' in locals() else "", auto_str if 'auto_str' in locals() else "", str(persisted).lower(), notes, gb_str, mb_str, iou_str]
    rows.append(row)
    print(f"Row {fname} {gt} mean {row[16]} auto {row[19]} persisted {row[20]}")

# Write CSV
with open(csv_path,'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
print(f"CSV written to {csv_path} with {len(rows)} rows")

# Now compute stats
import statistics

genuine_rows = [r for r in rows if r[1]=="GENUINE"]
forged_rows = [r for r in rows if r[1]=="FORGED"]

def get_scores(rs):
    vals=[]
    for r in rs:
        try:
            vals.append(float(r[16]))
        except:
            pass
    return vals

genuine_scores = get_scores(genuine_rows)
forged_scores = get_scores(forged_rows)
print(f"Genuine scores: {genuine_scores}")
print(f"Forged scores: {forged_scores}")

def percentile(data, p):
    if not data: return float('nan')
    s=sorted(data)
    k=(len(s)-1)*p/100
    f=math.floor(k); c=math.ceil(k)
    if f==c: return s[int(k)]
    d0=s[int(f)]*(c-k); d1=s[int(c)]*(k-f)
    return d0+d1

def stats(data):
    if not data: return {}
    return {
        "N": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "std": statistics.pstdev(data) if len(data)>1 else 0,
        "min": min(data),
        "max": max(data),
        "Q1": percentile(data,25),
        "Q3": percentile(data,75)
    }

g_stats=stats(genuine_scores)
f_stats=stats(forged_scores)
print(f"Genuine stats {g_stats}")
print(f"Forged stats {f_stats}")

# Pairwise
total_pairs=len(genuine_scores)*len(forged_scores)
gt=0; ties=0
for g in genuine_scores:
    for f in forged_scores:
        if g>f: gt+=1
        elif abs(g-f)<1e-9: ties+=1
auc = (gt + 0.5*ties)/total_pairs if total_pairs else float('nan')
print(f"AUC {auc:.4f} G>F {gt}/{total_pairs} ties {ties}")

# Overlap
max_forged = max(forged_scores) if forged_scores else float('nan')
min_genuine = min(genuine_scores) if genuine_scores else float('nan')
print(f"max forged {max_forged} min genuine {min_genuine} overlap {min_genuine>max_forged}")

# Policy counts
def count_decision(rows, dec):
    return sum(1 for r in rows if r[19]==dec)
g_conforme=count_decision(genuine_rows,"Conforme")
g_manual=count_decision(genuine_rows,"Contrôle manuel")
g_non=count_decision(genuine_rows,"Non conforme")
f_conforme=count_decision(forged_rows,"Conforme")
f_manual=count_decision(forged_rows,"Contrôle manuel")
f_non=count_decision(forged_rows,"Non conforme")
print(f"Genuine decisions: Conforme {g_conforme} Manual {g_manual} Non {g_non}")
print(f"Forged decisions: Conforme {f_conforme} Manual {f_manual} Non {f_non}")

# Extraction
success = sum(1 for r in rows if r[6]=="true")
clean = sum(1 for r in rows if r[7]=="CLEAN")
acceptable = sum(1 for r in rows if r[7]=="ACCEPTABLE")
contaminated = sum(1 for r in rows if r[7]=="CONTAMINATED")
failed = sum(1 for r in rows if r[7]=="FAILED")
print(f"Extraction success {success} clean {clean} acceptable {acceptable} contaminated {contaminated} failed {failed}")
# Qualities
g_quals=[float(r[8]) for r in genuine_rows if r[8]!=""]
f_quals=[float(r[8]) for r in forged_rows if r[8]!=""]
print(f"Mean qual genuine {statistics.mean(g_quals):.4f} forged {statistics.mean(f_quals):.4f} overall {statistics.mean(g_quals+f_quals):.4f}")
# IoU
ious=[float(r[24]) for r in rows if r[24]!=""]
print(f"IoU mean {statistics.mean(ious):.4f} median {statistics.median(ious):.4f} >=0.30 {sum(1 for v in ious if v>=0.30)} >=0.50 {sum(1 for v in ious if v>=0.50)}")

# Write markdown
with open(md_path,'w',encoding='utf-8') as f:
    f.write(f"# SSBI Phase 3 Summary\n\n")
    f.write(f"Generated: {__import__('datetime').datetime.utcnow().isoformat()} UTC\n\n")
    f.write(f"## Samples\n- Genuine 10\n- Forged 10\n- Total 20\n\n")
    f.write(f"## Extraction\n- Success {success}/20\n- Clean {clean} Acceptable {acceptable} Contaminated {contaminated} Failed {failed}\n")
    f.write(f"- Mean quality genuine {statistics.mean(g_quals):.4f} forged {statistics.mean(f_quals):.4f} overall {statistics.mean(g_quals+f_quals):.4f}\n")
    f.write(f"- Mean IoU {statistics.mean(ious):.4f} median {statistics.median(ious):.4f} >=0.30 {sum(1 for v in ious if v>=0.30)} >=0.50 {sum(1 for v in ious if v>=0.50)}\n\n")
    f.write(f"## Genuine Scores\n- [{', '.join(f'{v:.4f}' for v in genuine_scores)}]\n")
    f.write(f"- Mean {g_stats['mean']:.4f} Median {g_stats['median']:.4f} Std {g_stats['std']:.4f} Min {g_stats['min']:.4f} Max {g_stats['max']:.4f} Q1 {g_stats['Q1']:.4f} Q3 {g_stats['Q3']:.4f}\n\n")
    f.write(f"## Forged Scores\n- [{', '.join(f'{v:.4f}' for v in forged_scores)}]\n")
    f.write(f"- Mean {f_stats['mean']:.4f} Median {f_stats['median']:.4f} Std {f_stats['std']:.4f} Min {f_stats['min']:.4f} Max {f_stats['max']:.4f} Q1 {f_stats['Q1']:.4f} Q3 {f_stats['Q3']:.4f}\n\n")
    f.write(f"## Separation\n- Mean diff {g_stats['mean']-f_stats['mean']:.4f} Median diff {g_stats['median']-f_stats['median']:.4f}\n")
    f.write(f"- ROC AUC {auc:.4f} Pairwise G>F {gt}/{total_pairs} ties {ties}\n")
    f.write(f"- Lowest genuine {min_genuine:.4f} Highest forged {max_forged:.4f} Overlap {'YES' if min_genuine<=max_forged else 'NO'}\n\n")
    f.write(f"## Frozen Policy L=0.0895 U=0.6898\n- Genuine: Conforme {g_conforme} Contrôle manuel {g_manual} Non conforme {g_non}\n")
    f.write(f"- Forged: Conforme {f_conforme} Contrôle manuel {f_manual} Non conforme {f_non}\n")
    if f_conforme>0:
        forged_conforme_samples=[r[0] for r in forged_rows if r[19]=="Conforme"]
        f.write(f"- CRITICAL Forged auto-accepted: {f_conforme} ({', '.join(forged_conforme_samples)})\n")
    if g_non>0:
        f.write(f"- Genuine auto-rejected: {g_non} ({', '.join([r[0] for r in genuine_rows if r[19]=='Non conforme'])})\n")
    f.write(f"\n## Database\n")
    cur.execute("SELECT COUNT(*) FROM Cheque WHERE CustomerId=5")
    cnt_cheque=cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM VerificationResult WHERE ChequeId IN (SELECT ChequeId FROM Cheque WHERE CustomerId=5)")
    cnt_vr=cur.fetchone()[0]
    f.write(f"- Cheques for SSBI: {cnt_cheque}\n- VR: {cnt_vr}\n- Note: Genuine03 (S7_GENUINE_03.png) failed to persist due to CHECK constraint negative per-ref score (-0.012173) - not counted in VR\n")
    f.write(f"\n## Notes\n- Forged 09/10 reuse 2 forgery samples on different templates per README (8 unique forgeries, S7_FORGED_09 reuses F1#33 on check_002, S7_FORGED_10 reuses F1#34 on check_003)\n")
    f.write(f"- All samples processed via ChequeService, VerificationService, AI V2 mean raw cosine, no threshold change\n")
    f.write(f"- Extraction V2.4 remained frozen; one genuine sample (S7_GENUINE_03) had negative per-ref score causing DB CHECK violation (0<=score<=1) - handled as not persisted but included in distribution\n")

print(f"Markdown written to {md_path}")
# Plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Distribution
    plt.figure(figsize=(8,5))
    plt.hist(genuine_scores, bins=10, alpha=0.6, label='Genuine')
    plt.hist(forged_scores, bins=10, alpha=0.6, label='Forged')
    plt.axvline(0.0895, color='red', linestyle='--', label='L=0.0895')
    plt.axvline(0.6898, color='green', linestyle='--', label='U=0.6898')
    plt.xlabel('MeanRawScore')
    plt.ylabel('Count')
    plt.title('SSBI Phase3 Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "ssbi_score_distribution.png"))
    print("saved distribution")
    # ROC
    from sklearn.metrics import roc_curve, auc
    import pandas as pd
    df = __import__('pandas').read_csv(csv_path)
    y_true = (df.GroundTruth=='GENUINE').astype(int)
    y_score = df.MeanRawScore
    # Need to handle NaN? All have scores
    fpr,tpr,_=roc_curve(y_true, y_score)
    roc_auc=auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('SSBI Phase3 ROC (Genuine positive)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(artifacts_dir / "ssbi_roc_curve.png"))
    print(f"ROC AUC {roc_auc:.4f}")
except Exception as e:
    print(f"Plot failed {e}")
    import traceback; traceback.print_exc()

conn.close()
print("=== GENERATE DONE ===")
