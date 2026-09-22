"""Generate BLIND-TEST-003 to 007 - 5 independent blind 50/50 tests, no V5, no thresholds."""
import random
import hashlib
import zlib
from pathlib import Path
import cv2
import numpy as np

CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
BASE_OUT = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo")

def stable_seed(k: str) -> int:
    return zlib.crc32(k.encode("utf-8"))

def make_moroccan_demo_cheque(sig_gray: np.ndarray, rng: np.random.Generator, cheque_number: str):
    W, H = 1600, 700
    base = rng.normal(244.0, 2.5, (H, W)).astype(np.float32)
    for i in range(14):
        y0 = 30 + i * 48 + int(rng.integers(-6, 7))
        amp = rng.uniform(3.0, 9.0)
        phase = rng.uniform(0, 2 * np.pi)
        xs = np.arange(W)
        ys = (y0 + amp * np.sin(xs / rng.uniform(60, 140) + phase)).astype(int)
        ok = (ys >= 0) & (ys < H)
        base[ys[ok], xs[ok]] -= rng.uniform(6.0, 12.0)
    img = np.clip(base, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color, (18, 18), (W - 18, H - 18), (140, 140, 140), 1)
    cv2.rectangle(color, (20, 20), (W - 20, H - 20), (175, 175, 175), 1)
    cv2.putText(color, "BANQUE DEMO TEST  -  SPECIMEN", (42, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (55, 55, 55), 1, cv2.LINE_AA)
    cv2.putText(color, "AGENCE CENTRALE DEMO  |  TEST ONLY - NOT A REAL BANK CHEQUE", (42, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (95, 95, 95), 1, cv2.LINE_AA)
    cv2.putText(color, f"CHEQUE N: {cheque_number}", (1120, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.line(color, (60, 70), (420, 70), (150, 150, 150), 1)
    cv2.putText(color, "Date: 03/09/2026", (60, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(color, "Casablanca", (285, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(color, "Payez contre ce cheque la somme de :", (42, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.line(color, (60, 300), (900, 300), (150, 150, 150), 1)
    cv2.putText(color, "Quarante mille dirhams", (70, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(color, "A  M. / Mme  TEST DEMO CLIENT", (70, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(color, (1050, 90), (1540, 170), (150, 150, 150), 1)
    for k in range(6):
        p1 = (1070 + k * 75, 155 - int(rng.integers(10, 55)))
        p2 = (1120 + k * 75, 105 + int(rng.integers(5, 45)))
        cv2.line(color, p1, p2, (120, 120, 120), 1)
    cv2.putText(color, "40 000,00 MAD", (1080, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(color, "Dirhams Marocains", (1085, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (105, 105, 105), 1, cv2.LINE_AA)
    cv2.putText(color, f"N CHEQUE {cheque_number}  |  COMPTE DEMO 011 123456789  |  40 000,00 MAD  |  03/09/2026", (48, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (95, 95, 95), 1, cv2.LINE_AA)
    overlay = color.copy()
    cv2.putText(overlay, "TEST  -  DEMO  -  NOT A REAL CHEQUE", (320, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (185, 185, 185), 2, cv2.LINE_AA)
    cv2.putText(overlay, "SPECIMEN  -  DOCUMENT DE TEST  -  NE PAS ENCAISSER", (380, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
    color = cv2.addWeighted(overlay, 0.55, color, 0.45, 0)
    cv2.rectangle(color, (640, 38), (880, 78), (160, 90, 90), 1)
    cv2.putText(color, "TEST / DEMO - SPECIMEN", (652, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(color, "--- CE CHEQUE EST UN SPECIMEN DE TEST - DOCUMENT SYNTHETIQUE SANS VALEUR BANCAIRE ---", (150, 682), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (110, 110, 110), 1, cv2.LINE_AA)
    micr_y = 655
    x = 120
    while x < W - 160:
        digit = str(int(rng.integers(0, 10)))
        cv2.putText(color, digit, (x, micr_y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.9, 1.3), (30, 30, 30), 2, cv2.LINE_AA)
        x += int(rng.integers(38, 58))
    cv2.line(color, (700, 595), (1480, 595), (155, 155, 155), 1)
    cv2.putText(color, "Signature du tireur  /  Signature du client", (860, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.putText(color, "(Ne pas deborder du cadre)", (970, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (125, 125, 125), 1, cv2.LINE_AA)
    region_x0, region_x1 = 660, 1520
    region_y0, region_y1 = 290, 590
    target_h = int(rng.integers(130, 195))
    scale = min(target_h / sig_gray.shape[0], (region_x1 - region_x0) / sig_gray.shape[1])
    new_w = max(1, int(round(sig_gray.shape[1] * scale)))
    new_h = max(1, int(round(sig_gray.shape[0] * scale)))
    resized = cv2.resize(sig_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized.astype(np.float32), (3, 3), 0)
    otsu_thr, _ = cv2.threshold(blurred.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    alpha = (blurred < otsu_thr).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    px = int(rng.integers(region_x0, max(region_x0 + 1, region_x1 - new_w)))
    py = int(rng.integers(region_y0, max(region_y0 + 1, region_y1 - new_h)))
    roi = color[py:py + new_h, px:px + new_w].astype(np.float32)
    ink_color = np.array([40.0, 25.0, 15.0])
    blended = roi * (1.0 - alpha[..., None]) + ink_color * alpha[..., None]
    color[py:py + new_h, px:px + new_w] = np.clip(blended, 0, 255).astype(np.uint8)
    paste_rect = (px, py, new_w, new_h)
    return color, paste_rect

import sys
import time
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi

def iou(a,b):
    ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union else 0
def bbox_of(r,roi=None):
    x,y=int(r.x),int(r.y)
    if roi: x+=int(roi.x); y+=int(roi.y)
    return (x,y,int(r.width),int(r.height))

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

# Prepare random selection for 5 tests
avoid_writers = {2,7,8,30,54}
available = [w for w in range(1,56) if w not in avoid_writers]
# Ensure distinct across 5 tests
selected_writers = random.sample(available, 5)

# For reproducibility, we will generate in order 003-007
tests = []
for idx, test_id in enumerate(["003","004","005","006","007"]):
    writer = selected_writers[idx]
    writer_dir = CEDAR_ROOT / str(writer)
    originals = sorted(writer_dir.glob("original_*.png"))
    forgeries = sorted(writer_dir.glob("forgeries_*.png"))
    refs = random.sample(originals, 5)
    is_genuine = random.random() < 0.5
    cand_type = "GENUINE" if is_genuine else "FORGED"
    if is_genuine:
        remaining = [p for p in originals if p not in refs]
        candidate = random.choice(remaining)
    else:
        candidate = random.choice(forgeries)
    assert candidate not in refs
    tests.append({
        "test_id": test_id,
        "writer": writer,
        "refs": refs,
        "candidate": candidate,
        "cand_type": cand_type,
        "is_genuine": is_genuine,
    })

# Now generate each cheque with extraction precheck
for t in tests:
    test_id = t["test_id"]
    writer = t["writer"]
    refs = t["refs"]
    candidate = t["candidate"]
    cand_type = t["cand_type"]
    cheque_number = f"BLIND-TEST-{test_id}"
    out_path = BASE_OUT / f"{cheque_number}.png"
    gt_path = BASE_OUT / f"{cheque_number}-ground-truth.txt"
    # Ground truth
    cand_hash = sha256(candidate)
    ref_hashes = [(p, sha256(p)) for p in refs]
    selection_time = time.strftime("%Y-%m-%d %H:%M:%S")
    # Need to capture random details without revealing? We'll store but not print candidate type in console
    # For file, we store full details
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write(f"BLIND TEST {test_id} GROUND TRUTH - DO NOT SHARE BEFORE MANUAL VERIFICATION\n")
        f.write(f"Cheque: {cheque_number}.png\n")
        f.write(f"Writer ID: {writer}\n")
        f.write(f"Candidate source filename: {candidate.name}\n")
        f.write(f"Candidate full path: {candidate}\n")
        f.write(f"Candidate type: {cand_type}\n")
        f.write(f"Candidate SHA256: {cand_hash}\n")
        f.write("References (5 genuine):\n")
        for p, h in ref_hashes:
            f.write(f"  {p.name} SHA256 {h}\n")
        f.write(f"Random selection details: writer pool 1-55 excluding [2,7,8,30,54] -> {len(available)} writers, distinct sample {selected_writers}; refs random.sample 5; candidate 50/50 {cand_type} via random.random; no V5, no thresholds\n")
        f.write(f"Selection time: {selection_time}\n")
        f.write("AI used for candidate selection: NO\n")
        f.write("Score-based selection: NO\n")
        f.write("Thresholds used: NO (L=0.6585 U=0.9150 not used)\n")

    # Do not print writer/candidate to console for blind
    print(f"[BLIND {test_id}] Writer hidden, candidate hidden, refs {len(refs)} - ground truth saved")

    sig_gray = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    assert sig_gray is not None

    best = None
    found = None
    for attempt in range(100):
        key = f"{cheque_number}-attempt-{attempt}" if attempt>0 else cheque_number
        rng = np.random.default_rng(stable_seed(key))
        img, paste = make_moroccan_demo_cheque(sig_gray, rng, cheque_number)
        try:
            res = extract_signature(img,0.40,0.40,0.98,0.98)
        except Exception:
            continue
        full = bbox_of(res.signature_bbox, res.candidate_roi)
        i = iou(full, paste)
        roi = _compute_roi(img.shape[1], img.shape[0],0.40,0.40,0.98,0.98)
        roi_crop = img[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray,(3,3),0)
        an = _analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
        ok = i>0.30 and an.completeness_score>=0.99 and res.extraction_quality>0.60 and len(an.discarded_from_selected_group_indices)==0 and len(an.micr_rejected_components)==0
        # contamination check also for cheque text: we consider groups not too many but allow
        score = i*2 + res.extraction_quality*0.5
        if an.completeness_score<0.99:
            score -= 1
        if i<0.30:
            score -= 1
        if best is None or (ok and score>best[0]) or (not best[1] and ok):
            is_ok = ok
            prev_ok = best[1] if best else False
            if is_ok and not prev_ok:
                best = (score, is_ok, attempt, paste, i, res.extraction_quality, an.completeness_score, img, res, an, key)
                print(f"  NEW BEST ok {test_id} attempt {attempt} iou {i:.3f} q {res.extraction_quality:.3f} comp {an.completeness_score:.3f}")
            elif is_ok==prev_ok and score>(best[0] if best else -1):
                best = (score, is_ok, attempt, paste, i, res.extraction_quality, an.completeness_score, img, res, an, key)
                print(f"  UPDATE BEST {test_id} attempt {attempt} iou {i:.3f} q {res.extraction_quality:.3f}")
        if best and best[1] and best[4]>0.35 and best[5]>0.70 and best[6]==1.0 and attempt>=30:
            if attempt>=35:
                break

    if best is None or not best[1]:
        print(f"[WARN] {test_id} No strict ok, using best available")
        if best is None:
            raise RuntimeError(f"No extraction for {test_id}")
    # Save
    _, is_ok, attempt, paste, i, qual, comp, img, res, an, key = best
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"[SAVE] {test_id} {out_path} paste {paste} seed {key} IoU {i:.4f} q {qual:.4f} comp {comp}")
    # extracted
    full = bbox_of(res.signature_bbox, res.candidate_roi)
    roi = _compute_roi(img.shape[1], img.shape[0],0.40,0.40,0.98,0.98)
    out_crop = BASE_OUT / f"{cheque_number}_extracted.png"
    out_crop.write_bytes(res.signature_png_bytes)
    # diagnostic
    vis = img.copy()
    cv2.rectangle(vis, (paste[0], paste[1]), (paste[0]+paste[2], paste[1]+paste[3]), (0,255,0), 2)
    cv2.putText(vis, "paste", (paste[0], paste[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (full[0], full[1]), (full[0]+full[2], full[1]+full[3]), (0,0,255), 2)
    cv2.putText(vis, f"extracted IoU={i:.2f}", (full[0], full[1]+full[3]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (roi.x, roi.y), (roi.x+roi.width, roi.y+roi.height), (255,0,0), 1)
    diag = BASE_OUT / f"{cheque_number}_diagnostic.png"
    cv2.imwrite(str(diag), vis)
    # manifest for provisioning (refs only, not candidate type for final report but needed for DB)
    manifest = BASE_OUT / f"{cheque_number}-manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"writer={writer}\n")
        for p, h in ref_hashes:
            f.write(f"ref={p.name}\n")
        f.write(f"candidate={candidate.name}\n")
        f.write(f"candidate_type={cand_type}\n")
    print(f"[DONE] {test_id} ready")

print("ALL FIVE BLIND TESTS GENERATED - do not reveal ground truth")
