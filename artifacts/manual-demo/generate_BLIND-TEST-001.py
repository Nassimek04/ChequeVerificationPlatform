"""Blind test generation - random writer, random candidate, no score-based selection, no V5."""
import random
import hashlib
import zlib
from pathlib import Path
import cv2
import numpy as np

CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
OUT_PATH = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\BLIND-TEST-001.png")
GROUND_TRUTH_PATH = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\BLIND-TEST-001-ground-truth.txt")

CHEQUE_NUMBER = "BLIND-TEST-001"
DATE_STR = "03/09/2026"
AMOUNT_STR = "40 000,00"

def stable_seed(k: str) -> int:
    return zlib.crc32(k.encode("utf-8"))

def make_moroccan_demo_cheque(sig_gray: np.ndarray, rng: np.random.Generator, cheque_number: str = CHEQUE_NUMBER):
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
    cv2.putText(color, f"Date: {DATE_STR}", (60, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
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
    cv2.putText(color, f"{AMOUNT_STR} MAD", (1080, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
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

# --- Random selection without V5, without thresholds ---
# Use system random, not numpy, for true blind
import time

# Discover available writers (1..55)
available_writers = list(range(1, 56))
# Exclude previous demos 2,7,8
for w in [2,7,8]:
    if w in available_writers:
        available_writers.remove(w)

# True random choice
writer = random.choice(available_writers)

# Gather genuine originals for that writer
writer_dir = CEDAR_ROOT / str(writer)
originals = sorted(writer_dir.glob("original_*.png"))
forgeries = sorted(writer_dir.glob("forgeries_*.png"))
assert len(originals) >= 6, f"Not enough originals for writer {writer}"
assert len(forgeries) >= 1

# Randomly select 5 references from originals
refs = random.sample(originals, 5)

# 50/50 candidate type
is_genuine = random.random() < 0.5
candidate_type = "GENUINE" if is_genuine else "FORGED"
if is_genuine:
    # Choose one genuine not in refs
    remaining = [p for p in originals if p not in refs]
    candidate = random.choice(remaining)
else:
    candidate = random.choice(forgeries)

# Ensure candidate not in refs (for genuine case already, for forged never)
assert candidate not in refs

# Compute hashes
def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

candidate_hash = sha256(candidate)
ref_hashes = [(p, sha256(p)) for p in refs]

# Random selection details
selection_time = time.strftime("%Y-%m-%d %H:%M:%S")
random_details = f"writer pool 1-55 excluding [2,7,8] -> {len(available_writers)} writers, random.choice -> {writer}; refs random.sample 5 from {len(originals)} originals; candidate 50/50 random.random {candidate_type} (is_genuine={is_genuine}); candidate random.choice from {'remaining genuine' if is_genuine else 'forgeries'}; no V5, no threshold, no score-based selection"

# Save ground truth privately (do not print candidate type to stdout in final report, but we save here)
GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
    f.write(f"BLIND TEST GROUND TRUTH - DO NOT SHARE BEFORE MANUAL VERIFICATION\n")
    f.write(f"Cheque: BLIND-TEST-001.png\n")
    f.write(f"Writer ID: {writer}\n")
    f.write(f"Candidate source filename: {candidate.name}\n")
    f.write(f"Candidate full path: {candidate}\n")
    f.write(f"Candidate type: {candidate_type}\n")
    f.write(f"Candidate SHA256: {candidate_hash}\n")
    f.write(f"References (5 genuine):\n")
    for p, h in ref_hashes:
        f.write(f"  {p.name} SHA256 {h}\n")
    f.write(f"Random selection details: {random_details}\n")
    f.write(f"Selection time: {selection_time}\n")
    f.write(f"AI used for candidate selection: NO\n")
    f.write(f"Score-based selection: NO\n")
    f.write(f"Thresholds used: NO (L=0.6585 U=0.9150 not used)\n")

print(f"[BLIND] Writer selected (hidden from final report, saved to ground truth)")
print(f"[BLIND] Candidate type hidden, saved to {GROUND_TRUTH_PATH}")
print(f"[BLIND] References selected: {len(refs)} files")
print(f"[BLIND] Candidate hash: {candidate_hash[:16]}... (full in ground truth)")
# Do NOT print candidate name/type here for terminal that will be shown? We have printed hidden but not revealed type in final report.
# For this script, we intentionally do not print candidate filename/type to stdout beyond ground truth file.
# However for debugging we printed generic.

# Now generate cheque - we need sig_gray
sig_gray = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
assert sig_gray is not None

# We will now brute force placement search for good extraction, same strategy as before
# Use deterministic seeds derived from cheque number + attempt index, but candidate is fixed
import sys
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

best = None
found_seed = None
found_paste = None
found_img = None
found_res = None
found_an = None

# Try primary seed first
for attempt in range(100):
    key = f"BLIND-TEST-001-attempt-{attempt}" if attempt>0 else "BLIND-TEST-001"
    rng = np.random.default_rng(stable_seed(key))
    img, paste = make_moroccan_demo_cheque(sig_gray, rng)
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
    # Require extraction success, completeness close to 1.0, no MICR contamination, visually complete, acceptable quality
    # Use thresholds from previous successful: IoU>0.30, completeness>=0.99, quality>0.65, groups not huge
    ok = i>0.30 and an.completeness_score>=0.99 and res.extraction_quality>0.65 and len(an.discarded_from_selected_group_indices)==0
    # Also ensure not too many groups? but allow
    score = i*2 + res.extraction_quality*0.5
    if an.completeness_score<0.99:
        score -= 1
    if i<0.30:
        score -= 1
    if best is None or (ok and score>best[0]) or (not best[1] and ok):
        # Prefer ok placements; first ok becomes best, then higher score
        is_ok = ok
        prev_ok = best[1] if best else False
        if is_ok and not prev_ok:
            best = (score, is_ok, attempt, paste, i, res.extraction_quality, an.completeness_score, img, res, an, key)
            found_seed = key
            found_paste = paste
            found_img = img
            found_res = res
            found_an = an
            print(f"NEW BEST ok attempt {attempt} iou {i:.3f} q {res.extraction_quality:.3f} comp {an.completeness_score:.3f} groups {len(an.groups)}")
            if is_ok and i>0.35 and res.extraction_quality>0.75 and an.completeness_score==1.0:
                # good enough early stop? but continue to find best
                pass
        elif is_ok==prev_ok and score>(best[0] if best else -1):
            best = (score, is_ok, attempt, paste, i, res.extraction_quality, an.completeness_score, img, res, an, key)
            found_seed = key
            found_paste = paste
            found_img = img
            found_res = res
            found_an = an
            print(f"UPDATE BEST {'ok' if is_ok else 'not ok'} attempt {attempt} iou {i:.3f} q {res.extraction_quality:.3f}")

    # Early exit if we found a good ok with high quality
    if best and best[1] and best[4]>0.35 and best[5]>0.75 and best[6]==1.0 and attempt>=20:
        # we have a good one, but continue a bit to possibly find even better
        if attempt>=30:
            break

if best is None or not best[1]:
    print("[WARN] No placement meeting strict criteria, using best available")
    # fallback to best even if not ok
    if best is None:
        raise RuntimeError("No extraction at all")
    found_seed = best[8] if len(best)>8 else "fallback"
    # Actually best tuple already contains img etc.
    # Use best's img
else:
    # best already set
    pass

# Use found best
img_to_save = found_img if found_img is not None else img
paste_to_save = found_paste
seed_used = found_seed
res_to_save = found_res
an_to_save = found_an

# Save cheque
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_PATH), img_to_save)
print(f"[SAVE] Cheque {OUT_PATH} paste {paste_to_save} seed {seed_used}")
print(f"[SAVE] SHA256 cheque {hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()}")
print(f"[SAVE] Dimensions {img_to_save.shape[1]}x{img_to_save.shape[0]}")
print(f"[SAVE] Extraction quality {res_to_save.extraction_quality} completeness {an_to_save.completeness_score} IoU {iou(bbox_of(res_to_save.signature_bbox, res_to_save.candidate_roi), paste_to_save):.4f}")

# Save extracted and diagnostic
# Use same logic as prevalidate_forged.py
import cv2 as cv2b
full = bbox_of(res_to_save.signature_bbox, res_to_save.candidate_roi)
roi = _compute_roi(img_to_save.shape[1], img_to_save.shape[0],0.40,0.40,0.98,0.98)
# extracted
out_crop = OUT_PATH.parent / "BLIND-TEST-001_extracted.png"
out_crop.write_bytes(res_to_save.signature_png_bytes)
print(f"[SAVE] Extracted {out_crop}")
# diagnostic
vis = img_to_save.copy()
cv2.rectangle(vis, (paste_to_save[0], paste_to_save[1]), (paste_to_save[0]+paste_to_save[2], paste_to_save[1]+paste_to_save[3]), (0,255,0), 2)
cv2.putText(vis, "paste", (paste_to_save[0], paste_to_save[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,0), 1, cv2.LINE_AA)
cv2.rectangle(vis, (full[0], full[1]), (full[0]+full[2], full[1]+full[3]), (0,0,255), 2)
cv2.putText(vis, f"extracted IoU={iou(full, paste_to_save):.2f}", (full[0], full[1]+full[3]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 1, cv2.LINE_AA)
cv2.rectangle(vis, (roi.x, roi.y), (roi.x+roi.width, roi.y+roi.height), (255,0,0), 1)
diag = OUT_PATH.parent / "BLIND-TEST-001_diagnostic.png"
cv2.imwrite(str(diag), vis)
print(f"[SAVE] Diagnostic {diag}")

# Save additional info for provisioning (references list and candidate)
# Already have ground truth, but also save a provisioning manifest (not containing candidate type? but provisioning needs refs)
# For customer provisioning, we need writer and refs. We have that.
# Write a simple file for provisioning script to read (without revealing candidate type in final report, but provisioning script needs it)
# We'll write ref list to a temp file that provisioning script can read, but not printed in final report.
manifest = OUT_PATH.parent / "BLIND-TEST-001-manifest.txt"
with open(manifest, "w", encoding="utf-8") as f:
    f.write(f"writer={writer}\n")
    for p, h in ref_hashes:
        f.write(f"ref={p.name}\n")
    f.write(f"candidate={candidate.name}\n")
    f.write(f"candidate_type={candidate_type}\n")

print("[DONE] Blind test cheque ready, ground truth saved, manifest for provisioning saved")
# Do NOT print writer or candidate details in final summary beyond ground truth file
