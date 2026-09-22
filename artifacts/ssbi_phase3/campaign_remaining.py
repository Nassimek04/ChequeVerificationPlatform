import pyodbc, requests, pathlib, uuid, hashlib, json, os, sys, datetime, csv, re, statistics, math
from pathlib import Path

base_pkg = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7")
web_root = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot")
api_base = "http://localhost:8000"
conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=ChequeVerificationDB;Trusted_Connection=yes;"

# manifest
manifest_path = base_pkg / "manifest.csv"
manifest = {}
import csv as csvmod
with open(manifest_path, newline='', encoding='utf-8') as f:
    reader = csvmod.DictReader(f)
    for row in reader:
        manifest[row['file']] = row

def parse_bbox(s):
    # "[1562, 478, 253, 101]"
    inner = s.strip().strip('[]')
    parts = [int(p.strip()) for p in inner.split(',')]
    return (parts[0], parts[1], parts[2], parts[3])  # x,y,w,h

def iou(a,b):
    # a,b tuples x,y,w,h
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2]); y2 = min(a[1]+a[3], b[1]+b[3])
    if x2<=x1 or y2<=y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union

def sha256_file(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compute_sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

# DB helpers
conn = pyodbc.connect(conn_str)
conn.autocommit = False
cur = conn.cursor()

def get_next_cheque_number():
    cur.execute("SELECT TOP 1 ChequeNumber FROM Cheque ORDER BY ChequeId DESC")
    row = cur.fetchone()
    if row and row[0]:
        last = row[0]
        num = int(last.split('-')[1])
        return f"CHQ-{num+1:04d}"
    return "CHQ-0001"

def get_admin_user():
    cur.execute("SELECT TOP 1 UserId, Email FROM [User] WHERE RoleId=1")
    r = cur.fetchone()
    return r[0], r[1]

admin_id, admin_email = get_admin_user()
print(f"Admin {admin_id} {admin_email}")

# Remaining samples (Genuine04-10 and Forged01-10) - G02/G03 already done
genuine_remaining = [f"S7_GENUINE_{i:02d}.png" for i in range(2,11)]
forged_all = [f"S7_FORGED_{i:02d}.png" for i in range(1,11)]
all_samples = [(f, "GENUINE", "genuine_cheques") for f in genuine_remaining] + [(f, "FORGED", "forged_cheques") for f in forged_all]
print(f"Processing {len(all_samples)} samples")

# For CSV we need to include already done G01,G02,G03
# We'll create a new CSV from scratch, including all 20
artifacts_dir = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase3")
artifacts_dir.mkdir(parents=True, exist_ok=True)
csv_path = artifacts_dir / "ssbi_phase3_results.csv"
# If exists, we will regenerate fully after campaign, so for now process remaining and append to existing partial
# Check existing CSV has G01
if csv_path.exists():
    print(f"Existing CSV exists, will append remaining")
    # Read existing to avoid duplicates
    existing_samples = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csvmod.DictReader(f):
            existing_samples.add(row['Sample'])
    print(f"Existing samples: {existing_samples}")
else:
    existing_samples = set()

# Helper to call debug for bbox/IoU
def get_debug_info(file_path):
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(f"{api_base}/api/signatures/debug", files={"file": (Path(file_path).name, f, "image/png")}, timeout=30)
        if resp.status_code != 200:
            print(f"  debug failed {resp.status_code} {resp.text[:200]}")
            return None
        j = resp.json()
        roi = j.get('candidate_roi')
        bbox = j.get('signature_bbox')
        qual = j.get('extraction_quality')
        comp = j.get('completeness_score')
        return {"roi": roi, "bbox": bbox, "quality": qual, "completeness": comp, "raw": j}
    except Exception as e:
        print(f"  debug exception {e}")
        return None

def get_extract_info(file_path):
    try:
        with open(file_path, 'rb') as f:
            resp = requests.post(f"{api_base}/api/signatures/extract", files={"file": (Path(file_path).name, f, "image/png")}, timeout=30)
        if resp.status_code != 200:
            print(f"  extract failed {resp.status_code}")
            return None
        j = resp.json()
        return j
    except Exception as e:
        print(f"  extract exception {e}")
        return None

# Process each remaining sample
for fname, gt, subfolder in all_samples:
    if fname in existing_samples:
        print(f"Skipping {fname} already in CSV")
        continue
    file_path = base_pkg / subfolder / fname
    manifest_key = f"{subfolder}/{fname}"
    mrow = manifest[manifest_key]
    manifest_bbox = parse_bbox(mrow['placed_signature_bbox'])
    print(f"\n--- {fname} {gt} manifest {manifest_bbox} ---")
    # Debug for extraction quality/bbox
    debug = get_debug_info(file_path)
    if debug:
        roi = debug['roi']
        bbox = debug['bbox']
        qual = debug['quality']
        comp = debug['completeness']
        if bbox:
            # bbox is ROI-local, roi is global
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
            print(f"  Debug: ROI {roi} bbox local {bbox} global {global_bbox} qual {qual} comp {comp} IoU {iou_val:.4f} class {classification}")
        else:
            classification="FAILED"
            global_bbox=None
            iou_val=None
            print(f"  Debug: no bbox qual {qual} comp {comp} class FAILED")
    else:
        qual=None; comp=None; classification="FAILED"; global_bbox=None; iou_val=None

    # Now create cheque via DB (mimic ChequeService)
    try:
        cheque_number = get_next_cheque_number()
        print(f"  Cheque number {cheque_number}")
        # Copy file to wwwroot/uploads/cheques/{guid}.png
        guid = uuid.uuid4().hex
        ext = ".png"
        rel_path = f"/uploads/cheques/{guid}{ext}"
        dest = web_root / f"uploads/cheques/{guid}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy bytes
        data = Path(file_path).read_bytes()
        dest.write_bytes(data)
        # Insert Cheque
        now = datetime.datetime.utcnow()
        # Need to get next ChequeId? Use OUTPUT
        cur.execute("INSERT INTO Cheque (Amount, ChequeNumber, CustomerId, ImagePath, ImportedByUserId, IssueDate, Status, UploadedAt) OUTPUT INSERTED.ChequeId VALUES (?,?,?,?,?,?,?,?)",
                    (1000.00, cheque_number, 5, rel_path, admin_id, datetime.date.today(), 1, now))
        cheque_id = cur.fetchone()[0]
        # Audit IMPORT_CHEQUE
        cur.execute("INSERT INTO AuditLog (Action, CreatedAt, Description, EntityId, EntityName, UserId) VALUES (?,?,?,?,?,?)",
                    ("IMPORT_CHEQUE", now, f"Import du chèque {cheque_number} pour le client n° 5", cheque_id, "Cheque", admin_id))
        conn.commit()
        print(f"  Created cheque {cheque_number} Id {cheque_id}")
    except Exception as e:
        print(f"  Cheque create failed {e}")
        conn.rollback()
        continue

    # Now extraction via service: call API extract and then persist
    # We will mimic VerificationService.ExtractAndPersistSignatureAsync
    try:
        # Call extract API to get crop
        ext_resp = get_extract_info(file_path)
        if not ext_resp or not ext_resp.get('success'):
            print(f"  Extract API failed")
            # Update cheque status? Keep as is
            # need to record extraction failure
            # For CSV, we will record with extractionSuccess false
            # But we still need to handle
            # Let's fetch quality from debug
            extraction_success=False
            extraction_quality=qual
            extraction_completeness=comp
        else:
            extraction_success=True
            extraction_quality = ext_resp.get('extraction_quality')
            # The API returns base64 crop
            b64 = ext_resp.get('signature_image_base64')
            if b64:
                crop_bytes = __import__('base64').b64decode(b64)
                # Save crop
                guid2 = uuid.uuid4().hex
                rel_crop = f"/uploads/signatures/extracted/{guid2}.png"
                dest_crop = web_root / f"uploads/signatures/extracted/{guid2}.png"
                dest_crop.parent.mkdir(parents=True, exist_ok=True)
                dest_crop.write_bytes(crop_bytes)
                crop_hash = compute_sha256_bytes(crop_bytes)
                # Insert ExtractedSignature
                now2 = datetime.datetime.utcnow()
                cur2 = conn.cursor()
                cur2.execute("INSERT INTO ExtractedSignature (ChequeId, ExtractedAt, ExtractionConfidence, FileHash, ImagePath) OUTPUT INSERTED.ExtractedSignatureId VALUES (?,?,?,?,?)",
                             (cheque_id, now2, extraction_quality, crop_hash, rel_crop))
                ext_id = cur2.fetchone()[0]
                # Update cheque status to 2
                cur2.execute("UPDATE Cheque SET Status=2 WHERE ChequeId=?", (cheque_id,))
                # Audit EXTRACT_SIGNATURE
                cur2.execute("INSERT INTO AuditLog (Action, CreatedAt, Description, EntityId, EntityName, UserId) VALUES (?,?,?,?,?,?)",
                             ("EXTRACT_SIGNATURE", now2, f"Extraction de la signature du chèque {cheque_number}.", ext_id, "ExtractedSignature", admin_id))
                conn.commit()
                print(f"  ExtractedSignature Id {ext_id} quality {extraction_quality} crop {len(crop_bytes)} bytes")
                # Store for later AI
                extracted_crop_path = dest_crop
            else:
                print(f"  No crop base64")
                extraction_success=False
    except Exception as e:
        print(f"  Extraction persist failed {e}")
        try: conn.rollback()
        except: pass
        extraction_success=False
        # Need to clear?
        # Continue to AI? No, if extraction failed, no AI
        # Record and continue to next sample
        # For CSV we need to handle
        # Let's still record with no AI
        # Create a row with no AI scores
        # But we need to ensure we have variables
        # We'll set mean etc to None
        pass

    # If extraction failed, record CSV row with no AI and continue
    if not extraction_success:
        # Need to create CSV row
        # For now, we will handle after
        # Let's create a simple row
        # We need to append to CSV
        # We'll do it after the try
        pass

    # AI comparison
    mean_raw = None
    ref_scores = {}
    compared = 0
    best_id = None
    best_score = None
    auto_decision = "N/A"
    persisted = False
    verification_id = None
    if extraction_success:
        try:
            # Get reference paths from DB
            cur.execute("SELECT ReferenceSignatureId, ImagePath FROM ReferenceSignature WHERE CustomerId=5 AND IsActive=1 ORDER BY ReferenceSignatureId")
            refs_db = cur.fetchall()
            # For each ref, call compare-ai
            scores = []
            for ref_id, rel_ref in refs_db:
                ref_phys = web_root / rel_ref.lstrip('/').replace('/', os.sep)
                # Ensure ref file exists
                if not ref_phys.exists():
                    print(f"  Ref {ref_id} file missing {ref_phys}")
                    continue
                with open(extracted_crop_path, 'rb') as f1, open(ref_phys, 'rb') as f2:
                    resp = requests.post(f"{api_base}/api/signatures/compare-ai",
                                         files={"extracted_file": (Path(extracted_crop_path).name, f1, "image/png"),
                                                "reference_file": (ref_phys.name, f2, "image/png")},
                                         timeout=30)
                    if resp.status_code==200:
                        j=resp.json()
                        if j.get('success'):
                            sc = j.get('similarityScore')
                            ref_scores[ref_id]=sc
                            scores.append(sc)
                            print(f"    Ref {ref_id} score {sc:.6f}")
                        else:
                            print(f"    Ref {ref_id} API success false {j}")
                    else:
                        print(f"    Ref {ref_id} HTTP {resp.status_code}")
            if scores:
                compared = len(scores)
                mean_raw = sum(scores)/len(scores)
                # Best
                best_id = max(ref_scores, key=lambda k: ref_scores[k])
                # Tie: lowest Id among max score
                max_sc = max(ref_scores.values())
                candidates = [k for k,v in ref_scores.items() if abs(v-max_sc)<1e-9]
                best_id = min(candidates)
                best_score = ref_scores[best_id]
                print(f"  Mean {mean_raw:.6f} best {best_id} {best_score:.6f}")
                # Decision
                if mean_raw <= 0.0895:
                    auto_decision="Non conforme"
                elif mean_raw >= 0.6898:
                    auto_decision="Conforme"
                else:
                    auto_decision="Contrôle manuel"
                print(f"  Decision {auto_decision}")
                # Try to persist via LaunchVerification logic (insert VR + SC)
                # We will attempt direct DB insert, handling CHECK constraint
                try:
                    now3 = datetime.datetime.utcnow()
                    # Need ExtractedSignatureId
                    cur.execute("SELECT ExtractedSignatureId FROM ExtractedSignature WHERE ChequeId=?", (cheque_id,))
                    ext_id_row = cur.fetchone()
                    if ext_id_row:
                        ext_id = ext_id_row[0]
                        # Insert VerificationResult
                        # Note: need to handle mean rounding to 4 decimals
                        mean_rounded = round(mean_raw,4)
                        auto_dec_map = {"Conforme":1, "Non conforme":2, "Contrôle manuel":3}
                        auto_dec_val = auto_dec_map[auto_decision]
                        # FinalDecision: only if Conforme or Non conforme
                        final_dec = auto_dec_val if auto_dec_val in (1,2) else None
                        # Insert VR
                        curVR = conn.cursor()
                        curVR.execute("INSERT INTO VerificationResult (AutomaticDecision, ChequeId, FinalDecision, LowerThresholdUsed, ModelName, ModelVersion, SimilarityScore, UpperThresholdUsed, VerifiedAt) OUTPUT INSERTED.VerificationId VALUES (?,?,?,?,?,?,?,?,?)",
                                      (auto_dec_val, cheque_id, final_dec, 0.0895, "sig-verif-ai-v2", "v2", mean_rounded, 0.6898, now3))
                        verification_id = curVR.fetchone()[0]
                        # Insert SignatureComparisons
                        # Need to insert 5 rows, one per ref, with IsBestMatch
                        for ref_id, sc in ref_scores.items():
                            is_best = 1 if ref_id==best_id else 0
                            # Round to 4 decimals
                            sc_r = round(sc,4)
                            curVR.execute("INSERT INTO SignatureComparison (ExtractedSignatureId, IsBestMatch, ReferenceSignatureId, SimilarityScore, VerificationId) VALUES (?,?,?,?,?)",
                                          (ext_id, is_best, ref_id, sc_r, verification_id))
                        # Update cheque status
                        status_map = {1:3, 2:5, 3:4} # Conforme->Verifie 3, Non conforme->Rejete 5, ControleManuel->4
                        new_status = status_map[auto_dec_val]
                        curVR.execute("UPDATE Cheque SET Status=? WHERE ChequeId=?", (new_status, cheque_id))
                        # Audit VERIFY_CHEQUE
                        curVR.execute("INSERT INTO AuditLog (Action, CreatedAt, Description, EntityId, EntityName, UserId) VALUES (?,?,?,?,?,?)",
                                      ("VERIFY_CHEQUE", now3, f"Vérification du chèque {cheque_number}: décision {auto_dec_val} (score {mean_rounded:.4f}), {compared}/5 références.", verification_id, "VerificationResult", admin_id))
                        conn.commit()
                        persisted=True
                        print(f"  Persisted VR {verification_id} status {new_status}")
                    else:
                        print(f"  No extracted id for AI")
                except pyodbc.Error as e:
                    # Check if CHECK constraint violation
                    msg = str(e)
                    print(f"  Persist failed (likely CHECK constraint negative score): {msg[:500]}")
                    try: conn.rollback()
                    except: pass
                    persisted=False
                    verification_id=None
                    # Keep auto_decision but not persisted
                    # Also need to ensure ChangeTracker not poisoned (pyodbc doesn't have)
                except Exception as e:
                    print(f"  Persist exception {e}")
                    try: conn.rollback()
                    except: pass
                    persisted=False
            else:
                print(f"  No scores")
        except Exception as e:
            print(f"  AI exception {e}")
            import traceback; traceback.print_exc()
            persisted=False

    # Now we have data for CSV row
    # Need to get global bbox and IoU already computed
    # For CSV, need all 5 ref scores in order 4,5,6,7,8
    r4 = ref_scores.get(4, "")
    r5 = ref_scores.get(5, "")
    r6 = ref_scores.get(6, "")
    r7 = ref_scores.get(7, "")
    r8 = ref_scores.get(8, "")
    # Format scores to 6 decimals if present
    def fmt(v): return f"{v:.6f}" if isinstance(v, float) else str(v) if v!="" else ""
    # Global bbox string
    gb_str = f"{global_bbox[0]},{global_bbox[1]},{global_bbox[2]},{global_bbox[3]}" if global_bbox else ""
    mb_str = f"{manifest_bbox[0]},{manifest_bbox[1]},{manifest_bbox[2]},{manifest_bbox[3]}"
    iou_str = f"{iou_val:.4f}" if iou_val is not None else ""
    # Need to ensure we have cheque_id and number
    # For failed cases, we still have cheque_id/number
    # For CSV, we need to append
    # Open CSV in append
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csvmod.writer(f, quoting=csvmod.QUOTE_MINIMAL)
        writer.writerow([fname, gt, f"{mrow['source_signature_sheet']}#{mrow['source_annotation_id']}", cheque_id if 'cheque_id' in locals() else "", cheque_number if 'cheque_number' in locals() else "", verification_id if verification_id else "", str(extraction_success).lower(), classification, f"{qual:.4f}" if qual is not None else "", f"{comp:.4f}" if comp is not None else "", fmt(r4), fmt(r5), fmt(r6), fmt(r7), fmt(r8), compared, f"{mean_raw:.6f}" if mean_raw is not None else "", best_id if best_id else "", f"{best_score:.6f}" if best_score is not None else "", auto_decision, str(persisted).lower(), "", gb_str, mb_str, iou_str])
    print(f"  CSV row appended for {fname}")

print("\n=== REMAINING CAMPAIGN DONE ===")
conn.close()
