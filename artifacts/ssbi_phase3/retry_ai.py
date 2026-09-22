import pyodbc, requests, pathlib, datetime, csv, os, hashlib, uuid
from pathlib import Path

base_pkg = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7")
web_root = Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot")
api_base="http://localhost:8000"
conn_str="DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=ChequeVerificationDB;Trusted_Connection=yes;"
conn=pyodbc.connect(conn_str)
conn.autocommit=False
cur=conn.cursor()

def get_admin():
    cur.execute("SELECT TOP 1 UserId FROM [User] WHERE RoleId=1")
    return cur.fetchone()[0]
admin_id=get_admin()
print(f"Admin {admin_id}")

# Get all cheques for SSBI that have extraction but no VR
cur.execute("SELECT ChequeId, ChequeNumber FROM Cheque WHERE CustomerId=5 AND ChequeId !=9 ORDER BY ChequeId")
cheques = cur.fetchall()
print(f"Cheques to retry: {len(cheques)}")

for cheque_id, cheque_num in cheques:
    print(f"\n--- Cheque {cheque_id} {cheque_num} ---")
    # Get extracted signature path
    cur2=conn.cursor()
    cur2.execute("SELECT ExtractedSignatureId, ImagePath FROM ExtractedSignature WHERE ChequeId=?", (cheque_id,))
    row=cur2.fetchone()
    if not row:
        print("  No extracted signature")
        continue
    ext_id, rel_crop = row
    crop_path = web_root / rel_crop.lstrip('/').replace('/', os.sep)
    if not crop_path.exists():
        print(f"  Crop missing {crop_path}")
        continue
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
                    sc=j['similarity_score']
                    ref_scores[ref_id]=sc
                    print(f"  Ref {ref_id} {sc:.6f}")
                else:
                    print(f"  Ref {ref_id} API success false {j}")
            else:
                print(f"  Ref {ref_id} HTTP {resp.status_code} {resp.text[:300]}")
    if not ref_scores:
        print("  No scores")
        continue
    compared=len(ref_scores)
    mean_raw=sum(ref_scores.values())/len(ref_scores)
    best_id = max(ref_scores, key=lambda k: ref_scores[k])
    max_sc = max(ref_scores.values())
    candidates=[k for k,v in ref_scores.items() if abs(v-max_sc)<1e-9]
    best_id=min(candidates)
    best_score=ref_scores[best_id]
    print(f"  Mean {mean_raw:.6f} best {best_id} {best_score:.6f}")
    if mean_raw <= 0.0895:
        auto="Non conforme"; auto_val=2
    elif mean_raw >=0.6898:
        auto="Conforme"; auto_val=1
    else:
        auto="Contrôle manuel"; auto_val=3
    print(f"  Decision {auto}")
    # Try to persist
    try:
        now=datetime.datetime.utcnow()
        mean_rounded=round(mean_raw,4)
        final_dec = auto_val if auto_val in (1,2) else None
        cur3=conn.cursor()
        cur3.execute("INSERT INTO VerificationResult (AutomaticDecision, ChequeId, FinalDecision, LowerThresholdUsed, ModelName, ModelVersion, SimilarityScore, UpperThresholdUsed, VerifiedAt) OUTPUT INSERTED.VerificationId VALUES (?,?,?,?,?,?,?,?,?)",
                     (auto_val, cheque_id, final_dec, 0.0895, "sig-verif-ai-v2", "v2", mean_rounded, 0.6898, now))
        vr_id=cur3.fetchone()[0]
        for ref_id, sc in ref_scores.items():
            is_best=1 if ref_id==best_id else 0
            sc_r=round(sc,4)
            cur3.execute("INSERT INTO SignatureComparison (ExtractedSignatureId, IsBestMatch, ReferenceSignatureId, SimilarityScore, VerificationId) VALUES (?,?,?,?,?)",
                         (ext_id, is_best, ref_id, sc_r, vr_id))
        status_map={1:3,2:5,3:4}
        new_status=status_map[auto_val]
        cur3.execute("UPDATE Cheque SET Status=? WHERE ChequeId=?", (new_status, cheque_id))
        cur3.execute("INSERT INTO AuditLog (Action, CreatedAt, Description, EntityId, EntityName, UserId) VALUES (?,?,?,?,?,?)",
                     ("VERIFY_CHEQUE", now, f"Vérification du chèque {cheque_num}: décision {auto_val} (score {mean_rounded:.4f}), {compared}/5 références.", vr_id, "VerificationResult", admin_id))
        conn.commit()
        print(f"  Persisted VR {vr_id} status {new_status}")
    except pyodbc.Error as e:
        print(f"  Persist failed (likely CHECK constraint negative score): {e}")
        try: conn.rollback()
        except: pass
    except Exception as e:
        print(f"  Persist exception {e}")
        import traceback; traceback.print_exc()
        try: conn.rollback()
        except: pass

print("\n=== RETRY DONE ===")
conn.close()
