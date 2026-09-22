import pathlib, csv, pyodbc, requests, json
base_pkg = pathlib.Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase1\SSBI_controlled_test_signer7")
web_root = pathlib.Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\src\ChequeVerification.Web\wwwroot")
api_base='http://localhost:8000'
conn_str='DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=ChequeVerificationDB;Trusted_Connection=yes;'
conn=pyodbc.connect(conn_str)
cur=conn.cursor()
cur.execute('SELECT ChequeId, ChequeNumber FROM Cheque WHERE ChequeId=9')
cheque_id, cheque_num = cur.fetchone()
cur.execute('SELECT VerificationId, SimilarityScore, AutomaticDecision FROM VerificationResult WHERE ChequeId=9')
vr_id, sim, auto = cur.fetchone()
cur.execute('SELECT ReferenceSignatureId, SimilarityScore, IsBestMatch FROM SignatureComparison WHERE VerificationId=? ORDER BY ReferenceSignatureId', (vr_id,))
comps = cur.fetchall()
ref_scores = {r[0]: r[1] for r in comps}
best_id = [r[0] for r in comps if r[2]==1][0]
best_score = ref_scores[best_id]
p = base_pkg / 'genuine_cheques' / 'S7_GENUINE_01.png'
with open(p,'rb') as f:
    r = requests.post(f'{api_base}/api/signatures/debug', files={'file': (p.name, f, 'image/png')}, timeout=30)
    j=r.json()
    roi=j['candidate_roi']
    bbox=j['signature_bbox']
    qual=j['extraction_quality']
    comp=j['completeness_score']
    global_bbox = (roi['x']+bbox['x'], roi['y']+bbox['y'], bbox['width'], bbox['height'])
    print(f'ROI {roi} bbox {bbox} global {global_bbox} qual {qual} comp {comp}')
    import csv as csvmod
    manifest={}
    with open(base_pkg/'manifest.csv') as mf:
        for row in csvmod.DictReader(mf):
            manifest[row['file']]=row
    mrow=manifest['genuine_cheques/S7_GENUINE_01.png']
    inner=mrow['placed_signature_bbox'].strip().strip('[]')
    parts=[int(p.strip()) for p in inner.split(',')]
    manifest_bbox=(parts[0],parts[1],parts[2],parts[3])
    def iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[0]+a[2],b[0]+b[2]); y2=min(a[1]+a[3],b[1]+b[3])
        if x2<=x1 or y2<=y1: return 0
        inter=(x2-x1)*(y2-y1)
        union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/union
    iou_val=iou(global_bbox, manifest_bbox)
    print(f'IoU {iou_val}')
    csv_path = pathlib.Path(r"C:\Users\nassime khatib\Desktop\ChequeVerificationPlatform\artifacts\ssbi_phase3\ssbi_phase3_results.csv")
    header=['Sample','GroundTruth','SourceSignature','ChequeId','ChequeNumber','VerificationId','ExtractionSuccess','ExtractionClassification','ExtractionQuality','ExtractionCompleteness','Ref4Score','Ref5Score','Ref6Score','Ref7Score','Ref8Score','ComparedReferenceCount','MeanRawScore','BestReferenceId','BestReferenceScore','AutomaticDecision','Persisted','Notes','GlobalBbox','ManifestBbox','IoU']
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(header)
        auto_map={1:'Conforme',2:'Non conforme',3:'Contrôle manuel'}
        auto_str=auto_map[auto]
        w.writerow(['S7_GENUINE_01.png','GENUINE', f"{mrow['source_signature_sheet']}#{mrow['source_annotation_id']}", cheque_id, cheque_num, vr_id, 'true','CLEAN', f'{qual:.4f}', f'{comp:.4f}', f"{ref_scores[4]:.6f}", f"{ref_scores[5]:.6f}", f"{ref_scores[6]:.6f}", f"{ref_scores[7]:.6f}", f"{ref_scores[8]:.6f}", 5, f'{sim:.6f}', best_id, f'{best_score:.6f}', auto_str, 'true', 'pre-existing Phase2', f'{global_bbox[0]},{global_bbox[1]},{global_bbox[2]},{global_bbox[3]}', f'{manifest_bbox[0]},{manifest_bbox[1]},{manifest_bbox[2]},{manifest_bbox[3]}', f'{iou_val:.4f}'])
    print('CSV created with G01')
    print(open(csv_path).read())
