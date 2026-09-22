import sys
from pathlib import Path
import cv2, numpy as np
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi, EXTRACTION_PIPELINE_VERSION

p = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-FORGED-001.png")
paste = (1276,405,213,151)
img = cv2.imread(str(p))
print(f"img {img.shape} paste {paste}")

def bbox_of(r, roi=None):
    x,y = int(r.x), int(r.y)
    if roi: x+=int(roi.x); y+=int(roi.y)
    return (x,y,int(r.width),int(r.height))
def iou(a,b):
    ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union else 0

try:
    res = extract_signature(img,0.40,0.40,0.98,0.98)
    print(f"success pipeline {EXTRACTION_PIPELINE_VERSION}")
    print(f"ROI {res.candidate_roi.x},{res.candidate_roi.y},{res.candidate_roi.width},{res.candidate_roi.height}")
    print(f"bbox ROI-local {res.signature_bbox.x},{res.signature_bbox.y},{res.signature_bbox.width},{res.signature_bbox.height}")
    full=bbox_of(res.signature_bbox,res.candidate_roi)
    print(f"full {full} paste {paste} iou={iou(full,paste):.4f} quality={res.extraction_quality}")
    crop=cv2.imdecode(np.frombuffer(res.signature_png_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
    print(f"crop {crop.shape}")
    # analysis
    roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
    roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
    gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(3,3),0)
    an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
    print(f"groups {len(an.groups)} sel {an.selected_group_index} refined {len(an.refined_components)} completeness {an.completeness_score} kept {len(an.kept_components)} all {len(an.all_components)}")
    for i,g in enumerate(an.groups):
        x0=min(c.x for c in g); x1=max(c.x+c.width for c in g); y0=min(c.y for c in g); y1=max(c.y+c.height for c in g)
        print(f"  g{i} n={len(g)} bbox {x0},{y0},{x1-x0},{y1-y0} score {an.group_scores[i]:.5f}")
except Exception as e:
    import traceback; traceback.print_exc()
