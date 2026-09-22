import sys
from pathlib import Path
import cv2, numpy as np
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi, EXTRACTION_PIPELINE_VERSION

CHEQUE_PATH = Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-FORGED-001.png")
PASTE_RECT = (1103, 331, 200, 142)  # from best seed attempt-40
SEED_KEY = "DEMO-V5-FORGED-001-attempt-40"

def iou(a,b):
    ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union else 0
def bbox_of(r,roi=None):
    x,y=int(r.x),int(r.y)
    if roi: x+=int(roi.x); y+=int(roi.y)
    return (x,y,int(r.width),int(r.height))

img = cv2.imread(str(CHEQUE_PATH))
assert img is not None, "imread failed"
print(f"Image {img.shape} paste {PASTE_RECT} seed {SEED_KEY}")
print(f"Pipeline {EXTRACTION_PIPELINE_VERSION}")
roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
print(f"ROI {roi.x},{roi.y},{roi.width},{roi.height}")
res=extract_signature(img,0.40,0.40,0.98,0.98)
print(f"Success {res is not None}")
print(f"bbox ROI-local {res.signature_bbox.x},{res.signature_bbox.y},{res.signature_bbox.width},{res.signature_bbox.height}")
full=bbox_of(res.signature_bbox,res.candidate_roi)
print(f"full {full} paste {PASTE_RECT} iou {iou(full,PASTE_RECT):.4f} quality {res.extraction_quality}")
crop=cv2.imdecode(np.frombuffer(res.signature_png_bytes,np.uint8),cv2.IMREAD_UNCHANGED)
print(f"crop shape {crop.shape}")
# analysis
roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(3,3),0)
an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
print(f"micr_band_top {an.micr_band_top} -> full y {roi.y+an.micr_band_top}")
print(f"all {len(an.all_components)} kept {len(an.kept_components)} rejected {len(an.rejected_components)} micr {len(an.micr_rejected_components)}")
print(f"groups {len(an.groups)} sel {an.selected_group_index} refined {len(an.refined_components)} discarded {len(an.discarded_from_selected_group_indices)}")
print(f"completeness {an.completeness_score:.4f} ({an.completeness_refined_ink}/{an.completeness_reference_ink}) quality_base {an.quality_base:.4f} factor {an.quality_completeness_factor:.4f} final {an.quality:.4f}")
print(f"dominant idx {an.dominant_core_index_in_group} ink {an.dominant_component_ink} score {an.dominant_component_score:.5f}")
print(f"selection {an.selection_reason[:160]}")
print(f"refinement {an.refinement_reason[:220]}")
# checks
iou_val=iou(full,PASTE_RECT)
quality_ok=res.extraction_quality>0.35
completeness_ok=an.completeness_score>=0.99
contamination_ok=len(an.discarded_from_selected_group_indices)<=2 and len(an.micr_rejected_components)<=10
candidate_ok=iou_val>0.30 and quality_ok and completeness_ok
print(f"Checks: iou>0.30 {iou_val>0.30} ({iou_val:.4f}) quality>0.35 {quality_ok} completeness>=0.99 {completeness_ok} contamination {contamination_ok} candidate_ok {candidate_ok}")

# save extracted
out_crop=Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-FORGED-001_extracted.png")
out_crop.write_bytes(res.signature_png_bytes)
print(f"Saved extracted {out_crop} {out_crop.stat().st_size} bytes")
# diagnostic overlay
vis=img.copy()
cv2.rectangle(vis,(PASTE_RECT[0],PASTE_RECT[1]),(PASTE_RECT[0]+PASTE_RECT[2],PASTE_RECT[1]+PASTE_RECT[3]),(0,255,0),2)
cv2.putText(vis,"paste",(PASTE_RECT[0],PASTE_RECT[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,180,0),1,cv2.LINE_AA)
cv2.rectangle(vis,(full[0],full[1]),(full[0]+full[2],full[1]+full[3]),(0,0,255),2)
cv2.putText(vis,f"extracted IoU={iou_val:.2f}",(full[0],full[1]+full[3]+14),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),1,cv2.LINE_AA)
cv2.rectangle(vis,(roi.x,roi.y),(roi.x+roi.width,roi.y+roi.height),(255,0,0),1)
diag=Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-FORGED-001_diagnostic.png")
cv2.imwrite(str(diag),vis)
print(f"Saved diagnostic {diag}")
print(f"Visually complete: {'YES' if candidate_ok else 'NO'}")
