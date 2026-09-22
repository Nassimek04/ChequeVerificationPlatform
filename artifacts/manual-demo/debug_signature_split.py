import sys
from pathlib import Path
import cv2, numpy as np, zlib
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi, EXTRACTION_PIPELINE_VERSION

CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
CANDIDATE_SRC = CEDAR_ROOT / "2" / "original_2_6.png"
sig_gray = cv2.imread(str(CANDIDATE_SRC), cv2.IMREAD_GRAYSCALE)

def stable_seed(k): return zlib.crc32(k.encode("utf-8"))

def make_pure(sig_gray, rng):
    W,H=1600,700
    base=rng.normal(244.0,2.5,(H,W)).astype(np.float32)
    for i in range(14):
        y0=30+i*48+int(rng.integers(-6,7))
        amp=rng.uniform(3.0,9.0)
        phase=rng.uniform(0,2*np.pi)
        xs=np.arange(W)
        ys=(y0+amp*np.sin(xs/rng.uniform(60,140)+phase)).astype(int)
        ok=(ys>=0)&(ys<H)
        base[ys[ok],xs[ok]]-=rng.uniform(6.0,12.0)
    img=np.clip(base,0,255).astype(np.uint8)
    color=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color,(18,18),(W-18,H-18),(140,140,140),1)
    cv2.line(color,(60,70),(420,70),(150,150,150),1)
    cv2.line(color,(60,620-320),(900,300),(150,150,150),1)
    cv2.rectangle(color,(1050,90),(1540,170),(150,150,150),1)
    for k in range(6):
        p1=(1070+k*75,155-int(rng.integers(10,55)))
        p2=(1120+k*75,105+int(rng.integers(5,45)))
        cv2.line(color,p1,p2,(120,120,120),1)
    micr_y=655
    x=120
    while x<W-160:
        digit=str(int(rng.integers(0,10)))
        cv2.putText(color,digit,(x,micr_y),cv2.FONT_HERSHEY_SIMPLEX,rng.uniform(0.9,1.3),(30,30,30),2,cv2.LINE_AA)
        x+=int(rng.integers(38,58))
    region_x0,region_x1=660,1520
    region_y0,region_y1=290,590
    target_h=int(rng.integers(130,195))
    scale=min(target_h/sig_gray.shape[0],(region_x1-region_x0)/sig_gray.shape[1])
    new_w=max(1,int(round(sig_gray.shape[1]*scale)))
    new_h=max(1,int(round(sig_gray.shape[0]*scale)))
    resized=cv2.resize(sig_gray,(new_w,new_h),interpolation=cv2.INTER_AREA)
    blurred=cv2.GaussianBlur(resized.astype(np.float32),(3,3),0)
    otsu_thr,_=cv2.threshold(blurred.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    alpha=(blurred<otsu_thr).astype(np.float32)
    alpha=cv2.GaussianBlur(alpha,(3,3),0)
    px=int(rng.integers(region_x0,max(region_x0+1,region_x1-new_w)))
    py=int(rng.integers(region_y0,max(region_y0+1,region_y1-new_h)))
    roi=color[py:py+new_h,px:px+new_w].astype(np.float32)
    ink_color=np.array([40.,25.,15.])
    blended=roi*(1-alpha[...,None])+ink_color*alpha[...,None]
    color[py:py+new_h,px:px+new_w]=np.clip(blended,0,255).astype(np.uint8)
    paste=(px,py,new_w,new_h)
    return color,paste

def analyze_path(img, paste):
    res=extract_signature(img,0.40,0.40,0.98,0.98)
    from pathlib import Path as P
    def bbox_of(r,roi=None):
        x,y=int(r.x),int(r.y)
        if roi is not None: x+=int(roi.x); y+=int(roi.y)
        return (x,y,int(r.width),int(r.height))
    def iou(a,b):
        ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
        ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
        return inter/union if union else 0
    full=bbox_of(res.signature_bbox,res.candidate_roi)
    print(f" paste {paste} full_bbox {full} iou={iou(full,paste):.4f} quality={res.extraction_quality} crop {res.signature_bbox.width}x{res.signature_bbox.height}")
    # also analyze ink
    roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
    roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
    gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
    print(f"  groups {len(an.groups)} selected {an.selected_group_index} refined {len(an.refined_components)} completeness {an.completeness_score} kept {len(an.kept_components)} all {len(an.all_components)}")
    # show group bboxes
    for idx,g in enumerate(an.groups):
        ink=sum(c.width*c.height for c in g)
        x0=min(c.x for c in g); x1=max(c.x+c.width for c in g); y0=min(c.y for c in g); y1=max(c.y+c.height for c in g)
        print(f"    group {idx}: n={len(g)} bbox roi-local ({x0},{y0},{x1-x0},{y1-y0}) inkArea~{ink} score {an.group_scores[idx]:.5f}")
        for c in g[:5]:
            print(f"       comp ({c.x},{c.y},{c.width},{c.height})")
    return res

# pure
rng_pure=np.random.default_rng(stable_seed("DEMO-V5-CONFORME-001"))
# Use same seed for pure but we need to replicate exactly same rng sequence as Moroccan? Moroccan adds extra rng draws for watermark? Actually Moroccan consumes rng for watermark? No watermark uses overlay not rng, but amount text draws are deterministic not rng. So sequence identical.
# For pure test, use same seed and same target_h etc — sequence identical.
pure_img,pure_paste=make_pure(sig_gray, np.random.default_rng(stable_seed("DEMO-V5-CONFORME-001")))
print("--- PURE VALIDATED ---")
analyze_path(pure_img, pure_paste)

# moroccan
from pathlib import Path as PP
mor_img=cv2.imread(str(Path(r"C:\Dev\ChequeVerificationPlatform\artifacts\manual-demo\DEMO-V5-CONFORME-001.png")))
print("\n--- MOROCCAN ---")
# paste known
analyze_path(mor_img, (1045,397,474,168))

# also check signature itself ink bbox
blurred=cv2.GaussianBlur(cv2.resize(sig_gray,(474,168),interpolation=cv2.INTER_AREA).astype(np.float32),(3,3),0)
otsu,_=cv2.threshold(blurred.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask=(blurred<otsu)
ys,xs=np.where(mask)
print(f"\nResized sig ink bbox: x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()} w {xs.max()-xs.min()+1} h {ys.max()-ys.min()+1} ink {np.count_nonzero(mask)} / {474*168}")

# show original sig crop
orig_crop=cv2.imread(str(CANDIDATE_SRC),cv2.IMREAD_UNCHANGED)
print(f"Original sig shape {orig_crop.shape}")
