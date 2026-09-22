import sys
from pathlib import Path
import cv2, numpy as np, zlib
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi

CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
sig_gray = cv2.imread(str(CEDAR_ROOT/"2"/"original_2_6.png"), cv2.IMREAD_GRAYSCALE)

def stable_seed(k): return zlib.crc32(k.encode("utf-8"))

def make(sig_gray, rng):
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
    otsu,_=cv2.threshold(blurred.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    alpha=(blurred<otsu).astype(np.float32)
    alpha=cv2.GaussianBlur(alpha,(3,3),0)
    px=int(rng.integers(region_x0,max(region_x0+1,region_x1-new_w)))
    py=int(rng.integers(region_y0,max(region_y0+1,region_y1-new_h)))
    roi=color[py:py+new_h,px:px+new_w].astype(np.float32)
    ink_color=np.array([40.,25.,15.])
    blended=roi*(1-alpha[...,None])+ink_color*alpha[...,None]
    color[py:py+new_h,px:px+new_w]=np.clip(blended,0,255).astype(np.uint8)
    return color,(px,py,new_w,new_h), target_h

def iou_calc(a,b):
    ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union else 0

def bbox_of(r,roi=None):
    x,y=int(r.x),int(r.y)
    if roi is not None: x+=int(roi.x); y+=int(roi.y)
    return (x,y,int(r.width),int(r.height))

best=None
for attempt in range(100):
    key=f"DEMO-V5-CONFORME-001-attempt-{attempt}"
    # Try deterministic seed per attempt
    rng=np.random.default_rng(stable_seed(key))
    img,paste,target_h = make(sig_gray, rng)
    try:
        res=extract_signature(img,0.40,0.40,0.98,0.98)
    except Exception as e:
        print(f"attempt {attempt} target_h {target_h} paste {paste} FAILED extraction {e}")
        continue
    full=bbox_of(res.signature_bbox,res.candidate_roi)
    iou=iou_calc(full,paste)
    # also analyze groups
    roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
    roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
    gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
    groups=len(an.groups)
    refined=len(an.refined_components)
    quality=res.extraction_quality
    # Heuristic: good if groups small (1-2) and iou high, or refined covers more
    score = iou*2 + quality*0.5 - groups*0.05
    if best is None or score>best[0]:
        best=(score, attempt, paste, target_h, iou, quality, groups, refined, an.completeness_score, res)
        print(f"NEW BEST attempt {attempt} target_h {target_h} paste {paste} iou {iou:.3f} quality {quality} groups {groups} refined {refined} completeness {an.completeness_score:.3f} score {score:.3f}")
    else:
        print(f"attempt {attempt} target_h {target_h} iou {iou:.3f} groups {groups} refined {refined} quality {quality}")

print("\n BEST:", best[1], best[2], best[3], f"iou {best[4]:.3f} groups {best[6]}")

# Also test specific benchmark seed for writer2 original_2_6
print("\n--- benchmark seed cheque-2-original_2_6.png ---")
rng=np.random.default_rng(stable_seed("cheque-2-original_2_6.png"))
img,paste,th=make(sig_gray,rng)
res=extract_signature(img,0.40,0.40,0.98,0.98)
full=bbox_of(res.signature_bbox,res.candidate_roi)
print(f"paste {paste} th {th} iou {iou_calc(full,paste):.3f} quality {res.extraction_quality}")

# Try manual small scale
print("\n--- manual target_h sweep ---")
for th in [130,135,140,150,168,180,190]:
    # fixed px/py center
    rng=np.random.default_rng(0) # dummy to get base without random target_h
    # manually construct with fixed th
    W,H=1600,700
    base=rng.normal(244.0,2.5,(H,W)).astype(np.float32)
    for i in range(14):
        y0=30+i*48+int(rng.integers(-6,7))
        amp=rng.uniform(3.0,9.0)
        phase=rng.uniform(0,2*np.pi)
        xs=np.arange(W); ys=(y0+amp*np.sin(xs/rng.uniform(60,140)+phase)).astype(int); ok=(ys>=0)&(ys<H); base[ys[ok],xs[ok]]-=rng.uniform(6.0,12.0)
    img=np.clip(base,0,255).astype(np.uint8)
    color=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(color,(18,18),(W-18,H-18),(140,140,140),1)
    cv2.line(color,(60,70),(420,70),(150,150,150),1)
    cv2.line(color,(60,300),(900,300),(150,150,150),1)
    cv2.rectangle(color,(1050,90),(1540,170),(150,150,150),1)
    for k in range(6):
        p1=(1070+k*75,155-int(rng.integers(10,55))); p2=(1120+k*75,105+int(rng.integers(5,45))); cv2.line(color,p1,p2,(120,120,120),1)
    micr_y=655; x=120
    while x<W-160:
        digit=str(int(rng.integers(0,10))); cv2.putText(color,digit,(x,micr_y),cv2.FONT_HERSHEY_SIMPLEX,rng.uniform(0.9,1.3),(30,30,30),2,cv2.LINE_AA); x+=int(rng.integers(38,58))
    region_x0,region_x1=660,1520; region_y0,region_y1=290,590
    scale=min(th/sig_gray.shape[0],(region_x1-region_x0)/sig_gray.shape[1])
    new_w=max(1,int(round(sig_gray.shape[1]*scale))); new_h=max(1,int(round(sig_gray.shape[0]*scale)))
    resized=cv2.resize(sig_gray,(new_w,new_h),interpolation=cv2.INTER_AREA)
    blurred=cv2.GaussianBlur(resized.astype(np.float32),(3,3),0)
    otsu,_=cv2.threshold(blurred.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    alpha=(blurred<otsu).astype(np.float32); alpha=cv2.GaussianBlur(alpha,(3,3),0)
    px=1000; py=400 # center
    # ensure fits
    px=min(px,region_x1-new_w); py=min(py,region_y1-new_h)
    roi=color[py:py+new_h,px:px+new_w].astype(np.float32); ink_color=np.array([40.,25.,15.]); blended=roi*(1-alpha[...,None])+ink_color*alpha[...,None]; color[py:py+new_h,px:px+new_w]=np.clip(blended,0,255).astype(np.uint8)
    paste=(px,py,new_w,new_h)
    try:
        res=extract_signature(color,0.40,0.40,0.98,0.98)
        full=bbox_of(res.signature_bbox,res.candidate_roi)
        roi2=_compute_roi(color.shape[1],color.shape[0],0.40,0.40,0.98,0.98)
        rc=color[roi2.y:roi2.y+roi2.height,roi2.x:roi2.x+roi2.width]
        g=cv2.cvtColor(rc,cv2.COLOR_BGR2GRAY); g=cv2.GaussianBlur(g,(3,3),0)
        an=_analyze_ink(g,roi2,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
        print(f"th {th} new_w {new_w} paste {paste} iou {iou_calc(full,paste):.3f} groups {len(an.groups)} sel {an.selected_group_index} refined {len(an.refined_components)} quality {res.extraction_quality}")
    except Exception as e:
        print(f"th {th} fail {e}")
