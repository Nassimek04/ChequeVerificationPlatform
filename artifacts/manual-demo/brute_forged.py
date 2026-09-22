import sys
from pathlib import Path
import cv2, numpy as np, zlib, hashlib
SERVICE_ROOT = Path(r"C:\Dev\ChequeVerificationPlatform\services\verification-api")
sys.path.insert(0, str(SERVICE_ROOT))
from app.services.signature_extraction_service import extract_signature, _analyze_ink, _compute_roi

CEDAR_ROOT = Path(r"C:\Users\nassime khatib\Downloads\archive\CEDAR\CEDAR")
sig_gray = cv2.imread(str(CEDAR_ROOT/"8"/"forgeries_8_4.png"), cv2.IMREAD_GRAYSCALE)
print(f"sig {sig_gray.shape}")

def stable_seed(k): return zlib.crc32(k.encode("utf-8"))

def make(sig_gray, rng, cheque_number="DEMO-V5-FORGED-001"):
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
    cv2.rectangle(color,(20,20),(W-20,H-20),(175,175,175),1)
    cv2.putText(color,"BANQUE DEMO TEST  -  SPECIMEN",(42,52),cv2.FONT_HERSHEY_SIMPLEX,0.62,(55,55,55),1,cv2.LINE_AA)
    cv2.putText(color,"AGENCE CENTRALE DEMO  |  TEST ONLY - NOT A REAL BANK CHEQUE",(42,73),cv2.FONT_HERSHEY_SIMPLEX,0.38,(95,95,95),1,cv2.LINE_AA)
    cv2.putText(color,f"CHEQUE N: {cheque_number}",(1120,48),cv2.FONT_HERSHEY_SIMPLEX,0.48,(45,45,45),1,cv2.LINE_AA)
    cv2.line(color,(60,70),(420,70),(150,150,150),1)
    cv2.putText(color,"Date: 03/09/2026",(60,62),cv2.FONT_HERSHEY_SIMPLEX,0.42,(90,90,90),1,cv2.LINE_AA)
    cv2.putText(color,"Casablanca",(285,92),cv2.FONT_HERSHEY_SIMPLEX,0.42,(90,90,90),1,cv2.LINE_AA)
    cv2.putText(color,"Payez contre ce cheque la somme de :",(42,128),cv2.FONT_HERSHEY_SIMPLEX,0.40,(100,100,100),1,cv2.LINE_AA)
    cv2.line(color,(60,300),(900,300),(150,150,150),1)
    cv2.putText(color,"Quarante mille dirhams",(70,292),cv2.FONT_HERSHEY_SIMPLEX,0.55,(35,35,35),1,cv2.LINE_AA)
    cv2.putText(color,"A  M. / Mme  TEST DEMO CLIENT",(70,325),cv2.FONT_HERSHEY_SIMPLEX,0.42,(90,90,90),1,cv2.LINE_AA)
    cv2.rectangle(color,(1050,90),(1540,170),(150,150,150),1)
    for k in range(6):
        p1=(1070+k*75,155-int(rng.integers(10,55)))
        p2=(1120+k*75,105+int(rng.integers(5,45)))
        cv2.line(color,p1,p2,(120,120,120),1)
    cv2.putText(color,"40 000,00 MAD",(1080,142),cv2.FONT_HERSHEY_SIMPLEX,0.78,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(color,"Dirhams Marocains",(1085,162),cv2.FONT_HERSHEY_SIMPLEX,0.32,(105,105,105),1,cv2.LINE_AA)
    cv2.putText(color,f"N CHEQUE {cheque_number}  |  COMPTE DEMO 011 123456789  |  40 000,00 MAD  |  03/09/2026",(48,630),cv2.FONT_HERSHEY_SIMPLEX,0.36,(95,95,95),1,cv2.LINE_AA)
    overlay=color.copy()
    cv2.putText(overlay,"TEST  -  DEMO  -  NOT A REAL CHEQUE",(320,380),cv2.FONT_HERSHEY_SIMPLEX,1.05,(185,185,185),2,cv2.LINE_AA)
    cv2.putText(overlay,"SPECIMEN  -  DOCUMENT DE TEST  -  NE PAS ENCAISSER",(380,420),cv2.FONT_HERSHEY_SIMPLEX,0.52,(190,190,190),1,cv2.LINE_AA)
    color=cv2.addWeighted(overlay,0.55,color,0.45,0)
    cv2.rectangle(color,(640,38),(880,78),(160,90,90),1)
    cv2.putText(color,"TEST / DEMO - SPECIMEN",(652,66),cv2.FONT_HERSHEY_SIMPLEX,0.42,(150,60,60),1,cv2.LINE_AA)
    cv2.putText(color,"--- CE CHEQUE EST UN SPECIMEN DE TEST - DOCUMENT SYNTHETIQUE SANS VALEUR BANCAIRE ---",(150,682),cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,110,110),1,cv2.LINE_AA)
    micr_y=655; x=120
    while x<W-160:
        digit=str(int(rng.integers(0,10)))
        cv2.putText(color,digit,(x,micr_y),cv2.FONT_HERSHEY_SIMPLEX,rng.uniform(0.9,1.3),(30,30,30),2,cv2.LINE_AA)
        x+=int(rng.integers(38,58))
    cv2.line(color,(700,595),(1480,595),(155,155,155),1)
    cv2.putText(color,"Signature du tireur  /  Signature du client",(860,615),cv2.FONT_HERSHEY_SIMPLEX,0.36,(110,110,110),1,cv2.LINE_AA)
    cv2.putText(color,"(Ne pas deborder du cadre)",(970,630),cv2.FONT_HERSHEY_SIMPLEX,0.30,(125,125,125),1,cv2.LINE_AA)
    region_x0,region_x1=660,1520; region_y0,region_y1=290,590
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

def iou(a,b):
    ax1,ay1=a[0],a[1]; ax2,ay2=a[0]+a[2],a[1]+a[3]; bx1,by1=b[0],b[1]; bx2,by2=b[0]+b[2],b[1]+b[3]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union else 0
def bbox_of(r,roi=None):
    x,y=int(r.x),int(r.y)
    if roi: x+=int(roi.x); y+=int(roi.y)
    return (x,y,int(r.width),int(r.height))

best=None
for attempt in range(100):
    key=f"DEMO-V5-FORGED-001-attempt-{attempt}"
    rng=np.random.default_rng(stable_seed(key))
    img,paste,th = make(sig_gray, rng)
    try:
        res=extract_signature(img,0.40,0.40,0.98,0.98)
    except Exception as e:
        print(f"attempt {attempt} th {th} paste {paste} FAIL {e}")
        continue
    full=bbox_of(res.signature_bbox,res.candidate_roi)
    i=iou(full,paste)
    roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
    roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
    gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(3,3),0)
    an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
    groups=len(an.groups); refined=len(an.refined_components); quality=res.extraction_quality
    score = i*2 + quality*0.5 - groups*0.02  # heuristic
    # Prefer completeness 1.0 and IoU>0.30
    if an.completeness_score < 0.99: score -= 1
    if i < 0.30: score -= 1
    if best is None or score>best[0]:
        best=(score, attempt, paste, th, i, quality, groups, refined, an.completeness_score, res, an)
        print(f"NEW BEST attempt {attempt} th {th} paste {paste} iou {i:.3f} q {quality} groups {groups} refined {refined} comp {an.completeness_score:.3f} score {score:.3f}")
    else:
        print(f"attempt {attempt} th {th} iou {i:.3f} q {quality} groups {groups} refined {refined} comp {an.completeness_score:.3f}")

print(f"\nBEST {best[1]} {best[2]} th {best[3]} iou {best[4]:.3f} q {best[5]} groups {best[6]} refined {best[7]} comp {best[8]}")

# Also check primary seed
print("\n=== primary seed ===")
rng=np.random.default_rng(stable_seed("DEMO-V5-FORGED-001"))
img,paste,th = make(sig_gray, rng)
res=extract_signature(img,0.40,0.40,0.98,0.98)
full=bbox_of(res.signature_bbox,res.candidate_roi)
print(f"paste {paste} th {th} iou {iou(full,paste):.3f} q {res.extraction_quality}")
roi=_compute_roi(img.shape[1],img.shape[0],0.40,0.40,0.98,0.98)
roi_crop=img[roi.y:roi.y+roi.height,roi.x:roi.x+roi.width]
gray=cv2.cvtColor(roi_crop,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(3,3),0)
an=_analyze_ink(gray,roi,0.10,0.0005,0.60,0.01,0.01,0.06,6.0,0.03,0.10,1,3,0.018,0.15,0.15,0.08,0.08,0.04,0.5)
print(f"groups {len(an.groups)} refined {len(an.refined_components)} comp {an.completeness_score}")
