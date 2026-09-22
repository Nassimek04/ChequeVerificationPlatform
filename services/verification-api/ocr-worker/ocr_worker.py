"""Isolated PaddleOCR worker — runs in dedicated venv-ocr (no torch).

Called by main FastAPI via subprocess:
  python ocr_worker.py --image /tmp/cheque.png --lang fr

Outputs JSON to stdout:
{
  "success": true,
  "full_text": "...",
  "lines": [{"text": "...", "confidence": 0.98, "box": [[x,y],...]}],
  "processing_ms": 120,
  "lang": "fr",
  "device": "cpu"
}
"""

import argparse
import json
import sys
import time
import traceback

import os

# Disable PIR to avoid "strides is not right" / "ConvertPirAttribute2RuntimeAttribute" bugs on Paddle 3.x
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import cv2
import numpy as np


def _normalize(raw):
    lines = []
    if raw is None:
        return lines
    # PaddleOCR 3.7 predict returns [OCRResult] where OCRResult is dict-like with rec_texts
    if isinstance(raw, list) and len(raw) == 1 and hasattr(raw[0], 'get'):
        obj = raw[0]
        # OCRResult is dict-like with keys rec_texts, rec_scores, rec_polys/dt_polys
        try:
            rec_texts = obj.get('rec_texts') or obj.get('texts') or []
            rec_scores = obj.get('rec_scores') or obj.get('scores') or []
            rec_polys = obj.get('rec_polys') or obj.get('rec_boxes') or obj.get('dt_polys') or []
            raw = {"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys, "rec_boxes": rec_polys}
        except Exception:
            try:
                raw = dict(obj)
            except Exception:
                pass
    elif hasattr(raw, 'get') and not isinstance(raw, dict):
        # Single OCRResult dict-like
        try:
            raw = {"rec_texts": raw.get('rec_texts', []), "rec_scores": raw.get('rec_scores', []), "rec_polys": raw.get('rec_polys', []), "rec_boxes": raw.get('rec_polys', [])}
        except Exception:
            pass
    # PaddleOCR 3.7 returns [ {rec_texts...} ] — unwrap single dict
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        raw = raw[0]
    try:
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            for line in raw[0]:
                if line is None or len(line) < 2:
                    continue
                box = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = text_info[0], float(text_info[1])
                else:
                    text, conf = str(text_info), 0.0
                flat = []
                if isinstance(box, (list, tuple)):
                    for pt in box:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            flat.append([float(pt[0]), float(pt[1])])
                lines.append({"text": str(text), "confidence": float(conf), "box": flat})
            return lines
        if isinstance(raw, dict):
            texts = raw.get("rec_texts") or raw.get("texts") or []
            scores = raw.get("rec_scores") or raw.get("scores") or []
            boxes = raw.get("rec_boxes") or raw.get("dt_polys") or []
            for idx, txt in enumerate(texts):
                conf = float(scores[idx]) if idx < len(scores) else 0.0
                box = boxes[idx] if idx < len(boxes) else []
                flat = []
                if isinstance(box, (list, tuple)):
                    for pt in box:
                        if isinstance(pt, (list, tuple)):
                            flat.append([float(pt[0]), float(pt[1])])
                lines.append({"text": str(txt), "confidence": conf, "box": flat})
            return lines
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "text" in item:
                    lines.append({"text": str(item.get("text", "")), "confidence": float(item.get("confidence", 0.0)), "box": item.get("box", [])})
        return lines
    except Exception:
        return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to cheque image")
    parser.add_argument("--lang", default="fr", help="OCR lang")
    args = parser.parse_args()

    start = time.perf_counter()
    try:
        from paddleocr import PaddleOCR

        # Lazy init inside worker process — PaddleOCR 3.x API varies
        ocr = None
        for kwargs in [
            {"lang": args.lang},
            {"use_angle_cls": True, "lang": args.lang},
            {"use_textline_orientation": True, "lang": args.lang},
        ]:
            try:
                ocr = PaddleOCR(**kwargs)
                break
            except TypeError as e:
                last = e
                continue
            except Exception as e:
                if "show_log" in str(e) or "Unknown argument" in str(e):
                    continue
                raise
        if ocr is None:
            raise RuntimeError(f"PaddleOCR init failed lang={args.lang}: {last}")

        img = cv2.imread(args.image)
        if img is None:
            print(json.dumps({"success": False, "error": "Image decode failed"}))
            sys.exit(0)

        # PaddleOCR 3.7: ocr.ocr(img) or ocr.predict(img) — try predict first for 3.7
        raw = None
        last_err = None
        for fn in [
            lambda: ocr.predict(img) if hasattr(ocr, "predict") else None,
            lambda: ocr.ocr(img),
            lambda: ocr.ocr(img, cls=True) if hasattr(ocr, "ocr") else None,
        ]:
            if fn is None:
                continue
            try:
                raw = fn()
                if raw is not None:
                    break
            except TypeError as e:
                last_err = e
                if "cls" in str(e) or "unexpected" in str(e):
                    continue
                raise
            except Exception as e:
                last_err = e
                continue
        if raw is None:
            raise RuntimeError(f"OCR raw is None, last_err={last_err}")

        lines = _normalize(raw)
        full_text = "\n".join(l["text"] for l in lines if l["text"])
        processing_ms = int((time.perf_counter() - start) * 1000)

        # Device
        device = "cpu"
        try:
            import paddle
            try:
                device = paddle.device.get_device()
            except Exception:
                device = "cpu"
        except Exception:
            device = "cpu"

        out = {
            "success": True,
            "full_text": full_text,
            "lines": lines,
            "processing_ms": processing_ms,
            "lang": args.lang,
            "device": device,
        }
        print(json.dumps(out, ensure_ascii=False))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(0)


if __name__ == "__main__":
    main()
