"""PaddleOCR service — lazy singleton, CPU, no torch."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Allowed validation mirrors main FastAPI
MAX_OCR_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIM = 32
MAX_DIM = 8000

# Lazy singleton
_ocr_instance: Optional[Any] = None
_ocr_lock = Lock()
_ocr_init_error: Optional[str] = None
_ocr_lang: str = "fr"
_ocr_device: str = "cpu"


def _init_paddle_env():
    import os

    # Disable PIR to avoid Paddle 3.x ConvertPirAttribute bugs seen on Windows.
    # Harmless on Linux CPU and keeps behavior consistent.
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")


def get_or_create_ocr(lang: str = "fr"):
    """Lazy init PaddleOCR — reuse across requests."""
    global _ocr_instance, _ocr_init_error, _ocr_lang
    if _ocr_instance is not None and _ocr_lang == lang:
        return _ocr_instance
    with _ocr_lock:
        if _ocr_instance is not None and _ocr_lang == lang:
            return _ocr_instance
        _init_paddle_env()
        start = time.perf_counter()
        try:
            from paddleocr import PaddleOCR  # type: ignore

            ocr = None
            # Try new 3.7 API first, fallback to legacy kwargs
            for kwargs in [
                {"lang": lang},
                {"use_angle_cls": True, "lang": lang},
                {"use_textline_orientation": True, "lang": lang},
            ]:
                try:
                    ocr = PaddleOCR(**kwargs)
                    break
                except TypeError as e:
                    last = e
                    continue
            if ocr is None:
                raise RuntimeError(f"PaddleOCR init failed lang={lang}: {last}")
            _ocr_instance = ocr
            _ocr_lang = lang
            _ocr_init_error = None
            elapsed = int((time.perf_counter() - start) * 1000)
            logger.info("PaddleOCR initialized lang=%s device=cpu in %sms", lang, elapsed)
            return ocr
        except Exception as exc:
            _ocr_init_error = str(exc)
            logger.exception("PaddleOCR initialization failed")
            raise


def validate_and_decode(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Le fichier est vide.")
    if len(image_bytes) > MAX_OCR_FILE_SIZE_BYTES:
        raise ValueError("Le fichier dépasse la taille maximale autorisée (10 Mo).")
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("L'image ne peut pas être décodée.")
    if img.size == 0:
        raise ValueError("Image vide après décodage.")
    h, w = img.shape[:2]
    if h < MIN_DIM or w < MIN_DIM:
        raise ValueError("Image trop petite pour l'OCR.")
    if h > MAX_DIM or w > MAX_DIM:
        raise ValueError("Image trop grande pour l'OCR.")
    return img


def _normalize(raw) -> List[Dict[str, Any]]:
    """Normalize various PaddleOCR return shapes to [{text,confidence,box}]."""
    lines: List[Dict[str, Any]] = []
    if raw is None:
        return lines
    # PaddleOCR 3.7 predict returns [OCRResult] dict-like
    if isinstance(raw, list) and len(raw) == 1 and hasattr(raw[0], "get"):
        obj = raw[0]
        try:
            rec_texts = obj.get("rec_texts") or obj.get("texts") or []
            rec_scores = obj.get("rec_scores") or obj.get("scores") or []
            rec_polys = obj.get("rec_polys") or obj.get("rec_boxes") or obj.get("dt_polys") or []
            raw = {"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys, "rec_boxes": rec_polys}
        except Exception:
            try:
                raw = dict(obj)
            except Exception:
                pass
    elif hasattr(raw, "get") and not isinstance(raw, dict):
        try:
            raw = {
                "rec_texts": raw.get("rec_texts", []),
                "rec_scores": raw.get("rec_scores", []),
                "rec_polys": raw.get("rec_polys", []),
                "rec_boxes": raw.get("rec_polys", []),
            }
        except Exception:
            pass
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
                flat: List[List[float]] = []
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
                flat: List[List[float]] = []
                if isinstance(box, (list, tuple)):
                    for pt in box:
                        if isinstance(pt, (list, tuple)):
                            flat.append([float(pt[0]), float(pt[1])])
                lines.append({"text": str(txt), "confidence": conf, "box": flat})
            return lines
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "text" in item:
                    lines.append(
                        {
                            "text": str(item.get("text", "")),
                            "confidence": float(item.get("confidence", 0.0)),
                            "box": item.get("box", []),
                        }
                    )
        return lines
    except Exception:
        return lines


def run_ocr_inference(image_bytes: bytes, lang: str = "fr") -> Dict[str, Any]:
    start = time.perf_counter()
    img = validate_and_decode(image_bytes)
    ocr = get_or_create_ocr(lang)

    # PaddleOCR 3.7: try predict then ocr
    raw = None
    last_err: Optional[Exception] = None
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

    # Device is always cpu for V1
    return {
        "success": True,
        "full_text": full_text,
        "lines": lines,
        "processing_ms": processing_ms,
        "lang": lang,
        "device": "cpu",
    }


def get_health_state() -> Dict[str, Any]:
    """Lightweight health — do NOT run inference."""
    available = _ocr_instance is not None or _ocr_init_error is None
    # If never initialized, we report available=True but not yet loaded
    # (worker can still serve; first request will init). Matches 503 only on init failure.
    if _ocr_init_error:
        return {"status": "ok", "ocr_available": False, "device": "cpu", "error": _ocr_init_error}
    return {"status": "ok", "ocr_available": True, "device": "cpu"}
