"""PaddleOCR Docker Linux worker — FastAPI."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.ocr_service import (
    MAX_OCR_FILE_SIZE_BYTES,
    get_health_state,
    run_ocr_inference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}

app = FastAPI(title="paddleocr-worker", version="1.0.0")


@app.get("/health")
async def health():
    # Lightweight — no inference
    state = get_health_state()
    return state


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...), lang: str = "fr"):
    # Validate content-type if provided (not trusted alone)
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        # Still allow if extension is image; strict 415
        logger.warning("OCR worker rejected content type: %s", file.content_type)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Type de contenu non supporté. Formats acceptés : image/jpeg, image/png.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Le fichier est vide.")
    if len(content) > MAX_OCR_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la taille maximale autorisée (10 Mo).")

    # Language validation — V1 only fr/latin; reject empty
    lang = (lang or "fr").strip().lower()
    if lang not in ("fr", "en", "latin"):
        lang = "fr"

    start = time.perf_counter()
    try:
        result = run_ocr_inference(content, lang=lang)
    except ValueError as exc:
        # Image validation error → 400
        logger.warning("OCR image error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("OCR inference failed")
        # Do NOT expose stack / paths
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'OCR.") from exc

    # Do NOT log image bytes or full cheque text
    logger.info(
        "OCR success lang=%s lines=%s processing_ms=%s",
        result["lang"],
        len(result["lines"]),
        result["processing_ms"],
    )
    return JSONResponse(
        content={
            "success": True,
            "message": "OCR completed.",
            "full_text": result["full_text"],
            "lines": result["lines"],
            "processing_ms": result["processing_ms"],
            "lang": result["lang"],
            "device": result["device"],
        }
    )


@app.get("/")
async def root():
    return {"status": "ok", "service": "paddleocr-worker"}
