# PaddleOCR Docker Linux Worker V1 — Architecture

## Why Docker

Windows native PaddlePaddle `paddle/base/libpaddle.pyd` is blocked by Windows Smart App Control (`DLL load failed while importing libpaddle: Une stratégie de contrôle d’application a bloqué ce fichier`). Importing `paddleocr` works with `chardet==5.2.0`, but real inference still loads `libpaddle.pyd` and fails.

Docker isolates PaddleOCR to Linux, avoids mixing PaddlePaddle with PyTorch CUDA (torch 2.8.0+cu128 stays on Windows), and bypasses Smart App Control without disabling any security policy.

This is a **deployment/runtime decision**; functional requirements are unchanged.

## Final Architecture

```
ASP.NET Core MVC (Windows, :5285)
        |
        | POST /api/cheques/ocr  (multipart, server-side chequeId only)
        v
Main FastAPI service (Windows, 127.0.0.1:8000)
  - Validates image (OpenCV decode, 10 MB)
  - Moroccan field parsing (MAD/DH/DHS/DIRHAM, date, account)
        |
        | HTTP 127.0.0.1:8010  (httpx multipart, timeout 60s)
        v
PaddleOCR Worker (Linux Docker, 0.0.0.0:8000 inside)
  - FastAPI + PaddleOCR 3.7.0 + paddlepaddle 3.2.2 CPU
  - Lazy model init on first request, persistent reuse
  - Returns normalized {success, full_text, lines[], processing_ms, lang, device}
        |
        v
Main FastAPI normalizes again + returns to ASP.NET
```

- ASP.NET **never** calls Docker directly; contract `POST /api/cheques/ocr` is preserved (DTOs, `VerificationApiClient`, `VerificationService`, Razor UI unchanged).
- Worker does **only** OCR: no torch, no AI V2, no signature comparison, no DB.

## Technology Choices

- **Python 3.11-slim-bookworm**: paddlepaddle 3.x officially supports 3.8-3.11; Python 3.13 (Windows env) is not supported by paddle Linux wheels. 3.11 maximizes compatibility.
- **paddlepaddle==3.2.2 CPU**: fixes PIR `strides is not right` bug present in 3.0.0 under Paddlex 3.7.2 pipeline. 3.2.2 stable on manylinux.
- **paddleocr==3.7.0, paddlex==3.7.2**: latest stable OCR pipeline (PP-OCRv6_medium_det/rec).
- **CPU first**: no `nvidia-docker`, no CUDA; academic prototype workload is small.
- **Pins**: `fastapi==0.115.12 uvicorn==0.34.0 python-multipart==0.0.9 opencv-contrib-python==4.10.0.84 numpy==1.26.4 Pillow==10.4.0 chardet==5.2.0 shapely==2.0.6 pyclipper==1.3.0.post6 huggingface-hub` — all pinned, `pip install -r requirements.txt` recreates without manual steps.

## Endpoints

- Worker `GET /health` → `{status:"ok", ocr_available:true, device:"cpu"}` lightweight, no inference.
- Worker `POST /ocr` multipart `file` (+ `lang=fr`) → `{success, message, full_text, lines:[{text, confidence, box}], processing_ms, lang, device}`.
- Main `GET /api/cheques/ocr-status` → proxies health; `POST /api/cheques/ocr` → proxies bytes + re-parses fields.

## Lazy Initialization

Model is created once (`app.ocr_service.get_or_create_ocr`) behind a `Lock`. First request pays model download (if cold cache) + init (~1.9s CPU) + inference; subsequent requests reuse instance (≈2s vs 15s cold).

## Model Cache

Docker volume `ocr-model-cache:/home/worker/.paddlex` persists `PP-OCRv6_*`, `PP-LCNet_*`, `UVDoc`. Container recreation does NOT re-download. Models are NOT committed to Git nor stored under `wwwroot`.

## Security / Exposure

- Compose binds `127.0.0.1:8010:8000` — reachable from Windows host only, not LAN.
- Worker validates file non-empty, ≤10 MB, image decode success, temp file via `tmp` (worker uses in-memory `cv2.imdecode`, no original filename).
- Logs: request success/failure, duration, lang, line count only; no image/text/sensitive cheque content.
- No Windows security modification.

## Moroccan Parsing

Kept in **main FastAPI** (`cheque_ocr_service._extract_fields`): MAD/DH/DHS/DIRHAM(S) only (no DA/DZD), date `DD/MM/YYYY` normalized, amount near currency token, account via COMPTE/RIB next-line handling. Worker only does recognition.

## Language

V1 French/Latin (`lang=fr`). Arabic model not configured — documented unsupported.

## Commands

```powershell
# Build & start
docker compose -f docker-compose.ocr.yml up -d --build

# Health
Invoke-RestMethod http://127.0.0.1:8010/health
Invoke-RestMethod http://127.0.0.1:8000/api/cheques/ocr-status

# Logs
docker compose -f docker-compose.ocr.yml logs -f

# Stop
docker compose -f docker-compose.ocr.yml down
```

## Legacy

`ocr-worker/.venv-ocr` retained but **not used** at runtime. See `README_LEGACY.md`.
