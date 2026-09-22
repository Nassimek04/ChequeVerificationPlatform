# Legacy Windows venv-ocr

`ocr-worker/.venv-ocr` is **legacy / local experimentation only**.

It was used for native Windows PaddleOCR via:

```
ocr-worker/.venv-ocr/Scripts/python.exe ocr_worker.py --image <tmp> --lang fr
```

Windows Smart App Control blocks `paddle/base/libpaddle.pyd` (`DLL load failed while importing libpaddle`), therefore native Windows PaddleOCR is NOT used at runtime anymore.

**Do NOT delete this folder automatically** — kept for reference.

**Runtime path (V1):**

```
ASP.NET -> Main FastAPI (Windows) POST /api/cheques/ocr -> Docker Linux worker http://127.0.0.1:8010/ocr -> PaddleOCR CPU
```

Docker worker location: `services/verification-api/ocr-worker/` (Dockerfile + app/).
