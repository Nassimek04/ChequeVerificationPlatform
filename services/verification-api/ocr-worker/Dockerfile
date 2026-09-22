# PaddleOCR Docker Linux Worker V1 — CPU only
# Python 3.11 chosen: paddlepaddle 3.0.0 supports 3.8-3.11 (not 3.13)
# Base: python:3.11-slim-bookworm for stability and small size

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLAGS_enable_pir_api=0 \
    FLAGS_enable_pir_in_executor=0

# Minimal OS deps for opencv / paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker source
COPY app/ ./app/

# Create non-root user (optional but production-reasonable)
RUN useradd -m -u 1000 worker && chown -R worker:worker /app
# Paddle model cache location: /home/worker/.paddlex (also /root/.paddlex for compat)
# Ensure writable
RUN mkdir -p /home/worker/.paddlex && chown -R worker:worker /home/worker

USER worker

EXPOSE 8000

# Run via uvicorn — single worker process keeps model in memory (lazy init on first request)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
