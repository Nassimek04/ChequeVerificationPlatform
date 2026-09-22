"""Verification API application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    cheques_ocr,
    health,
    images,
    signatures,
    signatures_ai,
    signatures_compare,
    signatures_debug,
)
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(images.router)
    app.include_router(signatures.router)
    app.include_router(signatures_compare.router)
    app.include_router(signatures_ai.router)
    app.include_router(cheques_ocr.router)

    # Development-only: the ROI calibration diagnostic endpoint is NOT
    # registered outside the development environment (requests then get 404).
    if settings.app_env == "development":
        app.include_router(signatures_debug.router)

    return app


app = create_app()