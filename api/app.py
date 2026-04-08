"""
FastAPI application factory.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from .routes import (
    chat_router,
    health_router,
    ingest_router,
    query_router,
    route_router,
    summarize_router,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG service powered by GEMMA (local via Ollama) "
            "and Firestore document storage."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)
    app.include_router(chat_router)
    app.include_router(summarize_router)
    app.include_router(route_router)

    return app
