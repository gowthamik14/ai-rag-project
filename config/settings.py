"""
Central configuration via environment variables + .env file.
All tuneable knobs live here so nothing is hard-coded elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # App
    # ------------------------------------------------------------------ #
    app_name: str = "AI RAG Service"
    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ------------------------------------------------------------------ #
    # Firestore
    # Set FIRESTORE_ENABLED=false to run without Firestore (local-only mode).
    # Sessions and document persistence will be disabled but Q&A still works.
    # ------------------------------------------------------------------ #
    firestore_enabled: bool = True
    google_application_credentials: str = Field(
        default="config/firestore_service_account.json",
        description="Path to GCP service-account JSON file.",
    )
    firestore_project_id: str = Field(
        default="your-gcp-project-id",
        description="GCP project that hosts Firestore.",
    )
    firestore_documents_collection: str = "rag_documents"
    firestore_chunks_collection: str = "rag_chunks"
    firestore_sessions_collection: str = "rag_sessions"

    # ------------------------------------------------------------------ #
    # LLM via Ollama  (default: Qwen)
    # Start Ollama:  OLLAMA_NO_GPU=1 OLLAMA_LLM_LIBRARY=cpu ollama serve
    # Pull model:    ollama pull qwen:latest
    # ------------------------------------------------------------------ #
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen:latest"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout: int = 180                   # seconds — CPU mode is slower

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ------------------------------------------------------------------ #
    # Vector store  (FAISS – local, persisted to disk)
    # ------------------------------------------------------------------ #
    faiss_index_path: str = "data/faiss_index"
    faiss_index_file: str = "index.faiss"
    faiss_metadata_file: str = "metadata.json"
    top_k_retrieval: int = 5

    # ------------------------------------------------------------------ #
    # Text splitting
    # ------------------------------------------------------------------ #
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False


settings = Settings()

# Auto-disable Firestore if credentials file contains placeholder values
def _creds_are_placeholder() -> bool:
    path = settings.google_application_credentials
    if not os.path.exists(path):
        return True
    try:
        import json
        with open(path) as f:
            data = json.load(f)
        return data.get("private_key", "") in ("REPLACE_ME", "", None)
    except Exception:
        return True

if settings.firestore_enabled and _creds_are_placeholder():
    # Silently switch to local-only mode instead of crashing
    object.__setattr__(settings, "firestore_enabled", False)

# Make sure the FAISS data directory exists at import time
Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
