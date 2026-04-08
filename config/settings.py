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
    # ------------------------------------------------------------------ #
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
    # GEMMA (via Ollama)
    # ------------------------------------------------------------------ #
    ollama_base_url: str = "http://localhost:11434"
    gemma_model: str = "gemma3:4b"           # model tag as listed in `ollama list`
    gemma_temperature: float = 0.7
    gemma_max_tokens: int = 2048
    gemma_timeout: int = 120                  # seconds

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    embedding_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
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

# Make sure the FAISS data directory exists at import time
Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
