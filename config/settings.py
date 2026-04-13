from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_name: str = "RAG Service"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Ollama / LLM ──────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma4:latest"

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 5

    # ── BigQuery ──────────────────────────────────────────────────────────
    gcp_project_id: str = Field(default="", description="GCP project that hosts BigQuery.")
    bigquery_dataset: str = Field(default="", description="BigQuery dataset name.")

    # Inline service-account credentials (set these OR use gcp_credentials_path)
    gcp_sa_type: str = Field(default="service_account")
    gcp_sa_private_key_id: str = Field(default="")
    gcp_sa_private_key: str = Field(default="")
    gcp_sa_client_email: str = Field(default="")
    gcp_sa_client_id: str = Field(default="")
    gcp_sa_auth_uri: str = Field(default="https://accounts.google.com/o/oauth2/auth")
    gcp_sa_token_uri: str = Field(default="https://oauth2.googleapis.com/token")
    gcp_sa_auth_provider_cert_url: str = Field(default="https://www.googleapis.com/oauth2/v1/certs")
    gcp_sa_client_cert_url: str = Field(default="")
    gcp_universe_domain: str = Field(default="googleapis.com")


settings = Settings()
