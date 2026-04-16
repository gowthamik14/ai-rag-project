from pydantic import Field, model_validator
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

    # Override API_HOST to "127.0.0.1" (or a private IP) in production to
    # avoid exposing the service on all network interfaces.
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Ollama / LLM ──────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:14b"

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

    @model_validator(mode="after")
    def _validate_gcp_credentials(self) -> "Settings":
        """Reject partial service-account configuration early with a clear message.

        Both client_email and private_key must be supplied together.
        If only one is present the credential builder would fail at runtime
        with an opaque error from google-auth.
        """
        has_email = bool(self.gcp_sa_client_email)
        has_key = bool(self.gcp_sa_private_key)
        if has_email and not has_key:
            raise ValueError(
                "GCP_SA_CLIENT_EMAIL is set but GCP_SA_PRIVATE_KEY is missing. "
                "Provide all service-account fields or none."
            )
        if has_key and not has_email:
            raise ValueError(
                "GCP_SA_PRIVATE_KEY is set but GCP_SA_CLIENT_EMAIL is missing. "
                "Provide all service-account fields or none."
            )
        return self


settings = Settings()
