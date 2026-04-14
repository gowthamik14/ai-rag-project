from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from langchain_community.document_loaders import BigQueryLoader
from langchain_core.documents import Document

from config.settings import settings


_BIGQUERY_SCOPES = ["https://www.googleapis.com/auth/bigquery"]


def _build_credentials() -> Optional[Credentials]:
    """Build service-account credentials from inline env vars.

    Returns None when GCP_SA_CLIENT_EMAIL is not set, which causes
    BigQueryLoader to fall back to Application Default Credentials (ADC).
    """
    if not settings.gcp_sa_client_email:
        return None

    info = {
        "type": settings.gcp_sa_type,
        "project_id": settings.gcp_project_id,
        "private_key_id": settings.gcp_sa_private_key_id,
        # python-dotenv interprets \n in double-quoted values; replace handles
        # any environment that loads the key as a raw escaped string instead.
        "private_key": settings.gcp_sa_private_key.replace("\\n", "\n"),
        "client_email": settings.gcp_sa_client_email,
        "client_id": settings.gcp_sa_client_id,
        "auth_uri": settings.gcp_sa_auth_uri,
        "token_uri": settings.gcp_sa_token_uri,
        "auth_provider_x509_cert_url": settings.gcp_sa_auth_provider_cert_url,
        "client_x509_cert_url": settings.gcp_sa_client_cert_url,
        "universe_domain": settings.gcp_universe_domain,
    }
    return service_account.Credentials.from_service_account_info(
        info, scopes=_BIGQUERY_SCOPES
    )


class DocumentLoader(ABC):
    """Abstract contract for loading source data as LangChain Documents.

    Each row / record becomes one Document:
      - page_content : the text the LLM will read
      - metadata     : structured fields (id, source, etc.)
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return a list of Documents."""


class BigQueryDocumentLoader(DocumentLoader):
    """Loads rows from a BigQuery table as LangChain Documents.

    Uses LangChain's built-in BigQueryLoader so there is no hand-rolled
    BigQuery client code — LangChain handles the connection and mapping.

    Args:
        query:                 Full SQL query to run.
        page_content_columns:  Columns whose values are joined into page_content.
                               When None, all columns are used.
        metadata_columns:      Columns to store in Document.metadata.
                               When None, all columns not in page_content are used.
    """

    def __init__(
        self,
        query: str,
        page_content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> None:
        self._query = query
        self._page_content_columns = page_content_columns
        self._metadata_columns = metadata_columns

    def load(self) -> list[Document]:
        loader = BigQueryLoader(
            query=self._query,
            project=settings.gcp_project_id,
            credentials=_build_credentials(),
            page_content_columns=self._page_content_columns,
            metadata_columns=self._metadata_columns,
        )
        return loader.load()


def create_model_loader(model: str, make: str, job_cost: float) -> DocumentLoader:
    """Return a loader that fetches repair history for a specific vehicle make/model.

    - page_content  → repair_description (what FAISS embeds for similarity search)
    - metadata      → car_make, car_model, car_variant, jobLineStatus, job_status_date, repair_cost

    Args:
        model:    Vehicle model name (e.g. "Ford Focus").
        make:     Vehicle make (e.g. "Ford").
    """

    query = (
        f"SELECT car_make, car_model, car_variant, repair_description, jobLineStatus, job_status_date, repair_cost, job_status_updated_by_user "
        f"FROM `{settings.gcp_project_id}.{settings.bigquery_dataset}.repair_data` "
        f"WHERE LOWER(car_make) = LOWER('{make}') AND LOWER(car_model) = LOWER('{model}') "
        f"ORDER BY job_status_date DESC"
    )
    return BigQueryDocumentLoader(
        query=query,
        page_content_columns=["repair_description"],
        metadata_columns=["car_make", "car_model", "car_variant", "jobLineStatus", "job_status_date", "repair_cost", "job_status_updated_by_user"],
    )
