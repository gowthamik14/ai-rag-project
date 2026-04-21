from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.document_loader import (
    BigQueryDocumentLoader,
    _build_credentials,
    _validate_sql_string,
    create_model_loader,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_loader(
    query: str = "SELECT * FROM `p.d.t`",
    page_content_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
) -> BigQueryDocumentLoader:
    """Return a loader with credentials mocked out (credentials are built in __init__)."""
    with patch("rag.document_loader._build_credentials", return_value=None):
        return BigQueryDocumentLoader(
            query=query,
            page_content_columns=page_content_columns,
            metadata_columns=metadata_columns,
        )


# ── BigQueryDocumentLoader.load ────────────────────────────────────────────

def test_load_returns_documents() -> None:
    docs = [Document(page_content="repair info", metadata={"id": "1"})]
    loader = make_loader()

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = docs
        result = loader.load()

    assert result == docs


def test_load_returns_empty_list_when_no_rows() -> None:
    loader = make_loader()

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        result = loader.load()

    assert result == []


def test_load_passes_query_to_bigquery_loader() -> None:
    sql = "SELECT job_title FROM `p.d.t` WHERE cost < 100"
    loader = make_loader(query=sql)

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["query"] == sql


def test_load_passes_project_from_settings() -> None:
    loader = make_loader()

    with (
        patch("rag.document_loader.BigQueryLoader") as MockBQ,
        patch("rag.document_loader.settings") as mock_settings,
    ):
        mock_settings.gcp_project_id = "my-project"
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["project"] == "my-project"


def test_load_passes_page_content_columns_when_specified() -> None:
    loader = make_loader(page_content_columns=["job_title", "description"])

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["page_content_columns"] == ["job_title", "description"]


def test_load_passes_none_page_content_columns_when_not_specified() -> None:
    loader = make_loader()

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["page_content_columns"] is None


def test_load_passes_metadata_columns_when_specified() -> None:
    loader = make_loader(metadata_columns=["id", "model"])

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["metadata_columns"] == ["id", "model"]


# ── create_model_loader ────────────────────────────────────────────────────

def test_create_model_loader_returns_bigquery_loader() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert isinstance(loader, BigQueryDocumentLoader)


def test_create_model_loader_query_contains_table_name() -> None:
    with patch("rag.document_loader.settings") as mock_settings:
        mock_settings.gcp_project_id = "proj"
        mock_settings.bigquery_dataset = "ds"
        mock_settings.gcp_sa_client_email = ""  # prevent _build_credentials from running
        loader = create_model_loader("Ford Focus", "Ford")
    assert "repair_data" in loader._query


def test_create_model_loader_query_contains_project_and_dataset() -> None:
    with patch("rag.document_loader.settings") as mock_settings:
        mock_settings.gcp_project_id = "my-project"
        mock_settings.bigquery_dataset = "my-dataset"
        mock_settings.gcp_sa_client_email = ""  # prevent _build_credentials from running
        loader = create_model_loader("Ford Focus", "Ford")
    assert "my-project" in loader._query
    assert "my-dataset" in loader._query


def test_create_model_loader_query_contains_model_value() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "Ford Focus" in loader._query


def test_create_model_loader_query_contains_make_value() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "Ford" in loader._query


def test_create_model_loader_query_uses_lower_for_case_insensitive() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "LOWER" in loader._query


def test_create_model_loader_different_vehicles_produce_different_queries() -> None:
    loader_vauxhall = create_model_loader("CROSSLAND X HATCHBACK", "VAUXHALL", )
    loader_bmw = create_model_loader("BMW X5", "BMW", )
    assert "CROSSLAND X HATCHBACK" in loader_vauxhall._query
    assert "BMW X5" in loader_bmw._query
    assert loader_vauxhall._query != loader_bmw._query


def test_create_model_loader_sets_repair_description_as_page_content() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert loader._page_content_columns == ["repair_description"]


def test_create_model_loader_includes_job_status_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "jobLineStatus" in loader._metadata_columns


def test_create_model_loader_includes_job_status_date_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "job_status_date" in loader._metadata_columns


def test_create_model_loader_includes_repair_cost_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford")
    assert "repair_cost" in loader._metadata_columns



# ── _build_credentials ─────────────────────────────────────────────────────


def test_build_credentials_returns_none_when_no_client_email() -> None:
    with patch("rag.document_loader.settings") as mock_settings:
        mock_settings.gcp_sa_client_email = ""
        result = _build_credentials()
    assert result is None


def test_build_credentials_returns_credential_object_when_email_set() -> None:
    with (
        patch("rag.document_loader.settings") as mock_settings,
        patch("rag.document_loader.service_account.Credentials.from_service_account_info") as mock_build,
    ):
        mock_settings.gcp_sa_client_email = "svc@project.iam.gserviceaccount.com"
        mock_settings.gcp_sa_type = "service_account"
        mock_settings.gcp_sa_project_id = "proj"
        mock_settings.gcp_sa_private_key_id = "kid"
        mock_settings.gcp_sa_private_key = "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n"
        mock_settings.gcp_sa_client_id = "123"
        mock_settings.gcp_sa_auth_uri = "https://accounts.google.com/o/oauth2/auth"
        mock_settings.gcp_sa_token_uri = "https://oauth2.googleapis.com/token"
        mock_settings.gcp_sa_auth_provider_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
        mock_settings.gcp_sa_client_cert_url = ""
        mock_settings.gcp_universe_domain = "googleapis.com"
        mock_settings.gcp_project_id = "proj"
        mock_build.return_value = object()

        result = _build_credentials()

    assert result is mock_build.return_value
    mock_build.assert_called_once()


# ── _validate_sql_string ───────────────────────────────────────────────────


def test_validate_sql_string_accepts_alphanumeric_and_spaces() -> None:
    assert _validate_sql_string("Ford Focus", "model") == "Ford Focus"


def test_validate_sql_string_accepts_hyphens() -> None:
    assert _validate_sql_string("Alfa-Romeo", "make") == "Alfa-Romeo"


def test_validate_sql_string_rejects_single_quote() -> None:
    with pytest.raises(ValueError, match="make"):
        _validate_sql_string("O'Hara", "make")


def test_validate_sql_string_rejects_semicolon() -> None:
    with pytest.raises(ValueError, match="model"):
        _validate_sql_string("Focus; DROP TABLE repair_data--", "model")


def test_validate_sql_string_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_sql_string("", "make")
