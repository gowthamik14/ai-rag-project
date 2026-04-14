from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.document_loader import BigQueryDocumentLoader, create_model_loader


# ── helpers ────────────────────────────────────────────────────────────────

def make_loader(
    query: str = "SELECT * FROM `p.d.t`",
    page_content_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    bq_documents: list[Document] | None = None,
) -> tuple[BigQueryDocumentLoader, MagicMock]:
    """Return a loader whose internal BigQueryLoader is mocked."""
    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = bq_documents or []
        loader = BigQueryDocumentLoader(
            query=query,
            page_content_columns=page_content_columns,
            metadata_columns=metadata_columns,
        )
    return loader, MockBQ


# ── BigQueryDocumentLoader.load ────────────────────────────────────────────

def test_load_returns_documents() -> None:
    docs = [Document(page_content="repair info", metadata={"id": "1"})]
    loader, _ = make_loader(bq_documents=docs)

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = docs
        result = loader.load()

    assert result == docs


def test_load_returns_empty_list_when_no_rows() -> None:
    loader, _ = make_loader(bq_documents=[])

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        result = loader.load()

    assert result == []


def test_load_passes_query_to_bigquery_loader() -> None:
    sql = "SELECT job_title FROM `p.d.t` WHERE cost < 100"
    loader, _ = make_loader(query=sql)

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["query"] == sql


def test_load_passes_project_from_settings() -> None:
    loader, _ = make_loader()

    with (
        patch("rag.document_loader.BigQueryLoader") as MockBQ,
        patch("rag.document_loader.settings") as mock_settings,
    ):
        mock_settings.gcp_project_id = "my-project"
        mock_settings.gcp_sa_client_email = ""  # skip credential building
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["project"] == "my-project"


def test_load_passes_page_content_columns_when_specified() -> None:
    loader, _ = make_loader(page_content_columns=["job_title", "description"])

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["page_content_columns"] == ["job_title", "description"]


def test_load_passes_none_page_content_columns_when_not_specified() -> None:
    loader, _ = make_loader()

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["page_content_columns"] is None


def test_load_passes_metadata_columns_when_specified() -> None:
    loader, _ = make_loader(metadata_columns=["id", "model"])

    with patch("rag.document_loader.BigQueryLoader") as MockBQ:
        MockBQ.return_value.load.return_value = []
        loader.load()
        _, kwargs = MockBQ.call_args
        assert kwargs["metadata_columns"] == ["id", "model"]


# ── create_model_loader ────────────────────────────────────────────────────

def test_create_model_loader_returns_bigquery_loader() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert isinstance(loader, BigQueryDocumentLoader)


def test_create_model_loader_query_contains_table_name() -> None:
    with patch("rag.document_loader.settings") as mock_settings:
        mock_settings.gcp_project_id = "proj"
        mock_settings.bigquery_dataset = "ds"
        loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "repair_data" in loader._query


def test_create_model_loader_query_contains_project_and_dataset() -> None:
    with patch("rag.document_loader.settings") as mock_settings:
        mock_settings.gcp_project_id = "my-project"
        mock_settings.bigquery_dataset = "my-dataset"
        loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "my-project" in loader._query
    assert "my-dataset" in loader._query


def test_create_model_loader_query_contains_model_value() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "Ford Focus" in loader._query


def test_create_model_loader_query_contains_make_value() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "Ford" in loader._query


def test_create_model_loader_query_uses_lower_for_case_insensitive() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "LOWER" in loader._query


def test_create_model_loader_different_vehicles_produce_different_queries() -> None:
    loader_vauxhall = create_model_loader("CROSSLAND X HATCHBACK", "VAUXHALL", job_cost=150.0)
    loader_bmw = create_model_loader("BMW X5", "BMW", job_cost=150.0)
    assert "CROSSLAND X HATCHBACK" in loader_vauxhall._query
    assert "BMW X5" in loader_bmw._query
    assert loader_vauxhall._query != loader_bmw._query


def test_create_model_loader_sets_repair_description_as_page_content() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert loader._page_content_columns == ["repair_description"]


def test_create_model_loader_includes_job_status_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "jobLineStatus" in loader._metadata_columns


def test_create_model_loader_includes_job_status_date_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "job_status_date" in loader._metadata_columns


def test_create_model_loader_includes_repair_cost_in_metadata() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=150.0)
    assert "repair_cost" in loader._metadata_columns


def test_create_model_loader_query_contains_cost_band_filter() -> None:
    loader = create_model_loader("Ford Focus", "Ford", job_cost=100.0)
    # min = 100 * 0.1 = 10.0, max = 100 * 10.0 = 1000.0
    assert "BETWEEN" in loader._query
    assert "10.0" in loader._query
    assert "1000.0" in loader._query


def test_create_model_loader_cost_band_scales_with_job_cost() -> None:
    loader_cheap = create_model_loader("Ford Focus", "Ford", job_cost=50.0)
    loader_expensive = create_model_loader("Ford Focus", "Ford", job_cost=500.0)
    assert "5.0" in loader_cheap._query    # 50 * 0.1
    assert "5000.0" in loader_expensive._query  # 500 * 10
