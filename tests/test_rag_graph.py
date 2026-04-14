from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from rag.document_loader import DocumentLoader
from rag.graph import (
    RAGGraph,
    RAGState,
    _COST_TOLERANCE_MULTIPLIER,
    _SIMILARITY_SCORE_BUFFER,
    _is_authorised_status,
    get_rag_graph,
)


# ── test doubles ───────────────────────────────────────────────────────────────

class FakeDocumentLoader(DocumentLoader):
    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or []
        self.load_call_count = 0

    def load(self) -> list[Document]:
        self.load_call_count += 1
        return self.documents


class FakeEmbeddings(Embeddings):
    """Stub embeddings — satisfies the interface without downloading any model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


class FakeVectorStore:
    """Stub vector store — returns documents as-is without calling FAISS.

    All documents are returned with a uniform score of 0.0 so that every chunk
    passes the _SIMILARITY_SCORE_BUFFER filter in _retrieve.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.last_query: str | None = None

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        self.last_query = query
        return [(doc, 0.0) for doc in self._documents[:k]]


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def documents() -> list[Document]:
    return [
        Document(
            page_content="Brake pad replacement",
            metadata={
                "jobLineStatus": "AUTHORISED",
                "job_status_date": "2024-03-15",
                "repair_cost": 120.0,
                "job_status_updated_by_user": "john.smith",
            },
        ),
        Document(
            page_content="Oil and filter change",
            metadata={
                "jobLineStatus": "NOT AUTHORISED",
                "job_status_date": "2024-01-10",
                "repair_cost": 45.0,
                "job_status_updated_by_user": "jane.doe",
            },
        ),
    ]


@pytest.fixture()
def fake_loader(documents: list[Document]) -> FakeDocumentLoader:
    return FakeDocumentLoader(documents=documents)


@pytest.fixture()
def graph(fake_loader: FakeDocumentLoader) -> RAGGraph:
    return RAGGraph(
        loader_factory=lambda model, make, job_cost: fake_loader,
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )


def _base_state(**overrides) -> RAGState:
    """Return a minimal valid RAGState, with optional field overrides."""
    state: RAGState = {
        "question":        "A 'brake pad replacement' repair has been submitted.",
        "job_title":       "brake pad replacement",
        "job_cost":        150.0,
        "make":            "Ford",
        "model":           "Ford Focus",
        "documents":       [],
        "relevant_chunks": [],
        "answer":          "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ── RAGGraph.run ───────────────────────────────────────────────────────────────

def test_run_returns_string(graph: RAGGraph) -> None:
    result = graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford", job_cost=150.0)
    assert isinstance(result, str)


def test_run_returns_valid_json(graph: RAGGraph) -> None:
    result = graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford", job_cost=150.0)
    parsed = json.loads(result)
    assert "reasoning" in parsed
    assert "verdict" in parsed


def test_run_calls_loader_once(graph: RAGGraph, fake_loader: FakeDocumentLoader) -> None:
    graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford", job_cost=150.0)
    assert fake_loader.load_call_count == 1


# ── _load node ─────────────────────────────────────────────────────────────────

def test_load_uses_model_make_and_job_cost_from_state() -> None:
    received: list[tuple[str, str, float]] = []

    def capturing_factory(model: str, make: str, job_cost: float) -> DocumentLoader:
        received.append((model, make, job_cost))
        return FakeDocumentLoader()

    graph = RAGGraph(
        loader_factory=capturing_factory,
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )
    graph._load(_base_state(model="BMW X5", make="BMW", job_cost=200.0))

    assert received == [("BMW X5", "BMW", 200.0)]


def test_load_returns_documents_from_loader(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._load(_base_state())
    assert result["documents"] == documents


def test_load_returns_empty_list_when_no_rows() -> None:
    graph = RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=[]),
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )
    result = graph._load(_base_state(model="Unknown"))
    assert result["documents"] == []


# ── _retrieve node ─────────────────────────────────────────────────────────────

def test_retrieve_uses_job_title_not_full_question(
    documents: list[Document],
) -> None:
    """FAISS search must use the bare job_title, not the verbose question string."""
    captured_store: list[FakeVectorStore] = []

    def capturing_factory(docs, emb):
        store = FakeVectorStore(docs)
        captured_store.append(store)
        return store

    graph = RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=documents),
        embeddings=FakeEmbeddings(),
        vector_store_factory=capturing_factory,
    )
    graph._retrieve(_base_state(
        job_title="brake pad replacement",
        question="A very long verbose question about authorisation policy",
        documents=documents,
    ))

    assert captured_store[0].last_query == "brake pad replacement"


def test_retrieve_returns_relevant_chunks(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._retrieve(_base_state(documents=documents))
    assert isinstance(result["relevant_chunks"], list)
    assert len(result["relevant_chunks"]) > 0


def test_retrieve_returns_empty_when_no_documents(graph: RAGGraph) -> None:
    result = graph._retrieve(_base_state(documents=[]))
    assert result["relevant_chunks"] == []


def test_retrieve_results_are_documents(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._retrieve(_base_state(documents=documents))
    for chunk in result["relevant_chunks"]:
        assert isinstance(chunk, Document)


def test_retrieve_returns_at_most_top_k(graph: RAGGraph) -> None:
    many_docs = [Document(page_content=f"Doc {i}") for i in range(20)]
    result = graph._retrieve(_base_state(documents=many_docs))
    assert len(result["relevant_chunks"]) <= settings.retrieval_top_k


def test_retrieve_filters_out_dissimilar_chunks() -> None:
    """Chunks with score > best_score + _SIMILARITY_SCORE_BUFFER are excluded."""
    similar_doc   = Document(page_content="fire extinguisher", metadata={"jobLineStatus": "AUTHORISED"})
    dissimilar_doc = Document(page_content="1123",             metadata={"jobLineStatus": "AUTHORISED"})

    class ScoreFakeVectorStore:
        def similarity_search_with_score(
            self, query: str, k: int = 5
        ) -> list[tuple[Document, float]]:
            # similar_doc is a good match; dissimilar_doc is clearly beyond the buffer
            return [
                (similar_doc,   0.0),
                (dissimilar_doc, 0.0 + _SIMILARITY_SCORE_BUFFER + 0.01),
            ]

    graph = RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=[similar_doc, dissimilar_doc]),
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: ScoreFakeVectorStore(),
    )
    result = graph._retrieve(_base_state(documents=[similar_doc, dissimilar_doc]))
    assert result["relevant_chunks"] == [similar_doc]


# ── _generate node — no-chunks fallback ───────────────────────────────────────

def test_generate_returns_not_authorised_when_no_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=[]))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "NOT AUTHORISED"


def test_generate_mentions_no_records_when_no_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=[]))
    parsed = json.loads(result["answer"])
    assert "no similar past repair records" in parsed["reasoning"].lower()


# ── _generate node — latest-date-wins logic (within similar records) ──────────

def test_generate_verdict_authorised_when_latest_record_is_authorised(
    graph: RAGGraph,
) -> None:
    """Most recent record is AUTHORISED → verdict is AUTHORISED."""
    chunks = [
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "AUTHORISED",
            "job_status_date": "2024-06-01",
            "job_status_updated_by_user": "alice",
            "repair_cost": 200.0,
        }),
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "NOT AUTHORISED",
            "job_status_date": "2024-05-01",
            "job_status_updated_by_user": "bob",
            "repair_cost": 200.0,
        }),
    ]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "AUTHORISED"


def test_generate_verdict_not_authorised_when_latest_record_is_declined(
    graph: RAGGraph,
) -> None:
    """Most recent record is NOT AUTHORISED → verdict is NOT AUTHORISED."""
    chunks = [
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "NOT AUTHORISED",
            "job_status_date": "2024-06-01",
            "job_status_updated_by_user": "bob",
            "repair_cost": 200.0,
        }),
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "AUTHORISED",
            "job_status_date": "2024-05-01",
            "job_status_updated_by_user": "alice",
            "repair_cost": 200.0,
        }),
    ]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "NOT AUTHORISED"


def test_generate_uses_latest_date_record_not_first_in_list(graph: RAGGraph) -> None:
    """The decision is based on the most recent date, not the list position."""
    chunks = [
        # comes first in the list but has an older date
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "NOT AUTHORISED",
            "job_status_date": "2024-01-01",
            "job_status_updated_by_user": "old-user",
            "repair_cost": 200.0,
        }),
        # newer date — must win
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "AUTHORISED",
            "job_status_date": "2024-12-31",
            "job_status_updated_by_user": "new-user",
            "repair_cost": 200.0,
        }),
    ]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "AUTHORISED"
    assert "new-user" in parsed["reasoning"]


def test_generate_reasoning_includes_job_status_updated_by_user(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "job_status_updated_by_user": "john.smith",
        "repair_cost": 200.0,
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    assert "john.smith" in json.loads(result["answer"])["reasoning"]


def test_generate_reasoning_includes_job_status_date(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-15",
        "job_status_updated_by_user": "alice",
        "repair_cost": 200.0,
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    assert "2024-06-15" in json.loads(result["answer"])["reasoning"]


def test_generate_authorised_reasoning_format(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-15",
        "job_status_updated_by_user": "alice",
        "repair_cost": 200.0,
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning == "Similar job has been authorised by alice on 2024-06-15."


def test_generate_declined_reasoning_format(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "NOT AUTHORISED",
        "job_status_date": "2024-06-15",
        "job_status_updated_by_user": "bob",
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning == "Similar job was declined by bob on 2024-06-15."


def test_generate_omits_user_part_when_missing(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-15",
        "repair_cost": 200.0,
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning == "Similar job has been authorised on 2024-06-15."


def test_generate_omits_date_part_when_missing(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_updated_by_user": "alice",
        "repair_cost": 200.0,
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning == "Similar job has been authorised by alice."


def test_generate_answer_is_valid_json(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._generate(_base_state(relevant_chunks=documents))
    parsed = json.loads(result["answer"])
    assert "reasoning" in parsed
    assert parsed["verdict"] in ("AUTHORISED", "NOT AUTHORISED")


# ── _generate node — plural messaging (multiple similar records) ───────────────

def _two_declined_chunks() -> list[Document]:
    return [
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "NOT AUTHORISED",
            "job_status_date": "2024-06-01",
            "job_status_updated_by_user": "allen.smith",
        }),
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "NOT AUTHORISED",
            "job_status_date": "2024-03-01",
            "job_status_updated_by_user": "bob",
        }),
    ]


def _two_authorised_chunks() -> list[Document]:
    return [
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "AUTHORISED",
            "job_status_date": "2024-06-01",
            "job_status_updated_by_user": "allen.smith",
            "repair_cost": 200.0,
        }),
        Document(page_content="windscreen", metadata={
            "jobLineStatus": "AUTHORISED",
            "job_status_date": "2024-03-01",
            "job_status_updated_by_user": "bob",
            "repair_cost": 200.0,
        }),
    ]


def test_generate_plural_declined_message_when_multiple_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=_two_declined_chunks()))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning.startswith("Similar jobs were declined.")


def test_generate_plural_declined_includes_latest_user_and_date(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=_two_declined_chunks()))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert "allen.smith" in reasoning
    assert "2024-06-01" in reasoning


def test_generate_plural_authorised_message_when_multiple_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=_two_authorised_chunks()))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert reasoning.startswith("Similar jobs have been authorised.")


def test_generate_plural_authorised_includes_latest_user_and_date(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=_two_authorised_chunks()))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert "allen.smith" in reasoning
    assert "2024-06-01" in reasoning


def test_generate_singular_message_when_single_chunk_declined(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "NOT AUTHORISED",
        "job_status_date": "2024-06-15",
        "job_status_updated_by_user": "bob",
    })]
    reasoning = json.loads(graph._generate(_base_state(relevant_chunks=chunks))["answer"])["reasoning"]
    assert reasoning == "Similar job was declined by bob on 2024-06-15."


def test_generate_singular_message_when_single_chunk_authorised(graph: RAGGraph) -> None:
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-15",
        "job_status_updated_by_user": "alice",
        "repair_cost": 200.0,
    })]
    reasoning = json.loads(graph._generate(_base_state(relevant_chunks=chunks))["answer"])["reasoning"]
    assert reasoning == "Similar job has been authorised by alice on 2024-06-15."


# ── _generate node — cost-ratio guard ────────────────────────────────────────

def test_generate_not_authorised_when_cost_exceeds_tolerance(graph: RAGGraph) -> None:
    """Submitted cost > historical * _COST_TOLERANCE_MULTIPLIER → NOT AUTHORISED."""
    historical = 9999.0
    submitted  = historical * _COST_TOLERANCE_MULTIPLIER + 0.01  # just over the limit
    chunks = [Document(page_content="fire extinguisher", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "repair_cost": historical,
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=submitted, relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "NOT AUTHORISED"


def test_generate_cost_exceeded_reasoning_mentions_both_costs(graph: RAGGraph) -> None:
    chunks = [Document(page_content="fire extinguisher", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "repair_cost": 9999.0,
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=50000.0, relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert "50000.00" in reasoning
    assert "9999.00" in reasoning


def test_generate_cost_exceeded_reasoning_mentions_user_and_date(graph: RAGGraph) -> None:
    chunks = [Document(page_content="fire extinguisher", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "repair_cost": 9999.0,
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=50000.0, relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert "alice" in reasoning
    assert "2024-06-01" in reasoning


def test_generate_authorised_when_cost_within_tolerance(graph: RAGGraph) -> None:
    """Submitted cost at exactly 2× historical is still authorised."""
    historical = 9999.0
    submitted  = historical * _COST_TOLERANCE_MULTIPLIER  # exactly at the limit
    chunks = [Document(page_content="fire extinguisher", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "repair_cost": historical,
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=submitted, relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "AUTHORISED"


def test_generate_not_authorised_when_repair_cost_missing_and_job_cost_positive(graph: RAGGraph) -> None:
    """If the historical record has no repair_cost and job_cost > 0, cost cannot be verified → NOT AUTHORISED."""
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "job_status_updated_by_user": "alice",
        # no repair_cost key
    })]
    result = graph._generate(_base_state(job_cost=999999.0, relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "NOT AUTHORISED"


def test_generate_not_authorised_reasoning_mentions_missing_cost(graph: RAGGraph) -> None:
    """Reasoning should explain that the historical cost is missing."""
    chunks = [Document(page_content="fire extinguisher", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=50000.0, relevant_chunks=chunks))
    reasoning = json.loads(result["answer"])["reasoning"]
    assert "no historical repair cost" in reasoning.lower()


def test_generate_authorised_when_repair_cost_missing_and_job_cost_zero(graph: RAGGraph) -> None:
    """If job_cost is 0, the cost check is bypassed even when repair_cost is absent."""
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "job_status_updated_by_user": "alice",
        # no repair_cost key
    })]
    result = graph._generate(_base_state(job_cost=0.0, relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "AUTHORISED"


def test_generate_skips_cost_check_when_job_cost_is_zero(graph: RAGGraph) -> None:
    """A zero job_cost request bypasses the cost comparison."""
    chunks = [Document(page_content="windscreen", metadata={
        "jobLineStatus": "AUTHORISED",
        "job_status_date": "2024-06-01",
        "repair_cost": 9999.0,
        "job_status_updated_by_user": "alice",
    })]
    result = graph._generate(_base_state(job_cost=0.0, relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert parsed["verdict"] == "AUTHORISED"


# ── _is_authorised_status ──────────────────────────────────────────────────────

def test_is_authorised_status_authorised_uppercase() -> None:
    assert _is_authorised_status("AUTHORISED") is True


def test_is_authorised_status_authorised_mixed_case() -> None:
    assert _is_authorised_status("Authorised") is True


def test_is_authorised_status_not_authorised() -> None:
    assert _is_authorised_status("NOT AUTHORISED") is False


def test_is_authorised_status_declined() -> None:
    assert _is_authorised_status("Declined") is False


def test_is_authorised_status_empty() -> None:
    assert _is_authorised_status("") is False


def test_is_authorised_status_unknown() -> None:
    assert _is_authorised_status("PENDING") is False


# ── get_rag_graph singleton ────────────────────────────────────────────────────

def test_get_rag_graph_returns_same_instance() -> None:
    get_rag_graph.cache_clear()
    a = get_rag_graph()
    b = get_rag_graph()
    assert a is b
    get_rag_graph.cache_clear()
