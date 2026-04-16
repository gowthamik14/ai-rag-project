from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from rag.document_loader import DocumentLoader
from rag.graph import (
    RAGGraph,
    RAGState,
    _SIMILARITY_SCORE_BUFFER,
    _evaluate_repair,
    _parse_llm_response,
    _to_iso_date_str,
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
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 8


class FakeVectorStore:
    """Returns all documents with score 0.0 so every chunk passes the similarity filter."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.last_query: str | None = None

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        self.last_query = query
        return [(doc, 0.0) for doc in self._documents[:k]]


class FakeLLMService:
    """Returns a controlled JSON response without calling Ollama."""

    def __init__(
        self,
        reply: str = '{"reasoning": "Approved based on history."}',
    ) -> None:
        self.reply = reply
        self.received_message: str | None = None
        self.call_count = 0

    def chat(self, message: str) -> str:
        self.received_message = message
        self.call_count += 1
        return self.reply


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def documents() -> list[Document]:
    return [
        Document(
            page_content="50k mile service",
            metadata={
                "jobLineStatus": "AUTHORISED",
                "job_status_date": "2024-12-01",
                "repair_cost": 45000.0,
                "job_status_updated_by_user": "john.smith",
            },
        ),
        Document(
            page_content="50k mile service",
            metadata={
                "jobLineStatus": "DECLINED",
                "job_status_date": "2024-06-01",
                "repair_cost": 50000.0,
                "job_status_updated_by_user": "jane.doe",
            },
        ),
        Document(
            page_content="50k mile service",
            metadata={
                "jobLineStatus": "AUTHORISED",
                "job_status_date": "2023-11-01",
                "repair_cost": 42000.0,
                "job_status_updated_by_user": "john.smith",
            },
        ),
    ]


@pytest.fixture()
def fake_loader(documents: list[Document]) -> FakeDocumentLoader:
    return FakeDocumentLoader(documents=documents)


@pytest.fixture()
def fake_llm() -> FakeLLMService:
    return FakeLLMService()


@pytest.fixture()
def graph(fake_loader: FakeDocumentLoader, fake_llm: FakeLLMService) -> RAGGraph:
    return RAGGraph(
        loader_factory=lambda model, make, job_cost: fake_loader,
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
        llm_service=fake_llm,
    )


def _base_state(**overrides) -> RAGState:
    state: RAGState = {
        "question":        "A '50k mile service' repair has been submitted.",
        "job_title":       "50k mile service",
        "job_cost":        44000.0,
        "make":            "Vauxhall",
        "model":           "Crossland X",
        "documents":       [],
        "relevant_chunks": [],
        "answer":          "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _make_graph(
    llm: FakeLLMService | None = None,
    docs: list[Document] | None = None,
) -> RAGGraph:
    return RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=docs or []),
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda chunks, emb: FakeVectorStore(chunks),
        llm_service=llm or FakeLLMService(),
    )


# ── RAGGraph.run ───────────────────────────────────────────────────────────────

def test_run_returns_string(graph: RAGGraph) -> None:
    result = graph.run("question", job_title="50k mile service", model="Crossland X", make="Vauxhall", job_cost=44000.0)
    assert isinstance(result, str)


def test_run_returns_valid_json(graph: RAGGraph) -> None:
    result = graph.run("question", job_title="50k mile service", model="Crossland X", make="Vauxhall", job_cost=44000.0)
    parsed = json.loads(result)
    assert "reasoning" in parsed
    assert "verdict" in parsed


def test_run_calls_loader_once(graph: RAGGraph, fake_loader: FakeDocumentLoader) -> None:
    graph.run("question", job_title="50k mile service", model="Crossland X", make="Vauxhall", job_cost=44000.0)
    assert fake_loader.load_call_count == 1


def test_run_returns_not_authorised_when_no_documents() -> None:
    graph = _make_graph(docs=[])
    result = graph.run(
        "question", job_title="50k mile service", model="Crossland X", make="Vauxhall", job_cost=44000.0
    )
    assert json.loads(result)["verdict"] == "NOT AUTHORISED"


def test_run_not_authorised_json_includes_reasoning_when_no_documents() -> None:
    graph = _make_graph(docs=[])
    result = graph.run(
        "question", job_title="50k mile service", model="Crossland X", make="Vauxhall", job_cost=44000.0
    )
    assert "reasoning" in json.loads(result)


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
        llm_service=FakeLLMService(),
    )
    graph._load(_base_state(model="Crossland X", make="Vauxhall", job_cost=44000.0))

    assert received == [("Crossland X", "Vauxhall", 44000.0)]


def test_load_returns_documents_from_loader(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._load(_base_state())
    assert result["documents"] == documents


def test_load_returns_empty_list_when_no_rows() -> None:
    graph = _make_graph(docs=[])
    result = graph._load(_base_state(model="Unknown"))
    assert result["documents"] == []


# ── _retrieve node ─────────────────────────────────────────────────────────────

def test_retrieve_uses_job_title_not_full_question(
    documents: list[Document],
) -> None:
    captured_store: list[FakeVectorStore] = []

    def capturing_factory(docs, emb):
        store = FakeVectorStore(docs)
        captured_store.append(store)
        return store

    graph = RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=documents),
        embeddings=FakeEmbeddings(),
        vector_store_factory=capturing_factory,
        llm_service=FakeLLMService(),
    )
    graph._retrieve(_base_state(
        job_title="50k mile service",
        question="A very long verbose question about authorisation policy",
        documents=documents,
    ))

    assert captured_store[0].last_query == "50k mile service"


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
    similar_doc    = Document(page_content="fire extinguisher", metadata={"jobLineStatus": "AUTHORISED"})
    dissimilar_doc = Document(page_content="1123",              metadata={"jobLineStatus": "AUTHORISED"})

    class ScoreFakeVectorStore:
        def similarity_search_with_score(
            self, query: str, k: int = 5
        ) -> list[tuple[Document, float]]:
            return [
                (similar_doc,    0.0),
                (dissimilar_doc, 0.0 + _SIMILARITY_SCORE_BUFFER + 0.01),
            ]

    graph = RAGGraph(
        loader_factory=lambda m, mk, jc: FakeDocumentLoader(documents=[similar_doc, dissimilar_doc]),
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: ScoreFakeVectorStore(),
        llm_service=FakeLLMService(),
    )
    result = graph._retrieve(_base_state(documents=[similar_doc, dissimilar_doc]))
    assert result["relevant_chunks"] == [similar_doc]


# ── _generate node — no-chunks fallback ───────────────────────────────────────

def test_generate_returns_not_authorised_when_no_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=[]))
    assert json.loads(result["answer"])["verdict"] == "NOT AUTHORISED"


def test_generate_mentions_no_records_when_no_chunks(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=[]))
    assert "no similar past repair records" in json.loads(result["answer"])["reasoning"].lower()


def test_generate_no_chunks_includes_last_updated_fields(graph: RAGGraph) -> None:
    result = graph._generate(_base_state(relevant_chunks=[]))
    parsed = json.loads(result["answer"])
    assert "last_updated_by" in parsed
    assert "last_updated_date" in parsed


def test_generate_does_not_call_llm_when_no_chunks(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    graph._generate(_base_state(relevant_chunks=[]))
    assert fake_llm.call_count == 0


# ── _generate node — LLM called with all records ──────────────────────────────

def test_generate_calls_llm_exactly_once(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    graph._generate(_base_state(relevant_chunks=chunks))
    assert fake_llm.call_count == 1


def test_generate_prompt_includes_all_records(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    """All retrieved records must appear in the prompt so the LLM can reason across them."""
    result = graph._retrieve(_base_state(documents=documents))
    graph._generate(_base_state(relevant_chunks=result["relevant_chunks"]))
    assert "AUTHORISED" in fake_llm.received_message
    assert "DECLINED" in fake_llm.received_message
    assert "45000.00" in fake_llm.received_message
    assert "50000.00" in fake_llm.received_message
    assert "42000.00" in fake_llm.received_message


def test_generate_prompt_includes_job_title(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    graph._generate(_base_state(job_title="50k mile service", relevant_chunks=chunks))
    assert "50k mile service" in fake_llm.received_message


def test_generate_prompt_includes_job_cost(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    graph._generate(_base_state(job_cost=44000.0, relevant_chunks=chunks))
    assert "44000.00" in fake_llm.received_message


def test_generate_prompt_includes_make_and_model(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    graph._generate(_base_state(make="Vauxhall", model="Crossland X", relevant_chunks=chunks))
    assert "Vauxhall" in fake_llm.received_message
    assert "Crossland X" in fake_llm.received_message


def test_generate_verdict_not_authorised_when_cost_too_low(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    # historical avg £85,219 — submitted £1,000 is far below avg/2
    fake_llm.reply = '{"reasoning": "Cost is far below the historical average."}'
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 85219.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(job_cost=1000.0, relevant_chunks=chunks))
    assert json.loads(result["answer"])["verdict"] == "NOT AUTHORISED"


def test_generate_verdict_authorised_when_cost_in_range(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    fake_llm.reply = '{"reasoning": "Cost is within the historical range."}'
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(job_cost=44000.0, relevant_chunks=chunks))
    assert json.loads(result["answer"])["verdict"] == "AUTHORISED"


def test_generate_answer_uses_llm_reasoning(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    fake_llm.reply = '{"reasoning": "Cost matches the historical average."}'
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(job_cost=44000.0, relevant_chunks=chunks))
    assert json.loads(result["answer"])["reasoning"] == "Cost matches the historical average."


def test_generate_answer_is_valid_json(graph: RAGGraph) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    parsed = json.loads(result["answer"])
    assert "reasoning" in parsed
    assert "verdict" in parsed


def test_generate_verdict_comes_from_cost_evaluation_not_llm(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    # Even if LLM somehow included a verdict field, it must be ignored
    fake_llm.reply = '{"verdict": "AUTHORISED", "reasoning": "Looks fine."}'
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 85219.0, "job_status_updated_by_user": "john.smith",
    })]
    # job_cost=1000 is far below historical avg → Python says NOT AUTHORISED
    result = graph._generate(_base_state(job_cost=1000.0, relevant_chunks=chunks))
    assert json.loads(result["answer"])["verdict"] == "NOT AUTHORISED"


def test_generate_falls_back_reasoning_when_llm_returns_none(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    fake_llm.reply = '{"verdict": "AUTHORISED"}'
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    assert json.loads(result["answer"])["reasoning"] != ""


def test_generate_extracts_reasoning_from_prose_wrapped_response(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    fake_llm.reply = (
        'Here is my explanation:\n'
        '{"reasoning": "Cost is within the approved band."}\n'
        'Let me know if you need more.'
    )
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "john.smith",
    })]
    result = graph._generate(_base_state(job_cost=44000.0, relevant_chunks=chunks))
    assert json.loads(result["answer"])["reasoning"] == "Cost is within the approved band."


def test_generate_context_sorted_most_recent_first(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    older = Document(page_content="older repair", metadata={
        "jobLineStatus": "DECLINED", "job_status_date": "2023-01-01",
        "repair_cost": 50000.0, "job_status_updated_by_user": "old-user",
    })
    newer = Document(page_content="newer repair", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-31",
        "repair_cost": 45000.0, "job_status_updated_by_user": "new-user",
    })
    graph._generate(_base_state(relevant_chunks=[older, newer]))
    prompt = fake_llm.received_message
    assert prompt.index("newer repair") < prompt.index("older repair")


# ── _generate node — attribution fields from metadata ─────────────────────────

def test_generate_injects_last_updated_by_from_most_recent_chunk(
    graph: RAGGraph,
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "alice.jones",
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    assert json.loads(result["answer"])["last_updated_by"] == "alice.jones"


def test_generate_injects_last_updated_date_from_most_recent_chunk(
    graph: RAGGraph,
) -> None:
    chunks = [Document(page_content="50k mile service", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-01",
        "repair_cost": 45000.0, "job_status_updated_by_user": "alice.jones",
    })]
    result = graph._generate(_base_state(relevant_chunks=chunks))
    assert json.loads(result["answer"])["last_updated_date"] == "2024-12-01"


def test_generate_attribution_comes_from_most_recent_not_oldest(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    older = Document(page_content="older repair", metadata={
        "jobLineStatus": "DECLINED", "job_status_date": "2023-01-01",
        "repair_cost": 50000.0, "job_status_updated_by_user": "old-user",
    })
    newer = Document(page_content="newer repair", metadata={
        "jobLineStatus": "AUTHORISED", "job_status_date": "2024-12-31",
        "repair_cost": 45000.0, "job_status_updated_by_user": "new-user",
    })
    result = graph._generate(_base_state(relevant_chunks=[older, newer]))
    parsed = json.loads(result["answer"])
    assert parsed["last_updated_by"] == "new-user"
    assert parsed["last_updated_date"] == "2024-12-31"


# ── get_rag_graph singleton ────────────────────────────────────────────────────

def test_get_rag_graph_returns_same_instance() -> None:
    get_rag_graph.cache_clear()
    a = get_rag_graph()
    b = get_rag_graph()
    assert a is b
    get_rag_graph.cache_clear()


# ── _parse_llm_response ───────────────────────────────────────────────────────

def test_parse_llm_response_clean_json() -> None:
    raw = '{"verdict": "AUTHORISED", "reasoning": "Cost is within the approved band."}'
    result = _parse_llm_response(raw)
    assert result["verdict"] == "AUTHORISED"
    assert result["reasoning"] == "Cost is within the approved band."


def test_parse_llm_response_extracts_json_from_prose() -> None:
    raw = (
        'Sure, here is my answer:\n'
        '{"verdict": "AUTHORISED", "reasoning": "Similar jobs approved."}\n'
        'Hope that helps.'
    )
    result = _parse_llm_response(raw)
    assert result["verdict"] == "AUTHORISED"


def test_parse_llm_response_fallback_when_no_json() -> None:
    raw = "The repair looks fine but I cannot decide."
    result = _parse_llm_response(raw)
    assert result == {}


def test_parse_llm_response_fallback_returns_empty_dict() -> None:
    result = _parse_llm_response("completely unparseable !!!")
    assert isinstance(result, dict)


# ── _evaluate_repair ───────────────────────────────────────────────────────────

def test_evaluate_repair_authorised_when_cost_in_range_and_status_ok() -> None:
    assessment, verdict = _evaluate_repair(44000.0, [42000.0, 45000.0], ["AUTHORISED", "AUTHORISED"])
    assert verdict == "AUTHORISED"
    assert "in line" in assessment.lower()


def test_evaluate_repair_not_authorised_when_majority_declined() -> None:
    # 5 of 5 DECLINED — should be NOT AUTHORISED regardless of cost
    assessment, verdict = _evaluate_repair(10000.0, [6999.0] * 5, ["DECLINED"] * 5)
    assert verdict == "NOT AUTHORISED"
    assert "declined" in assessment.lower()


def test_evaluate_repair_not_authorised_when_majority_declined_mixed() -> None:
    # 3 DECLINED, 2 AUTHORISED → majority declined
    _, verdict = _evaluate_repair(10000.0, [7000.0] * 5, ["DECLINED", "DECLINED", "DECLINED", "AUTHORISED", "AUTHORISED"])
    assert verdict == "NOT AUTHORISED"


def test_evaluate_repair_authorised_when_minority_declined() -> None:
    # 1 DECLINED, 4 AUTHORISED → minority declined, cost in range
    _, verdict = _evaluate_repair(44000.0, [45000.0] * 5, ["DECLINED", "AUTHORISED", "AUTHORISED", "AUTHORISED", "AUTHORISED"])
    assert verdict == "AUTHORISED"


def test_evaluate_repair_status_check_takes_priority_over_cost() -> None:
    # Cost is in range but all records DECLINED → NOT AUTHORISED
    _, verdict = _evaluate_repair(6999.0, [6999.0], ["DECLINED"])
    assert verdict == "NOT AUTHORISED"


def test_evaluate_repair_not_authorised_when_cost_too_low() -> None:
    assessment, verdict = _evaluate_repair(1000.0, [85219.0], ["AUTHORISED"])
    assert verdict == "NOT AUTHORISED"
    assert "lower" in assessment.lower()


def test_evaluate_repair_not_authorised_when_cost_too_high() -> None:
    assessment, verdict = _evaluate_repair(200000.0, [85219.0], ["AUTHORISED"])
    assert verdict == "NOT AUTHORISED"
    assert "higher" in assessment.lower()


def test_evaluate_repair_not_authorised_when_no_historical_costs() -> None:
    assessment, verdict = _evaluate_repair(1000.0, [], ["AUTHORISED"])
    assert verdict == "NOT AUTHORISED"
    assert "no historical" in assessment.lower()


def test_evaluate_repair_uses_average_of_all_costs() -> None:
    _, verdict = _evaluate_repair(44000.0, [40000.0, 50000.0], ["AUTHORISED", "AUTHORISED"])
    assert verdict == "AUTHORISED"


def test_evaluate_repair_assessment_includes_submitted_and_average_costs() -> None:
    assessment, _ = _evaluate_repair(1000.0, [85219.0], ["AUTHORISED"])
    assert "1000.00" in assessment
    assert "85219.00" in assessment


# ── _to_iso_date_str ──────────────────────────────────────────────────────────

def test_to_iso_date_str_date_object() -> None:
    assert _to_iso_date_str(date(2024, 12, 1)) == "2024-12-01"


def test_to_iso_date_str_datetime_object_returns_date_only() -> None:
    """datetime objects must return YYYY-MM-DD, not an ISO timestamp with time."""
    assert _to_iso_date_str(datetime(2024, 12, 1, 10, 30, 0)) == "2024-12-01"


def test_to_iso_date_str_iso_string_passthrough() -> None:
    assert _to_iso_date_str("2024-03-15") == "2024-03-15"


def test_to_iso_date_str_dd_mm_yyyy() -> None:
    assert _to_iso_date_str("15/03/2024") == "2024-03-15"


def test_to_iso_date_str_none_returns_empty_string() -> None:
    assert _to_iso_date_str(None) == ""


def test_to_iso_date_str_empty_string_returns_empty_string() -> None:
    assert _to_iso_date_str("") == ""


def test_to_iso_date_str_unknown_format_returns_as_is() -> None:
    # Unknown strings are returned verbatim so the caller can decide how to handle them.
    assert _to_iso_date_str("not-a-date") == "not-a-date"
