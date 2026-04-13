from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from rag.document_loader import DocumentLoader
from rag.graph import RAGGraph, RAGState, get_rag_graph
from services.llm_service import LLMService


# ── test doubles ───────────────────────────────────────────────────────────────

class FakeLLMService(LLMService):
    def __init__(self, reply: str = "mocked answer") -> None:
        self.reply = reply
        self.received_prompt: str | None = None

    def chat(self, message: str) -> str:
        self.received_prompt = message
        return self.reply


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
    """Stub vector store — returns documents as-is without calling FAISS."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.last_query: str | None = None

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        self.last_query = query
        return self._documents[:k]


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def documents() -> list[Document]:
    return [
        Document(
            page_content="Brake pad replacement",
            metadata={"jobLineStatus": "AUTHORISED", "job_status_date": "2024-03-15"},
        ),
        Document(
            page_content="Oil and filter change",
            metadata={"jobLineStatus": "NOT AUTHORISED", "job_status_date": "2024-01-10"},
        ),
    ]


@pytest.fixture()
def fake_loader(documents: list[Document]) -> FakeDocumentLoader:
    return FakeDocumentLoader(documents=documents)


@pytest.fixture()
def fake_llm() -> FakeLLMService:
    return FakeLLMService(reply="The cost is £80.")


@pytest.fixture()
def graph(fake_loader: FakeDocumentLoader, fake_llm: FakeLLMService) -> RAGGraph:
    return RAGGraph(
        llm_service=fake_llm,
        loader_factory=lambda model, make: fake_loader,
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )


def _base_state(**overrides) -> RAGState:
    """Return a minimal valid RAGState, with optional field overrides."""
    state: RAGState = {
        "question":        "A 'brake pad replacement' repair has been submitted.",
        "job_title":       "brake pad replacement",
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
    result = graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford")
    assert isinstance(result, str)


def test_run_returns_llm_answer(graph: RAGGraph, fake_llm: FakeLLMService) -> None:
    fake_llm.reply = "Brake pads cost £80."
    result = graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford")
    assert result == "Brake pads cost £80."


def test_run_calls_loader_once(graph: RAGGraph, fake_loader: FakeDocumentLoader) -> None:
    graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford")
    assert fake_loader.load_call_count == 1


def test_run_calls_llm(graph: RAGGraph, fake_llm: FakeLLMService) -> None:
    graph.run("question", job_title="brake pads", model="Ford Focus", make="Ford")
    assert fake_llm.received_prompt is not None


# ── _load node ─────────────────────────────────────────────────────────────────

def test_load_uses_model_and_make_from_state(fake_llm: FakeLLMService) -> None:
    received: list[tuple[str, str]] = []

    def capturing_factory(model: str, make: str) -> DocumentLoader:
        received.append((model, make))
        return FakeDocumentLoader()

    graph = RAGGraph(
        llm_service=fake_llm,
        loader_factory=capturing_factory,
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )
    graph._load(_base_state(model="BMW X5", make="BMW"))

    assert received == [("BMW X5", "BMW")]


def test_load_returns_documents_from_loader(
    graph: RAGGraph, documents: list[Document]
) -> None:
    result = graph._load(_base_state())
    assert result["documents"] == documents


def test_load_returns_empty_list_when_no_rows(fake_llm: FakeLLMService) -> None:
    graph = RAGGraph(
        llm_service=fake_llm,
        loader_factory=lambda m, mk: FakeDocumentLoader(documents=[]),
        embeddings=FakeEmbeddings(),
        vector_store_factory=lambda docs, emb: FakeVectorStore(docs),
    )
    result = graph._load(_base_state(model="Unknown"))
    assert result["documents"] == []


# ── _retrieve node ─────────────────────────────────────────────────────────────

def test_retrieve_uses_job_title_not_full_question(
    fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    """FAISS search must use the bare job_title, not the verbose question string."""
    captured_store: list[FakeVectorStore] = []

    def capturing_factory(docs, emb):
        store = FakeVectorStore(docs)
        captured_store.append(store)
        return store

    graph = RAGGraph(
        llm_service=fake_llm,
        loader_factory=lambda m, mk: FakeDocumentLoader(documents=documents),
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


# ── _generate node ─────────────────────────────────────────────────────────────

def test_generate_includes_question_in_prompt(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(
        question="A brake pad replacement has been submitted.",
        relevant_chunks=documents,
    ))
    assert "A brake pad replacement has been submitted." in fake_llm.received_prompt


def test_generate_includes_repair_description_in_prompt(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(relevant_chunks=documents))
    assert "Brake pad replacement" in fake_llm.received_prompt
    assert "Oil and filter change" in fake_llm.received_prompt


def test_generate_includes_job_line_status_in_prompt(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(relevant_chunks=documents))
    assert "AUTHORISED" in fake_llm.received_prompt


def test_generate_includes_job_status_date_in_prompt(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(relevant_chunks=documents))
    assert "2024-03-15" in fake_llm.received_prompt


def test_generate_prompt_instructs_use_only_retrieved_records(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(relevant_chunks=documents))
    assert "SOLELY" in fake_llm.received_prompt


def test_generate_prompt_prohibits_inventing_data(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(relevant_chunks=documents))
    assert "do not invent" in fake_llm.received_prompt.lower()


def test_generate_returns_llm_reply(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    fake_llm.reply = "The answer is 42."
    result = graph._generate(_base_state(relevant_chunks=documents))
    assert result["answer"] == "The answer is 42."


def test_generate_uses_fallback_context_when_no_chunks(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    graph._generate(_base_state(relevant_chunks=[]))
    assert "No similar past repair records" in fake_llm.received_prompt


def test_generate_prompt_instructs_not_authorised_verdict(
    graph: RAGGraph, fake_llm: FakeLLMService
) -> None:
    graph._generate(_base_state(relevant_chunks=[]))
    assert "NOT AUTHORISED" in fake_llm.received_prompt


def test_generate_prompt_requests_json_output(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    """Prompt must instruct the model to respond with a JSON object."""
    graph._generate(_base_state(relevant_chunks=documents))
    assert "JSON" in fake_llm.received_prompt


def test_generate_includes_make_and_model_in_prompt(
    graph: RAGGraph, fake_llm: FakeLLMService, documents: list[Document]
) -> None:
    graph._generate(_base_state(make="Toyota", model="Corolla", relevant_chunks=documents))
    assert "Toyota" in fake_llm.received_prompt
    assert "Corolla" in fake_llm.received_prompt


# ── get_rag_graph singleton ────────────────────────────────────────────────────

def test_get_rag_graph_returns_same_instance() -> None:
    with patch("rag.graph.get_llm_service"):
        get_rag_graph.cache_clear()
        a = get_rag_graph()
        b = get_rag_graph()
        assert a is b
        get_rag_graph.cache_clear()
