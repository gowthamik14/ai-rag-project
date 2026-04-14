from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from config.settings import settings
from rag.document_loader import DocumentLoader, create_model_loader


# ── Constants ─────────────────────────────────────────────────────────────────

# If the submitted job cost exceeds the best matching historical repair cost by
# more than this factor the job is NOT AUTHORISED regardless of jobLineStatus.
# Example: historical £9 999, submitted £50 000 → ratio ≈ 5 × → rejected.
_COST_TOLERANCE_MULTIPLIER: float = 2.0

# Maximum L2 distance (vs the best-match score) a chunk may have and still be
# considered "similar enough" to participate in the date-sort decision.
# FAISS with normalised embeddings gives L2 distances in [0, 2].  A buffer of
# 0.3 admits paraphrases of the same repair type while excluding unrelated
# records (e.g. a test entry "1123") that happen to have a recent date.
_SIMILARITY_SCORE_BUFFER: float = 0.3


# ── Graph state ────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """Data that flows between nodes in the RAG graph."""
    question:        str             # full request shown to the LLM
    job_title:       str             # bare repair description — used as the FAISS query
    job_cost:        float           # submitted repair cost in GBP
    make:            str             # vehicle make  — used by _load to filter BigQuery
    model:           str             # vehicle model — used by _load to filter BigQuery
    documents:       list[Document]  # raw rows from BigQuery
    relevant_chunks: list[Document]  # top-k chunks after vector search
    answer:          str


# ── Abstraction ────────────────────────────────────────────────────────────────

class BaseRAGGraph(ABC):
    """Abstract contract for the RAG pipeline."""

    @abstractmethod
    def run(self, question: str, job_title: str, model: str, make: str, job_cost: float) -> str:
        """Run the pipeline for a given question, filtering data by vehicle make/model."""


# ── Embeddings singleton ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse across requests."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Concrete implementation ────────────────────────────────────────────────────

class RAGGraph(BaseRAGGraph):
    """LangGraph-based RAG pipeline.

    Graph structure:
        load → retrieve → generate → END

    - load     : fetches Documents from BigQuery filtered by vehicle model, make, and
                 an optional cost band derived from job_cost (skipped when job_cost=0).
    - retrieve : splits into chunks, builds an in-memory FAISS index, returns
                 the top-k most relevant chunks for the question.
    - generate : picks the **most semantically similar** retrieved record (chunk[0]
                 from FAISS), derives the verdict from its jobLineStatus, and formats
                 the reasoning message using job_status_updated_by_user and
                 job_status_date. No LLM call — the decision is fully deterministic.

    Args:
        loader_factory:       Callable that takes (model, make, job_cost) and returns a
                              DocumentLoader. Defaults to create_model_loader.
                              Inject a fake in tests to avoid BigQuery calls.
        embeddings:           LangChain Embeddings implementation. Defaults to
                              HuggingFaceEmbeddings. Inject a fake in tests to
                              avoid downloading the sentence-transformers model.
        vector_store_factory: Callable that takes (chunks, embeddings) and
                              returns a vector store with a similarity_search
                              method. Defaults to FAISS.from_documents.
                              Inject a fake in tests to avoid native FAISS calls.
    """

    def __init__(
        self,
        loader_factory: Callable[[str, str, float], DocumentLoader] = create_model_loader,
        embeddings: Embeddings | None = None,
        vector_store_factory: Callable[[list[Document], Embeddings], Any] | None = None,
    ) -> None:
        self._loader_factory        = loader_factory
        self._embeddings            = embeddings or _get_embeddings()
        self._vector_store_factory  = vector_store_factory or FAISS.from_documents
        self._splitter              = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._graph = self._build_graph()

    # ── public ────────────────────────────────────────────────────────────

    def run(self, question: str, job_title: str, model: str, make: str, job_cost: float) -> str:
        """Run the full RAG pipeline and return the generated answer."""
        result = self._graph.invoke({
            "question":        question,
            "job_title":       job_title,
            "job_cost":        job_cost,
            "make":            make,
            "model":           model,
            "documents":       [],
            "relevant_chunks": [],
            "answer":          "",
        })
        return result["answer"]

    # ── graph construction ─────────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("load",     self._load)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.set_entry_point("load")
        builder.add_edge("load",     "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    # ── nodes ──────────────────────────────────────────────────────────────

    def _load(self, state: RAGState) -> dict:
        """Node: fetch Documents from BigQuery filtered by vehicle make, model, and cost band."""
        loader = self._loader_factory(state["model"], state["make"], state["job_cost"])
        return {"documents": loader.load()}

    def _retrieve(self, state: RAGState) -> dict:
        """Node: chunk documents, build in-memory FAISS, return top-k relevant chunks.

        Uses similarity scores to discard records that are semantically unrelated
        to job_title (e.g. a test entry "1123" in a sparse dataset).  Only chunks
        whose L2 distance is within _SIMILARITY_SCORE_BUFFER of the best match are
        kept, so the date-sort in _generate only compares like-for-like repairs.
        """
        if not state["documents"]:
            return {"relevant_chunks": []}

        chunks = self._splitter.split_documents(state["documents"])
        if not chunks:
            return {"relevant_chunks": []}

        vector_store = self._vector_store_factory(chunks, self._embeddings)
        # Use job_title (bare repair phrase) as the retrieval query — it matches
        # repair_description embeddings far better than the full verbose question.
        results: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
            state["job_title"], k=settings.retrieval_top_k
        )
        if not results:
            return {"relevant_chunks": []}

        best_score = results[0][1]
        relevant = [
            doc for doc, score in results
            if score <= best_score + _SIMILARITY_SCORE_BUFFER
        ]
        return {"relevant_chunks": relevant}

    def _generate(self, state: RAGState) -> dict:
        """Node: deterministically derive verdict from the most recent similar repair record.

        relevant_chunks has already been filtered by _retrieve to contain only
        records that are semantically close to job_title.  Within that filtered
        set the most recent record (by job_status_date DESC) wins — if today's
        record is AUTHORISED and yesterday's is DECLINED, the verdict is AUTHORISED.
        The reasoning message is built from job_status_updated_by_user and
        job_status_date of the winning record.
        """
        if not state["relevant_chunks"]:
            return {"answer": json.dumps({
                "reasoning": "No similar past repair records found for this vehicle.",
                "verdict": "NOT AUTHORISED",
            })}

        # Sort latest-first among the pre-filtered similar records.
        # str() handles str, date, and datetime values uniformly.
        sorted_chunks = sorted(
            state["relevant_chunks"],
            key=lambda d: str(d.metadata.get("job_status_date") or ""),
            reverse=True,
        )
        latest = sorted_chunks[0]

        status      = (latest.metadata.get("jobLineStatus") or "").strip()
        date        = str(latest.metadata.get("job_status_date") or "")
        user        = str(latest.metadata.get("job_status_updated_by_user") or "")
        repair_cost = latest.metadata.get("repair_cost")
        multiple    = len(state["relevant_chunks"]) > 1
        attr        = _attribution(user, date)
        latest_sfx  = _latest_suffix(user, date)

        if not _is_authorised_status(status):
            return _verdict("NOT AUTHORISED",
                f"Similar jobs were declined.{latest_sfx}" if multiple
                else f"Similar job was declined{attr}.")

        if state["job_cost"] > 0 and repair_cost is None:
            return _verdict("NOT AUTHORISED",
                f"Cannot verify cost: no historical repair cost on record for similar jobs.{latest_sfx}" if multiple
                else f"Cannot verify cost: no historical repair cost on record for the similar job authorised{attr}.")

        if (state["job_cost"] > 0
                and repair_cost is not None
                and float(state["job_cost"]) > float(repair_cost) * _COST_TOLERANCE_MULTIPLIER):
            return _verdict("NOT AUTHORISED",
                f"Submitted cost (£{state['job_cost']:.2f}) significantly exceeds "
                f"the historical repair cost (£{float(repair_cost):.2f}) for a similar "
                f"job authorised{attr}.")

        return _verdict("AUTHORISED",
            f"Similar jobs have been authorised.{latest_sfx}" if multiple
            else f"Similar job has been authorised{attr}.")


# ── helpers ────────────────────────────────────────────────────────────────────

def _verdict(verdict: str, reasoning: str) -> dict:
    return {"answer": json.dumps({"reasoning": reasoning, "verdict": verdict})}


def _attribution(user: str, date: str) -> str:
    """Return ' by {user} on {date}', ' by {user}', ' on {date}', or ''."""
    if user and date:
        return f" by {user} on {date}"
    return f" by {user}" if user else f" on {date}" if date else ""


def _latest_suffix(user: str, date: str) -> str:
    """Return ' Latest is by {user} on {date}.' for plural messages, or ''."""
    inner = _attribution(user, date).strip()
    return f" Latest is {inner}." if inner else ""


def _is_authorised_status(status: str) -> bool:
    """Return True when the jobLineStatus string represents an authorised outcome.

    Matches "AUTHORISED" / "Authorised" but not "NOT AUTHORISED" / "Declined".
    """
    upper = status.upper().strip()
    return "AUTHORISED" in upper and "NOT" not in upper


# ── dependency ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_graph() -> BaseRAGGraph:
    """FastAPI dependency — returns a single compiled RAGGraph instance."""
    return RAGGraph()
