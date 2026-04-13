from __future__ import annotations

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
from services.llm_service import LLMService, get_llm_service


# ── Graph state ────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """Data that flows between nodes in the RAG graph."""
    question:        str             # full request shown to the LLM
    job_title:       str             # bare repair description — used as the FAISS query
    make:            str             # vehicle make  — used by _load to filter BigQuery
    model:           str             # vehicle model — used by _load to filter BigQuery
    documents:       list[Document]  # raw rows from BigQuery
    relevant_chunks: list[Document]  # top-k chunks after vector search
    answer:          str


# ── Abstraction ────────────────────────────────────────────────────────────────

class BaseRAGGraph(ABC):
    """Abstract contract for the RAG pipeline.

    Routes depend on this, not on RAGGraph directly, so the pipeline can be
    swapped or faked in tests without patching internals.
    """

    @abstractmethod
    def run(self, question: str, job_title: str, model: str, make: str) -> str:
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

    - load     : fetches Documents from BigQuery filtered by vehicle model and make.
    - retrieve : splits into chunks, builds an in-memory FAISS index, returns
                 the top-k most relevant chunks for the question.
    - generate : formats the relevant chunks as context and asks the LLM.

    Args:
        llm_service:          LLM abstraction for generating answers.
        loader_factory:       Callable that takes a model name and returns a
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
        llm_service: LLMService,
        loader_factory: Callable[[str, str], DocumentLoader] = create_model_loader,
        embeddings: Embeddings | None = None,
        vector_store_factory: Callable[[list[Document], Embeddings], Any] | None = None,
    ) -> None:
        self._llm                   = llm_service
        self._loader_factory        = loader_factory
        self._embeddings            = embeddings or _get_embeddings()
        self._vector_store_factory  = vector_store_factory or FAISS.from_documents
        self._splitter              = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self._graph = self._build_graph()

    # ── public ────────────────────────────────────────────────────────────

    def run(self, question: str, job_title: str, model: str, make: str) -> str:
        """Run the full RAG pipeline and return the generated answer."""
        result = self._graph.invoke({
            "question":        question,
            "job_title":       job_title,
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
        """Node: fetch Documents from BigQuery filtered by vehicle make and model."""
        loader = self._loader_factory(state["model"], state["make"])
        return {"documents": loader.load()}

    def _retrieve(self, state: RAGState) -> dict:
        """Node: chunk documents, build in-memory FAISS, return top-k relevant chunks."""
        if not state["documents"]:
            return {"relevant_chunks": []}

        chunks = self._splitter.split_documents(state["documents"])
        if not chunks:
            return {"relevant_chunks": []}

        vector_store = self._vector_store_factory(chunks, self._embeddings)
        # Use job_title (bare repair phrase) as the retrieval query — it matches
        # repair_description embeddings far better than the full verbose question.
        relevant = vector_store.similarity_search(
            state["job_title"], k=settings.retrieval_top_k
        )
        return {"relevant_chunks": relevant}

    def _generate(self, state: RAGState) -> dict:
        """Node: build a prompt from past repair records (with job_status) and call the LLM.

        Each relevant chunk's page_content is the repair_description of a past job.
        Its metadata contains job_status so the LLM can see whether that job was
        historically authorised or rejected before making a decision.
        """
        if state["relevant_chunks"]:
            lines = []
            for i, doc in enumerate(state["relevant_chunks"], 1):
                status = doc.metadata.get("jobLineStatus", "UNKNOWN")
                date = doc.metadata.get("job_status_date", "")
                date_str = f" on {date}" if date else ""
                lines.append(
                    f"{i}. Repair: {doc.page_content.strip()} | "
                    f"Outcome: {status}{date_str}"
                )
            past_records = "\n".join(lines)
        else:
            past_records = "No similar past repair records found for this vehicle."

        prompt = (
            "You are an automotive repair authorisation assistant.\n"
            "IMPORTANT: Base your decision SOLELY on the past repair records below. "
            "Do not use any general automotive knowledge or assume anything not "
            "present in the records.\n\n"
            f"Vehicle: {state['make']} {state['model']}\n\n"
            "Past repair records retrieved for similar jobs:\n"
            f"{past_records}\n\n"
            f"Request: {state['question']}\n\n"
            "Examine the records above. If an AUTHORISED record exists, name the repair "
            "description and its date. If no AUTHORISED record exists or no records were "
            "retrieved, state that — do not invent or assume any data.\n\n"
            "You must respond ONLY with a single JSON object that contains two keys: "
            "'reasoning' (string) and 'verdict' (string). "
            "The 'verdict' value must be exactly 'AUTHORISED' or 'NOT AUTHORISED'. "
            "Do not include any text outside the JSON object."
        )
        return {"answer": self._llm.chat(prompt)}


# ── dependency ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_graph() -> BaseRAGGraph:
    """FastAPI dependency — returns a single compiled RAGGraph instance."""
    return RAGGraph(llm_service=get_llm_service())
