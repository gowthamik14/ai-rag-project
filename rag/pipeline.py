"""
RAGPipeline: orchestrates the full Retrieve → Augment → Generate cycle.

Usage
-----
pipeline = RAGPipeline()

# One-shot
result = pipeline.query("What is the refund policy?")

# Conversational (tracks session in Firestore)
session_id = pipeline.create_session()
result = pipeline.chat("What is the refund policy?", session_id=session_id)
result = pipeline.chat("Can I get a full refund?", session_id=session_id)

# Streaming
for token in pipeline.stream_query("Summarise the document"):
    print(token, end="", flush=True)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .retriever import Retriever
from .generator import Generator
from db.document_store import DocumentStore
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context_used: str = ""
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """End-to-end RAG pipeline with optional session management."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        document_store: Optional[DocumentStore] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._generator = generator or Generator()
        self._doc_store = document_store or DocumentStore()
        self._top_k = top_k
        self._score_threshold = score_threshold

    # ------------------------------------------------------------------ #
    # One-shot query
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> RAGResult:
        """Single-turn RAG query."""
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            score_threshold=score_threshold if score_threshold is not None else self._score_threshold,
        )
        context = self._retriever.format_context(chunks)
        answer = self._generator.generate(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )
        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            context_used=context,
        )

    # ------------------------------------------------------------------ #
    # Streaming query
    # ------------------------------------------------------------------ #

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream tokens for a single-turn query."""
        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            score_threshold=score_threshold if score_threshold is not None else self._score_threshold,
        )
        context = self._retriever.format_context(chunks)
        yield from self._generator.stream_generate(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------ #
    # Conversational query (session-aware)
    # ------------------------------------------------------------------ #

    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new Firestore-persisted chat session. Returns session_id."""
        return self._doc_store.create_session(user_id=user_id)

    def chat(
        self,
        question: str,
        session_id: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> RAGResult:
        """
        Multi-turn conversational RAG.
        Loads history from Firestore, generates an answer, persists the exchange.
        """
        history = self._doc_store.get_session_history(session_id)

        chunks = self._retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            score_threshold=score_threshold if score_threshold is not None else self._score_threshold,
        )
        context = self._retriever.format_context(chunks)

        answer = self._generator.chat_generate(
            question=question,
            context=context,
            history=history,
            system_prompt=system_prompt,
        )

        # Persist exchange to Firestore
        self._doc_store.append_message(session_id, "user", question)
        self._doc_store.append_message(session_id, "assistant", answer)

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            context_used=context,
            session_id=session_id,
        )

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        from core.vector_store import VectorStore
        from core.llm import GemmaLLM

        vs = VectorStore()
        llm = GemmaLLM()
        return {
            "vector_store_total": vs.total_vectors,
            "llm_available": llm.health_check(),
        }
