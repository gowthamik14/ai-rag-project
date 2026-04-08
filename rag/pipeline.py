"""
RAGPipeline: orchestrates the full Retrieve → Augment → Generate cycle.

All queries pass through TopicGuardrail first — non-vehicle questions
are rejected before any retrieval or LLM call is made.

Usage
-----
pipeline = RAGPipeline()

# One-shot
result = pipeline.query("What is the tire pressure for a Honda Civic?")

# Blocked (non-vehicle)
result = pipeline.query("What is the weather today?")
# → result.blocked == True, result.answer == guardrail message

# Conversational (tracks session in Firestore)
session_id = pipeline.create_session()
result = pipeline.chat("My brake light is on — what should I do?", session_id=session_id)

# Streaming
for token in pipeline.stream_query("How often should I change engine oil?"):
    print(token, end="", flush=True)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .retriever import Retriever
from .generator import Generator
from core.guardrails import TopicGuardrail
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
    blocked: bool = False                # True when guardrail rejected the question
    block_reason: Optional[str] = None  # e.g. "keyword_block", "llm_block"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """End-to-end RAG pipeline restricted to the vehicle domain."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        document_store: Optional[DocumentStore] = None,
        guardrail: Optional[TopicGuardrail] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._retriever       = retriever or Retriever()
        self._generator       = generator or Generator()
        self._doc_store       = document_store or DocumentStore()
        self._guardrail       = guardrail or TopicGuardrail()
        self._top_k           = top_k
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
        """Single-turn RAG query — blocked if not vehicle-related."""
        # Guardrail check
        allowed, reason = self._guardrail.check(question)
        if not allowed:
            logger.info("Query blocked by guardrail (%s): %s", reason, question[:80])
            return RAGResult(
                question=question,
                answer=TopicGuardrail.BLOCKED_RESPONSE,
                blocked=True,
                block_reason=reason,
            )

        chunks  = self._retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            score_threshold=score_threshold if score_threshold is not None else self._score_threshold,
        )
        context = self._retriever.format_context(chunks)
        answer  = self._generator.generate(
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
        """Stream tokens — yields blocked message if not vehicle-related."""
        allowed, reason = self._guardrail.check(question)
        if not allowed:
            logger.info("Stream query blocked by guardrail (%s): %s", reason, question[:80])
            yield TopicGuardrail.BLOCKED_RESPONSE
            return

        chunks  = self._retriever.retrieve(
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
        Multi-turn conversational RAG — blocked if not vehicle-related.
        Blocked exchanges are NOT persisted to the session history.
        """
        # Guardrail check
        allowed, reason = self._guardrail.check(question)
        if not allowed:
            logger.info("Chat blocked by guardrail (%s): %s", reason, question[:80])
            return RAGResult(
                question=question,
                answer=TopicGuardrail.BLOCKED_RESPONSE,
                session_id=session_id,
                blocked=True,
                block_reason=reason,
            )

        history = self._doc_store.get_session_history(session_id)
        chunks  = self._retriever.retrieve(
            query=question,
            top_k=top_k or self._top_k,
            score_threshold=score_threshold if score_threshold is not None else self._score_threshold,
        )
        context = self._retriever.format_context(chunks)
        answer  = self._generator.chat_generate(
            question=question,
            context=context,
            history=history,
            system_prompt=system_prompt,
        )

        # Only persist valid vehicle exchanges
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
        from core.llm import OllamaLLM

        vs  = VectorStore()
        llm = OllamaLLM()
        return {
            "vector_store_total": vs.total_vectors,
            "llm_available": llm.health_check(),
        }
