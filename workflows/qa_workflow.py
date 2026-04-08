"""
QAWorkflow
==========
Answers a question using the RAG pipeline.
Supports one-shot and session-based (conversational) modes.

Inputs (dict)
-------------
  question    : str   (required)
  session_id  : str   (optional – enables conversational mode)
  user_id     : str   (optional – used when creating a new session)
  top_k       : int   (optional – override retrieval count)
  stream      : bool  (optional – not used here; use pipeline.stream_query directly)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base_workflow import BaseWorkflow
from rag.pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class QAWorkflow(BaseWorkflow):
    name = "qa"

    def __init__(self, pipeline: Optional[RAGPipeline] = None) -> None:
        self._pipeline = pipeline or RAGPipeline()

    def _validate(self, inputs: Dict[str, Any]) -> None:
        if not inputs.get("question", "").strip():
            raise ValueError("'question' must be a non-empty string.")

    @property
    def _steps(self):
        return [
            ("resolve_session", self._step_resolve_session),
            ("run_rag", self._step_run_rag),
            ("format_output", self._step_format_output),
        ]

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _step_resolve_session(self, ctx: Dict[str, Any]) -> None:
        inputs = ctx["inputs"]
        session_id = inputs.get("session_id")

        if not session_id and inputs.get("conversational"):
            # Auto-create a new session when conversational mode is requested
            session_id = self._pipeline.create_session(user_id=inputs.get("user_id"))
            logger.info("Auto-created session_id=%s", session_id)

        ctx["session_id"] = session_id

    def _step_run_rag(self, ctx: Dict[str, Any]) -> None:
        inputs = ctx["inputs"]
        question = inputs["question"]
        session_id = ctx.get("session_id")
        top_k = inputs.get("top_k")

        if session_id:
            result = self._pipeline.chat(
                question=question,
                session_id=session_id,
                top_k=top_k,
            )
        else:
            result = self._pipeline.query(
                question=question,
                top_k=top_k,
            )

        ctx["rag_result"] = result

    def _step_format_output(self, ctx: Dict[str, Any]) -> None:
        result = ctx["rag_result"]
        ctx["output"] = {
            "question": result.question,
            "answer": result.answer,
            "session_id": result.session_id,
            "sources": [
                {
                    "score": c.get("score"),
                    "source": c.get("source", c.get("doc_id", "")),
                    "chunk_index": c.get("chunk_index"),
                    "text_preview": c.get("text", "")[:200],
                }
                for c in result.retrieved_chunks
            ],
        }
