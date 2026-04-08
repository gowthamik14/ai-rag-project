"""
WorkflowRouter
==============
Selects and runs the correct workflow based on an 'intent' field,
or uses GEMMA to auto-classify the intent from the input.

Registered workflows:
  ingestion      → IngestionWorkflow
  qa             → QAWorkflow
  summarization  → SummarizationWorkflow

Usage
-----
router = WorkflowRouter()

# Explicit intent
result = router.route({"intent": "qa", "question": "What is GDPR?"})

# Auto-detect intent
result = router.route({"question": "Summarise document abc123", "doc_id": "abc123"})
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base_workflow import WorkflowResult, WorkflowStatus
from .ingestion_workflow import IngestionWorkflow
from .qa_workflow import QAWorkflow
from .summarization_workflow import SummarizationWorkflow
from core.llm import GemmaLLM
from utils.logger import get_logger

logger = get_logger(__name__)

INTENT_DETECTION_PROMPT = """\
Given the following user request, classify it as exactly one of:
  ingestion, qa, summarization

Rules:
  ingestion     – user wants to add / upload / ingest a document
  qa            – user wants to ask a question and get an answer
  summarization – user wants a summary of a document

Request: "{request}"

Respond with ONLY one word (ingestion, qa, or summarization):"""


class WorkflowRouter:
    def __init__(self, llm: Optional[GemmaLLM] = None) -> None:
        self._llm = llm or GemmaLLM()
        self._registry = {
            "ingestion": IngestionWorkflow,
            "qa": QAWorkflow,
            "summarization": SummarizationWorkflow,
        }

    def route(self, inputs: Dict[str, Any]) -> WorkflowResult:
        """
        Dispatch to the correct workflow.

        If `inputs` contains 'intent', use it directly.
        Otherwise, ask GEMMA to classify the intent.
        """
        intent = inputs.pop("intent", None)

        if not intent:
            intent = self._detect_intent(inputs)

        intent = intent.strip().lower()
        logger.info("Routing to workflow: %s", intent)

        workflow_cls = self._registry.get(intent)
        if workflow_cls is None:
            return WorkflowResult(
                workflow="router",
                status=WorkflowStatus.FAILED,
                errors=[f"Unknown intent: '{intent}'. Valid: {list(self._registry)}"],
            )

        return workflow_cls().run(inputs)

    def register(self, intent: str, workflow_cls) -> None:
        """Register a custom workflow under a new intent name."""
        self._registry[intent] = workflow_cls
        logger.info("Registered workflow '%s' → %s", intent, workflow_cls.__name__)

    # ------------------------------------------------------------------ #
    # Private
    # ------------------------------------------------------------------ #

    def _detect_intent(self, inputs: Dict[str, Any]) -> str:
        # Build a human-readable summary of the request
        parts = []
        if q := inputs.get("question"):
            parts.append(q)
        if inputs.get("text") or inputs.get("file_path"):
            parts.append("ingest document")
        if inputs.get("doc_id") and not inputs.get("question"):
            parts.append("summarise document")

        request_text = " ".join(parts) or str(inputs)
        prompt = INTENT_DETECTION_PROMPT.format(request=request_text)

        try:
            intent = self._llm.generate(prompt).strip().lower().split()[0]
            logger.info("Auto-detected intent: '%s' from request: '%s'", intent, request_text[:80])
            return intent
        except Exception as exc:
            logger.error("Intent detection failed: %s — defaulting to 'qa'", exc)
            return "qa"
