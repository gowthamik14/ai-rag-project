"""
SummarizationWorkflow
=====================
Summarises a document already stored in Firestore (by doc_id) or raw text.

Steps:
  1. load_content   – fetch doc from Firestore or use inline text
  2. chunk_if_long  – split very long docs into windows
  3. summarize      – call Qwen with a summarization prompt
  4. merge_summary  – if multiple windows, combine partial summaries

Inputs (dict)
-------------
  doc_id      : str   (load from Firestore)   ← one of these required
  text        : str   (raw text to summarise)
  style       : str   "concise" | "detailed" | "bullet_points"  (default: "concise")
  max_length  : int   approximate word count for the summary (default: 200)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base_workflow import BaseWorkflow
from core.llm import OllamaLLM
from db.document_store import DocumentStore
from utils.text_splitter import TextSplitter
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

SUMMARIZATION_SYSTEM_PROMPT = "You are an expert summarizer. Produce clear, accurate summaries."

STYLE_INSTRUCTIONS = {
    "concise": "Write a concise summary in 2-4 sentences.",
    "detailed": "Write a detailed summary covering all key points.",
    "bullet_points": "Write the summary as a bullet-point list of key takeaways.",
}

SUMMARIZE_PROMPT = """\
{style_instruction}
Maximum length: approximately {max_length} words.

Text to summarise:
\"\"\"
{text}
\"\"\"

Summary:"""

MERGE_PROMPT = """\
You have been given partial summaries of different sections of a document.
Merge them into one coherent {style} summary (max ~{max_length} words).

Partial summaries:
{partials}

Final summary:"""


class SummarizationWorkflow(BaseWorkflow):
    name = "summarization"

    # Characters per summarization window (avoid overflowing context)
    WINDOW_SIZE = 3000

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        document_store: Optional[DocumentStore] = None,
    ) -> None:
        self._llm       = llm or OllamaLLM()
        self._doc_store = document_store or DocumentStore()
        self._splitter  = TextSplitter(chunk_size=self.WINDOW_SIZE, chunk_overlap=100)

    def _validate(self, inputs: Dict[str, Any]) -> None:
        if not inputs.get("doc_id") and not inputs.get("text", "").strip():
            raise ValueError("Either 'doc_id' or 'text' must be provided.")

    @property
    def _steps(self):
        return [
            ("load_content",  self._step_load_content),
            ("chunk_if_long", self._step_chunk_if_long),
            ("summarize",     self._step_summarize),
            ("merge_summary", self._step_merge_summary),
        ]

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _step_load_content(self, ctx: Dict[str, Any]) -> None:
        inputs = ctx["inputs"]
        text   = inputs.get("text", "")

        if not text and (doc_id := inputs.get("doc_id")):
            doc = self._doc_store.get_document(doc_id)
            if not doc:
                raise ValueError(f"Document '{doc_id}' not found in Firestore.")
            text = doc.get("content", "")

        ctx["content"]    = text.strip()
        ctx["style"]      = inputs.get("style", "concise")
        ctx["max_length"] = int(inputs.get("max_length", 200))
        logger.info("Loaded %d chars for summarization", len(ctx["content"]))

    def _step_chunk_if_long(self, ctx: Dict[str, Any]) -> None:
        content = ctx["content"]
        ctx["windows"] = (
            [content] if len(content) <= self.WINDOW_SIZE
            else self._splitter.split(content)
        )
        logger.info("Summarization windows: %d", len(ctx["windows"]))

    def _step_summarize(self, ctx: Dict[str, Any]) -> None:
        style_instr = STYLE_INSTRUCTIONS.get(ctx["style"], STYLE_INSTRUCTIONS["concise"])
        partials: List[str] = []

        for i, window in enumerate(ctx["windows"]):
            prompt = SUMMARIZE_PROMPT.format(
                style_instruction=style_instr,
                max_length=ctx["max_length"],
                text=window,
            )
            partials.append(
                self._llm.generate(prompt=prompt, system_prompt=SUMMARIZATION_SYSTEM_PROMPT)
            )
            logger.debug("Summarized window %d/%d", i + 1, len(ctx["windows"]))

        ctx["partial_summaries"] = partials

    def _step_merge_summary(self, ctx: Dict[str, Any]) -> None:
        partials = ctx["partial_summaries"]

        if len(partials) == 1:
            final = partials[0]
        else:
            partials_text = "\n\n---\n\n".join(
                f"Section {i+1}:\n{p}" for i, p in enumerate(partials)
            )
            final = self._llm.generate(
                prompt=MERGE_PROMPT.format(
                    style=ctx["style"],
                    max_length=ctx["max_length"],
                    partials=partials_text,
                ),
                system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            )

        ctx["output"] = {
            "summary":           final,
            "style":             ctx["style"],
            "windows_processed": len(partials),
        }
