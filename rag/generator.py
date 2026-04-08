"""
Generator: builds the prompt from retrieved context and calls the LLM (Qwen via Ollama).

Supports:
  - Single-turn RAG answer
  - Multi-turn conversational RAG (with session history)
  - Streaming RAG answer
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from core.llm import OllamaLLM
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
# Prompt templates
# ------------------------------------------------------------------ #

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Use ONLY the provided context
to answer the user's question. If the answer is not in the context, say so clearly.
Do not make up information. Cite the source numbers (e.g., [1], [2]) when referring
to specific pieces of context."""

RAG_PROMPT_TEMPLATE = """\
Context information:
{context}

Question: {question}

Answer:"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a helpful conversational AI assistant with access
to retrieved documents. Use the context to answer questions accurately. Maintain conversation
continuity based on the message history."""

CONVERSATIONAL_PROMPT_TEMPLATE = """\
Context information:
{context}

Current question: {question}

Answer:"""


class Generator:
    def __init__(self, llm: Optional[OllamaLLM] = None) -> None:
        self._llm = llm or OllamaLLM()

    # ------------------------------------------------------------------ #
    # Single-turn
    # ------------------------------------------------------------------ #

    def generate(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate an answer grounded in the provided context."""
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._llm.generate(
            prompt=prompt,
            system_prompt=system_prompt or RAG_SYSTEM_PROMPT,
        )
        logger.info("Generated answer (len=%d) for question (len=%d)", len(answer), len(question))
        return answer

    # ------------------------------------------------------------------ #
    # Streaming
    # ------------------------------------------------------------------ #

    def stream_generate(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """Streaming version — yields tokens as they are produced."""
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        yield from self._llm.stream_generate(
            prompt=prompt,
            system_prompt=system_prompt or RAG_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------ #
    # Conversational (multi-turn)
    # ------------------------------------------------------------------ #

    def chat_generate(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Multi-turn answer generation.
        history: list of prior messages [{"role": "user"|"assistant", "content": "..."}]
        """
        messages: List[Dict[str, str]] = list(history or [])
        user_content = CONVERSATIONAL_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        messages.append({"role": "user", "content": user_content})

        answer = self._llm.chat(
            messages=messages,
            system_prompt=system_prompt or CONVERSATIONAL_SYSTEM_PROMPT,
        )
        return answer

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def build_prompt(self, question: str, context: str) -> str:
        """Expose the raw prompt for debugging / logging."""
        return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
