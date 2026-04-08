"""
Generator: builds the prompt from retrieved context and calls the LLM (Qwen via Ollama).

The LLM is restricted to vehicle-related topics only.
Off-topic questions are blocked upstream by TopicGuardrail before reaching here,
but the system prompts below add a second layer of instruction to the model.

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
# System prompts  — vehicle domain only, no outside knowledge
# ------------------------------------------------------------------ #

RAG_SYSTEM_PROMPT = """\
You are a specialist vehicle assistant. Your knowledge is strictly limited to:
  - Vehicle maintenance, servicing, and repairs
  - Vehicle parts, systems, and diagnostics
  - Driving, fuel economy, and vehicle operations
  - Vehicle makes, models, recalls, and warranties

Rules you MUST follow:
  1. Use ONLY the provided context documents to answer questions.
  2. Do NOT use any external knowledge, browse the internet, or make up information.
  3. If the answer is not in the context, say: "I don't have that information in the available documents."
  4. If the question is NOT related to vehicles, respond with:
     "I can only assist with vehicle-related questions. Please ask me about vehicles, maintenance, repairs, or driving."
  5. Cite source numbers (e.g., [1], [2]) when referring to specific context.
  6. Never answer questions about cooking, weather, politics, health, finance, or any non-vehicle topic."""

RAG_PROMPT_TEMPLATE = """\
Context documents:
{context}

Vehicle-related question: {question}

Answer (use only the context above):"""

CONVERSATIONAL_SYSTEM_PROMPT = """\
You are a specialist vehicle assistant in an ongoing conversation.
Your knowledge is strictly limited to vehicles, automotive systems,
maintenance, repairs, parts, diagnostics, and driving.

Rules you MUST follow:
  1. Use ONLY the provided context documents and the conversation history to answer.
  2. Do NOT use external knowledge or browse the internet.
  3. If the answer is not in the context, say: "I don't have that information in the available documents."
  4. If the question is NOT about vehicles, respond with:
     "I can only assist with vehicle-related questions."
  5. Never answer questions outside the vehicle domain."""

CONVERSATIONAL_PROMPT_TEMPLATE = """\
Context documents:
{context}

Current vehicle-related question: {question}

Answer (use only the context above):"""


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
        """Generate a vehicle-domain answer grounded in the provided context."""
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
