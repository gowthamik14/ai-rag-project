from abc import ABC, abstractmethod
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config.settings import settings


class LLMService(ABC):
    """Abstract contract for LLM communication.

    Depend on this, not on ChatOllama directly, so the LLM backend
    can be swapped (Ollama → OpenAI, etc.) without touching route code.
    """

    @abstractmethod
    def chat(self, message: str) -> str:
        """Send a user message and return the model's reply."""


class OllamaLLMService(LLMService):
    """Concrete implementation that talks to a local Ollama instance."""

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            format="json",   # forces Ollama to emit valid JSON — no markdown fences or prose
            temperature=0,   # deterministic output — same input always produces same result
        )

    def chat(self, message: str) -> str:
        response = self._llm.invoke([HumanMessage(content=message)])
        return response.content


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """FastAPI dependency — returns a single shared OllamaLLMService instance."""
    return OllamaLLMService()
