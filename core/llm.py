"""
OllamaLLM — LLM connector via Ollama's HTTP API.

Supports any model served by Ollama. Default: qwen2.5:7b

Start Ollama:
  OLLAMA_NO_GPU=1 ollama serve      ← required on Macs with Metal/GPU issues

Pull the model first:
  ollama pull qwen2.5:7b

Public interface:
  - generate(prompt)          → str
  - chat(messages)            → str   (OpenAI-style message list)
  - stream_generate(prompt)   → Iterator[str]
  - health_check()            → bool
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import httpx

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM:
    """Thin async-free client for any model served by a local Ollama instance."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model       = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens  = max_tokens or settings.llm_max_tokens
        self.base_url    = (base_url or settings.ollama_base_url).rstrip("/")
        self._client     = httpx.Client(timeout=settings.llm_timeout)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Single-turn text generation. Returns the full response string."""
        payload = self._build_generate_payload(prompt, system_prompt)
        logger.debug("LLM generate | model=%s | prompt_len=%d", self.model, len(prompt))

        response = self._client.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()

        # Ollama streams NDJSON — aggregate all tokens into one string
        full_text = ""
        for line in response.text.strip().splitlines():
            if line:
                chunk = json.loads(line)
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break

        logger.debug("LLM generate | response_len=%d", len(full_text))
        return full_text.strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Multi-turn chat completion.
        messages: [{"role": "user"|"assistant", "content": "..."}]
        """
        payload = self._build_chat_payload(messages, system_prompt)
        logger.debug("LLM chat | model=%s | turns=%d", self.model, len(messages))

        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()

        full_text = ""
        for line in response.text.strip().splitlines():
            if line:
                chunk = json.loads(line)
                full_text += chunk.get("message", {}).get("content", "")
                if chunk.get("done"):
                    break

        return full_text.strip()

    def stream_generate(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Streaming generation — yields token strings as they arrive."""
        payload = self._build_generate_payload(prompt, system_prompt)

        with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the configured model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model.split(":")[0] in m for m in models)
            if not available:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s", self.model, models
                )
            return available
        except Exception as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_generate_payload(
        self, prompt: str, system_prompt: Optional[str]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _build_chat_payload(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
    ) -> Dict[str, Any]:
        chat_messages = []
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.extend(messages)

        return {
            "model": self.model,
            "messages": chat_messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
