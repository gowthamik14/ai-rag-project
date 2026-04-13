from unittest.mock import MagicMock, patch

from services.llm_service import OllamaLLMService, get_llm_service


# ── OllamaLLMService.chat ──────────────────────────────────────────────────

def test_chat_returns_model_content() -> None:
    # The service must return the text content from the LLM response
    with patch("services.llm_service.ChatOllama") as MockChatOllama:
        mock_llm = MockChatOllama.return_value
        mock_llm.invoke.return_value = MagicMock(content="Paris")

        service = OllamaLLMService()
        result = service.chat("What is the capital of France?")

    assert result == "Paris"


def test_chat_passes_message_as_human_message() -> None:
    # The service must wrap the message in a HumanMessage before calling invoke
    with patch("services.llm_service.ChatOllama") as MockChatOllama:
        mock_llm = MockChatOllama.return_value
        mock_llm.invoke.return_value = MagicMock(content="ok")

        service = OllamaLLMService()
        service.chat("Hello")

    args = mock_llm.invoke.call_args[0][0]
    assert args[0].content == "Hello"


def test_chat_calls_llm_invoke_once() -> None:
    with patch("services.llm_service.ChatOllama") as MockChatOllama:
        mock_llm = MockChatOllama.return_value
        mock_llm.invoke.return_value = MagicMock(content="ok")

        service = OllamaLLMService()
        service.chat("Hi")

    assert mock_llm.invoke.call_count == 1


def test_ollama_is_initialised_with_settings() -> None:
    # The service must read model and base_url from settings, not hardcode them
    with patch("services.llm_service.ChatOllama") as MockChatOllama:
        OllamaLLMService()
        _, kwargs = MockChatOllama.call_args
        assert "model" in kwargs
        assert "base_url" in kwargs


# ── get_llm_service ────────────────────────────────────────────────────────

def test_get_llm_service_returns_same_instance() -> None:
    # lru_cache must ensure a single shared instance
    get_llm_service.cache_clear()
    a = get_llm_service()
    b = get_llm_service()
    assert a is b
    get_llm_service.cache_clear()
