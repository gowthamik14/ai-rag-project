import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from services.llm_service import LLMService, get_llm_service


# ── test double ────────────────────────────────────────────────────────────

class FakeLLMService(LLMService):
    """In-memory stub — implements the same contract as OllamaLLMService."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.received_message: str | None = None
        self.call_count = 0

    def chat(self, message: str) -> str:
        self.received_message = message
        self.call_count += 1
        return self.reply


class ErrorLLMService(LLMService):
    """Always raises — simulates Ollama being unreachable."""

    def chat(self, message: str) -> str:
        raise ConnectionError("Ollama unreachable")


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_service() -> FakeLLMService:
    return FakeLLMService(reply="mocked reply")


@pytest.fixture()
def client(fake_service: FakeLLMService) -> TestClient:
    """App with the real LLM dependency swapped for FakeLLMService."""
    app = create_app()
    app.dependency_overrides[get_llm_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def error_client() -> TestClient:
    """App wired to a service that always raises."""
    app = create_app()
    app.dependency_overrides[get_llm_service] = lambda: ErrorLLMService()
    return TestClient(app, raise_server_exceptions=False)


# ── happy path ─────────────────────────────────────────────────────────────

def test_chat_returns_200(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hi"})
    assert response.status_code == 200


def test_chat_reply_matches_service_output(
    client: TestClient, fake_service: FakeLLMService
) -> None:
    fake_service.reply = "The answer is 4"
    response = client.post("/chat", json={"message": "What is 2 + 2?"})
    assert response.json()["reply"] == "The answer is 4"


def test_chat_response_contains_reply_key(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "Hello"})
    assert "reply" in response.json()


def test_chat_forwards_message_to_service(
    client: TestClient, fake_service: FakeLLMService
) -> None:
    client.post("/chat", json={"message": "Tell me a joke"})
    assert fake_service.received_message == "Tell me a joke"


def test_chat_calls_service_exactly_once(
    client: TestClient, fake_service: FakeLLMService
) -> None:
    client.post("/chat", json={"message": "Hello"})
    assert fake_service.call_count == 1


# ── validation ─────────────────────────────────────────────────────────────

def test_chat_missing_message_returns_422(client: TestClient) -> None:
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_wrong_content_type_returns_422(client: TestClient) -> None:
    response = client.post(
        "/chat", content="hello", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422


def test_chat_empty_string_message_is_accepted(client: TestClient) -> None:
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 200


def test_chat_long_message_is_accepted(client: TestClient) -> None:
    response = client.post("/chat", json={"message": "a" * 10_000})
    assert response.status_code == 200


# ── error handling ──────────────────────────────────────────────────────────

def test_chat_service_error_returns_500(error_client: TestClient) -> None:
    response = error_client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 500
