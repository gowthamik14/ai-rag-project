from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from rag.graph import BaseRAGGraph, get_rag_graph


# ── test double ────────────────────────────────────────────────────────────────

class FakeRAGGraph(BaseRAGGraph):
    """Returns a controlled knowledge string without touching BigQuery or Ollama."""

    def __init__(self, knowledge: str = '{"reasoning": "Authorised.", "verdict": "AUTHORISED"}') -> None:
        self.knowledge = knowledge
        self.received_question: str | None = None
        self.received_job_title: str | None = None
        self.received_model: str | None = None
        self.received_make: str | None = None
        self.call_count = 0

    def run(self, question: str, job_title: str, model: str, make: str) -> str:
        self.received_question = question
        self.received_job_title = job_title
        self.received_model = model
        self.received_make = make
        self.call_count += 1
        return self.knowledge


class ErrorRAGGraph(BaseRAGGraph):
    """Always raises — simulates a pipeline failure."""

    def run(self, question: str, job_title: str, model: str, make: str) -> str:
        raise RuntimeError("RAG pipeline failed")


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def fake_graph() -> FakeRAGGraph:
    return FakeRAGGraph(knowledge='{"reasoning": "Brake pads are covered.", "verdict": "AUTHORISED"}')


@pytest.fixture()
def client(fake_graph: FakeRAGGraph) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag_graph] = lambda: fake_graph
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def error_client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_rag_graph] = lambda: ErrorRAGGraph()
    return TestClient(app, raise_server_exceptions=False)


VALID_PAYLOAD = {
    "model": "CROSSLAND X HATCHBACK",
    "make": "VAUXHALL",
    "job_title": "ADAS Calibration",
    "job_cost": 150.0,
}


# ── happy path ─────────────────────────────────────────────────────────────────

def test_returns_200(client: TestClient) -> None:
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_response_contains_knowledge_field(client: TestClient) -> None:
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "knowledge" in response.json()


def test_response_contains_can_be_authorised_field(client: TestClient) -> None:
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "canbeAuthorised" in response.json()


def test_response_does_not_contain_reason_field(client: TestClient) -> None:
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "reason" not in response.json()


def test_knowledge_matches_rag_graph_output(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Oil change is routine.", "verdict": "AUTHORISED"}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["knowledge"] == "Oil change is routine."


def test_calls_rag_graph_exactly_once(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert fake_graph.call_count == 1


# ── forwarding to rag graph ────────────────────────────────────────────────────

def test_question_includes_job_title(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "ADAS Calibration" in fake_graph.received_question


def test_question_includes_model(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "CROSSLAND X HATCHBACK" in fake_graph.received_question


def test_question_includes_make(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "VAUXHALL" in fake_graph.received_question


def test_question_includes_cost(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert "150.00" in fake_graph.received_question


def test_job_title_forwarded_to_rag_graph(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert fake_graph.received_job_title == "ADAS Calibration"


def test_model_forwarded_to_rag_graph(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json={**VALID_PAYLOAD, "model": "BMW X5"})
    assert fake_graph.received_model == "BMW X5"


def test_make_forwarded_to_rag_graph(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    client.post("/authorised-knowledge", json={**VALID_PAYLOAD, "make": "BMW"})
    assert fake_graph.received_make == "BMW"


# ── verdict parsing ────────────────────────────────────────────────────────────

def test_can_be_authorised_true_when_verdict_authorised(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Job is within policy.", "verdict": "AUTHORISED"}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is True


def test_can_be_authorised_false_when_verdict_not_authorised(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Cost exceeds limit.", "verdict": "NOT AUTHORISED"}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is False


def test_can_be_authorised_false_when_verdict_missing_from_json(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Inconclusive."}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is False


def test_can_be_authorised_false_when_verdict_unrecognised(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Unknown.", "verdict": "MAYBE"}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is False


def test_knowledge_field_contains_reasoning_not_raw_json(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = '{"reasoning": "Job is routine.", "verdict": "AUTHORISED"}'
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["knowledge"] == "Job is routine."


def test_fallback_to_verdict_marker_when_response_is_not_json(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = "Job is within policy. VERDICT: AUTHORISED"
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is True


def test_fallback_verdict_marker_false_when_not_authorised(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = "Cost exceeds limit. VERDICT: NOT AUTHORISED"
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is False


def test_fallback_false_when_no_json_and_no_verdict_marker(
    client: TestClient, fake_graph: FakeRAGGraph
) -> None:
    fake_graph.knowledge = "Inconclusive response with no verdict."
    response = client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.json()["canbeAuthorised"] is False


# ── validation ─────────────────────────────────────────────────────────────────

def test_missing_model_returns_422(client: TestClient) -> None:
    response = client.post(
        "/authorised-knowledge",
        json={"make": "VAUXHALL", "job_title": "MOT", "job_cost": 50.0},
    )
    assert response.status_code == 422


def test_missing_make_returns_422(client: TestClient) -> None:
    response = client.post(
        "/authorised-knowledge",
        json={"model": "CROSSLAND X HATCHBACK", "job_title": "MOT", "job_cost": 50.0},
    )
    assert response.status_code == 422


def test_missing_job_title_returns_422(client: TestClient) -> None:
    response = client.post(
        "/authorised-knowledge",
        json={"model": "CROSSLAND X HATCHBACK", "make": "VAUXHALL", "job_cost": 50.0},
    )
    assert response.status_code == 422


def test_missing_job_cost_returns_422(client: TestClient) -> None:
    response = client.post(
        "/authorised-knowledge",
        json={"model": "CROSSLAND X HATCHBACK", "make": "VAUXHALL", "job_title": "MOT"},
    )
    assert response.status_code == 422


def test_non_numeric_cost_returns_422(client: TestClient) -> None:
    response = client.post(
        "/authorised-knowledge", json={**VALID_PAYLOAD, "job_cost": "expensive"}
    )
    assert response.status_code == 422


# ── error handling ─────────────────────────────────────────────────────────────

def test_rag_graph_error_returns_500(error_client: TestClient) -> None:
    response = error_client.post("/authorised-knowledge", json=VALID_PAYLOAD)
    assert response.status_code == 500
