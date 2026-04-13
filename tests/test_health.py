from fastapi.testclient import TestClient
from api.app import create_app

client = TestClient(create_app())


def test_health_status_code():
    # The /health endpoint must always be reachable
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body():
    # Must return exactly {"status": "ok"} — nothing more, nothing less
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_health_content_type():
    # FastAPI returns JSON by default
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
