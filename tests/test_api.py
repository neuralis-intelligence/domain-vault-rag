"""Tests for the FastAPI backend (graph and services mocked)."""

import pytest
import requests
from fastapi.testclient import TestClient

from app.graph import get_graph
from app.main import app


class FakeGraph:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.questions: list[str] = []

    async def run(self, question, context=None):
        if self.error:
            raise self.error
        self.questions.append(question)
        return {"answer": "42 complaints", "route": "sql", "metadata": {"route": "sql"}}


@pytest.fixture
def fake_graph():
    return FakeGraph()


@pytest.fixture
def client(fake_graph):
    app.dependency_overrides[get_graph] = lambda: fake_graph
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Complaint Insights API"


def test_health_reports_each_service(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.get_llm_client", lambda: type("L", (), {"is_available": lambda s: True})()
    )
    monkeypatch.setattr(
        "app.main.get_rag_tool", lambda: type("R", (), {"is_available": lambda s: False})()
    )

    data = client.get("/health").json()
    assert data["ollama"] == "connected"
    assert data["qdrant"] == "unavailable"
    assert data["status"] == "degraded"
    assert "bigquery" in data


def test_ask_returns_graph_answer(client, fake_graph):
    response = client.post("/ask", json={"question": "How many complaints did Chase receive?"})
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "question": "How many complaints did Chase receive?",
        "answer": "42 complaints",
        "route": "sql",
        "metadata": {"route": "sql"},
    }
    assert fake_graph.questions == ["How many complaints did Chase receive?"]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_returns_503_when_llm_unreachable(client):
    app.dependency_overrides[get_graph] = lambda: FakeGraph(requests.ConnectionError("refused"))
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "Is Ollama running?" in response.json()["detail"]


def test_ask_returns_500_when_graph_fails(client):
    app.dependency_overrides[get_graph] = lambda: FakeGraph(RuntimeError("boom"))
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 500
    assert response.json()["detail"] == "boom"


def test_feedback_endpoint(client):
    payload = {"question": "Q", "answer": "A", "rating": 5, "comments": "Great!"}
    assert client.post("/feedback", json=payload).status_code == 200


def test_feedback_rejects_bad_rating(client):
    payload = {"question": "Q", "answer": "A", "rating": 9}
    assert client.post("/feedback", json=payload).status_code == 422
