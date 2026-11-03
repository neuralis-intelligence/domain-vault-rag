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
        self.runs: list[tuple[str, str | None]] = []

    async def run(self, question, lob, model=None, context=None):
        if self.error:
            raise self.error
        self.questions.append(question)
        self.runs.append((lob.id, model))
        return {"answer": "42 complaints", "route": "sql", "metadata": {"route": "sql"}}


@pytest.fixture(autouse=True)
def installed_models(monkeypatch, registry):
    """Pretend two models are installed and use the test registry."""
    models = [
        {"name": "llama3.2", "size_gb": 2.0, "parameter_size": "3.2B"},
        {"name": "qwen2.5:7b-instruct", "size_gb": 4.7, "parameter_size": "7.6B"},
    ]
    monkeypatch.setattr("app.llm.list_models", lambda: models)
    monkeypatch.setattr("app.main.list_models", lambda: models)
    test_settings = app_settings(ollama_model="qwen2.5:7b-instruct")
    monkeypatch.setattr("app.llm.settings", test_settings)
    monkeypatch.setattr("app.main.settings", test_settings)
    monkeypatch.setattr("app.main.get_registry", lambda: registry)
    return models


def app_settings(**overrides):
    from app.config import Settings

    return Settings(**overrides)


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
    monkeypatch.setattr(
        "app.main.get_sql_tool", lambda: type("S", (), {"is_available": lambda s: True})()
    )

    data = client.get("/health").json()
    assert data["ollama"] == "connected"
    assert data["qdrant"] == "unavailable"
    assert data["duckdb"] == "connected"
    assert data["status"] == "degraded"


def test_ask_returns_graph_answer(client, fake_graph):
    response = client.post("/ask", json={"question": "How many complaints did Chase receive?"})
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "question": "How many complaints did Chase receive?",
        "answer": "42 complaints",
        "route": "sql",
        "lob": "cards",
        "model": "qwen2.5:7b-instruct",
        "metadata": {"route": "sql"},
    }
    assert fake_graph.questions == ["How many complaints did Chase receive?"]
    assert fake_graph.runs == [("cards", "qwen2.5:7b-instruct")]


def test_ask_with_chosen_lob_and_model(client, fake_graph):
    response = client.post("/ask", json={"question": "q", "lob": "mortgage", "model": "llama3.2"})
    assert response.status_code == 200
    assert response.json()["lob"] == "mortgage"
    assert fake_graph.runs == [("mortgage", "llama3.2")]


def test_ask_rejects_unknown_lob(client, fake_graph):
    response = client.post("/ask", json={"question": "q", "lob": "crypto"})
    assert response.status_code == 400
    assert "Unknown line of business 'crypto'" in response.json()["detail"]
    assert fake_graph.runs == []


def test_ask_rejects_uninstalled_model(client, fake_graph):
    response = client.post("/ask", json={"question": "q", "model": "mistral"})
    assert response.status_code == 400
    assert "ollama pull mistral" in response.json()["detail"]
    assert fake_graph.runs == []


def test_models_endpoint(client):
    data = client.get("/models").json()
    assert data["default"] == "qwen2.5:7b-instruct"
    assert [m["name"] for m in data["models"]] == ["llama3.2", "qwen2.5:7b-instruct"]


def test_models_endpoint_when_ollama_down(client, monkeypatch):
    def down():
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("app.main.list_models", down)
    assert client.get("/models").status_code == 503


def test_lobs_endpoint(client, monkeypatch, built_db):
    from app.sql_tool import SQLTool

    monkeypatch.setattr("app.main.get_sql_tool", lambda: SQLTool(db_path=built_db))
    monkeypatch.setattr(
        "app.main.get_rag_tool",
        lambda: type("R", (), {"collection_exists": lambda s, c: c == "complaint_narratives"})(),
    )
    data = client.get("/lobs").json()
    by_id = {lob["id"]: lob for lob in data["lobs"]}

    assert data["default"] == "cards"
    assert by_id["cards"]["rows"] == 3
    assert by_id["cards"]["sql_available"] and by_id["cards"]["rag_available"]
    assert by_id["cards"]["example_questions"] == ["Top issue?"]
    assert by_id["transactions"]["sql_available"] and not by_id["transactions"]["rag_available"]
    assert not by_id["not_pulled"]["sql_available"]


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
