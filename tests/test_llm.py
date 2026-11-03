"""Tests for the Ollama client and context building."""

import pytest

from app import llm
from app.llm import ModelNotAvailableError, OllamaClient, build_context, list_models, resolve_model


class FakeResponse:
    ok = True

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "hello"}


def test_explicit_zero_temperature_is_respected(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return FakeResponse()

    monkeypatch.setattr("app.llm.requests.post", fake_post)
    client = OllamaClient(host="http://ollama:11434", model="m", temperature=0.1)

    assert client.generate("hi", temperature=0.0) == "hello"
    assert sent["options"]["temperature"] == 0.0


def test_build_context_includes_sql_and_citations():
    context = build_context(
        sql_result={"sql": "SELECT 1", "results": [{"issue": "Fees", "n": 3}]},
        rag_result={
            "documents": [
                {
                    "id": "99",
                    "text": "they charged me",
                    "fields": {"company": "C", "issue": "I", "date_received": "2024-01-01"},
                }
            ]
        },
        doc_label="Complaint",
    )
    assert "SELECT 1" in context
    assert "'issue': 'Fees'" in context
    assert "- [Complaint 99] C / I: they charged me" in context


def test_build_context_reports_sql_error():
    assert "SQL query failed: boom" in build_context(sql_result={"error": "boom"})


def test_build_context_empty():
    assert build_context() == "No data was retrieved."


def test_one_client_per_model(monkeypatch):
    monkeypatch.setattr(llm, "_llm_clients", {})
    a, b = llm.get_llm_client("a"), llm.get_llm_client("b")
    assert (a.model, b.model) == ("a", "b")
    assert llm.get_llm_client("a") is a


TAGS = {
    "models": [
        {
            "name": "qwen2.5:7b",
            "size": 4.7e9,
            "details": {"family": "qwen2", "parameter_size": "7.6B"},
        },
        {
            "name": "llama3.2:3b",
            "size": 2.0e9,
            "details": {"family": "llama", "parameter_size": "3.2B"},
        },
        {"name": "llama3.1:70b", "size": 40e9, "details": {"family": "llama"}},
        {"name": "nomic-embed-text", "size": 0.3e9, "details": {"families": ["nomic-bert"]}},
    ]
}


@pytest.fixture
def ollama_tags(monkeypatch):
    class Tags(FakeResponse):
        def json(self):
            return TAGS

    monkeypatch.setattr("app.llm.requests.get", lambda url, timeout: Tags())


def _settings(monkeypatch, **overrides):
    monkeypatch.setattr(llm, "settings", llm.settings.__class__(**overrides))


def test_list_models_hides_embedders_and_large_models(monkeypatch, ollama_tags):
    _settings(monkeypatch, allowed_models=(), max_model_size_gb=10)
    names = [m["name"] for m in list_models()]
    assert names == ["llama3.2:3b", "qwen2.5:7b"]  # smallest first


def test_list_models_allowlist(monkeypatch, ollama_tags):
    _settings(monkeypatch, allowed_models=("qwen2.5:7b", "not-installed"))
    assert [m["name"] for m in list_models()] == ["qwen2.5:7b"]


def test_resolve_model(monkeypatch, ollama_tags):
    _settings(monkeypatch, ollama_model="qwen2.5:7b", allowed_models=(), max_model_size_gb=10)
    assert resolve_model(None) == "qwen2.5:7b"
    assert resolve_model("llama3.2:3b") == "llama3.2:3b"
    with pytest.raises(ModelNotAvailableError, match="ollama pull mistral"):
        resolve_model("mistral")
