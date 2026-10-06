"""Tests for the Ollama client and context building."""

from app.llm import OllamaClient, build_context


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
            "documents": [{"complaint_id": "99", "company": "C", "issue": "I", "narrative": "text"}]
        },
    )
    assert "SELECT 1" in context
    assert "'issue': 'Fees'" in context
    assert "[Complaint 99]" in context


def test_build_context_reports_sql_error():
    assert "SQL query failed: boom" in build_context(sql_result={"error": "boom"})


def test_build_context_empty():
    assert build_context() == "No data was retrieved."
