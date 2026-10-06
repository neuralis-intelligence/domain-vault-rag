"""Tests for the LangGraph workflow (tools are mocked)."""

import asyncio

import pytest

from app.graph import ComplaintInsightsGraph, filters_from_sql_result

SQL_RESULT = {
    "sql": "SELECT issue, COUNT(*) AS n FROM t WHERE company = 'WELLS FARGO & COMPANY' "
    "GROUP BY issue ORDER BY n DESC LIMIT 1",
    "results": [{"issue": "Trouble during payment process", "n": 4113}],
}


class TestFiltersFromSQLResult:
    def test_uses_where_clause_and_top_row(self):
        assert filters_from_sql_result(SQL_RESULT) == {
            "company": "WELLS FARGO & COMPANY",
            "issue": "Trouble during payment process",
        }

    def test_error_result_gives_no_filters(self):
        assert filters_from_sql_result({"error": "boom", "results": []}) == {}

    def test_ignores_non_filterable_columns(self):
        result = {"sql": "SELECT 1 WHERE timely_response = 'Yes'", "results": [{"n": 1}]}
        assert filters_from_sql_result(result) == {}


@pytest.fixture
def mocked_tools(monkeypatch):
    calls = {"rag_filters": []}

    def fake_rag(question, filters=None):
        calls["rag_filters"].append(filters)
        docs = [] if filters else [{"complaint_id": "1", "narrative": "late fee"}]
        return {"documents": docs, "count": len(docs)}

    monkeypatch.setattr("app.graph.execute_sql_query", lambda q: SQL_RESULT)
    monkeypatch.setattr("app.graph.execute_rag_query", fake_rag)
    monkeypatch.setattr("app.graph.synthesize_answer", lambda **kw: f"answer via {kw['route']}")
    return calls


@pytest.mark.parametrize("route", ["sql", "rag", "hybrid"])
def test_graph_runs_each_route(monkeypatch, mocked_tools, route):
    monkeypatch.setattr("app.graph.route_query", lambda q: route)
    result = asyncio.run(ComplaintInsightsGraph().run("question", context={"user": "test"}))

    assert result["route"] == route
    assert result["answer"] == f"answer via {route}"
    assert result["metadata"]["user"] == "test"
    assert result["metadata"]["route"] == route
    assert ("sql_result" in result["metadata"]) == (route in ("sql", "hybrid"))
    assert ("rag_result" in result["metadata"]) == (route in ("rag", "hybrid"))


def test_hybrid_falls_back_to_unfiltered_rag(monkeypatch, mocked_tools):
    monkeypatch.setattr("app.graph.route_query", lambda q: "hybrid")
    result = asyncio.run(ComplaintInsightsGraph().run("question"))

    assert mocked_tools["rag_filters"][0]["company"] == "WELLS FARGO & COMPANY"
    assert mocked_tools["rag_filters"][1] is None
    assert result["metadata"]["rag_result"]["count"] == 1
