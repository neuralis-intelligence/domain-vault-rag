"""Tests for the LangGraph workflow (tools are mocked)."""

import asyncio

import pytest

from app.graph import ComplaintInsightsGraph, filters_from_sql_result

SQL_RESULT = {
    "sql": "SELECT issue, COUNT(*) AS n FROM t WHERE company = 'WELLS FARGO & COMPANY' "
    "GROUP BY issue ORDER BY n DESC LIMIT 1",
    "results": [{"issue": "Trouble during payment process", "n": 4113}],
}


FILTER_FIELDS = ("company", "product", "issue", "state")


class TestFiltersFromSQLResult:
    def test_uses_where_clause_and_top_row(self):
        assert filters_from_sql_result(SQL_RESULT, FILTER_FIELDS) == {
            "company": "WELLS FARGO & COMPANY",
            "issue": "Trouble during payment process",
        }

    def test_error_result_gives_no_filters(self):
        assert filters_from_sql_result({"error": "boom", "results": []}, FILTER_FIELDS) == {}

    def test_ignores_non_filterable_columns(self):
        result = {"sql": "SELECT 1 WHERE timely_response = 'Yes'", "results": [{"n": 1}]}
        assert filters_from_sql_result(result, FILTER_FIELDS) == {}

    def test_uses_the_lobs_filter_fields(self):
        assert filters_from_sql_result(SQL_RESULT, ("issue",)) == {
            "issue": "Trouble during payment process"
        }


@pytest.fixture
def mocked_tools(monkeypatch):
    calls = {"rag": [], "sql": [], "synth": [], "router": []}

    def fake_rag(question, lob, filters=None):
        calls["rag"].append((lob.id, filters))
        docs = [] if filters else [{"id": "1", "text": "late fee"}]
        return {"documents": docs, "count": len(docs)}

    def fake_sql(question, lob, model):
        calls["sql"].append((lob.id, model))
        return SQL_RESULT

    def fake_synth(**kw):
        calls["synth"].append(kw)
        return f"answer via {kw['route']}"

    monkeypatch.setattr("app.graph.execute_sql_query", fake_sql)
    monkeypatch.setattr("app.graph.execute_rag_query", fake_rag)
    monkeypatch.setattr("app.graph.synthesize_answer", fake_synth)
    return calls


def set_route(monkeypatch, calls, route):
    def fake_router(question, model=None, dataset=""):
        calls["router"].append((model, dataset))
        return route

    monkeypatch.setattr("app.graph.route_query", fake_router)


@pytest.mark.parametrize("route", ["sql", "rag", "hybrid"])
def test_graph_runs_each_route(monkeypatch, mocked_tools, registry, route):
    set_route(monkeypatch, mocked_tools, route)
    result = asyncio.run(
        ComplaintInsightsGraph().run(
            "question", lob=registry.get("cards"), model="m1", context={"user": "test"}
        )
    )

    assert result["route"] == route
    assert result["answer"] == f"answer via {route}"
    assert result["metadata"]["user"] == "test"
    assert result["metadata"]["route"] == route
    assert ("sql_result" in result["metadata"]) == (route in ("sql", "hybrid"))
    assert ("rag_result" in result["metadata"]) == (route in ("rag", "hybrid"))


def test_lob_and_model_reach_every_step(monkeypatch, mocked_tools, registry):
    set_route(monkeypatch, mocked_tools, "hybrid")
    asyncio.run(ComplaintInsightsGraph().run("q", lob=registry.get("mortgage"), model="m2"))

    assert mocked_tools["router"] == [("m2", "Mortgages")]
    assert mocked_tools["sql"] == [("mortgage", "m2")]
    assert {lob_id for lob_id, _ in mocked_tools["rag"]} == {"mortgage"}
    assert mocked_tools["synth"][0]["lob"].id == "mortgage"
    assert mocked_tools["synth"][0]["model"] == "m2"


def test_lob_without_rag_skips_router(monkeypatch, mocked_tools, registry):
    set_route(monkeypatch, mocked_tools, "rag")
    result = asyncio.run(ComplaintInsightsGraph().run("q", lob=registry.get("transactions")))

    assert result["route"] == "sql"
    assert mocked_tools["router"] == []
    assert mocked_tools["rag"] == []


def test_hybrid_falls_back_to_unfiltered_rag(monkeypatch, mocked_tools, registry):
    set_route(monkeypatch, mocked_tools, "hybrid")
    result = asyncio.run(ComplaintInsightsGraph().run("question", lob=registry.get("cards")))

    assert mocked_tools["rag"][0][1]["company"] == "WELLS FARGO & COMPANY"
    assert mocked_tools["rag"][1][1] is None
    assert result["metadata"]["rag_result"]["count"] == 1
