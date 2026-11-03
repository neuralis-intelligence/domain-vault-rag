"""Tests for the RAG tool."""

from types import SimpleNamespace

import pytest

from app.lob import RagConfig, get_lob
from app.rag_tool import RAGTool, execute_rag_query


@pytest.fixture
def rag_tool():
    # QdrantClient does not connect on construction and the embedding model is lazy,
    # so this needs neither Qdrant nor a model download.
    return RAGTool(qdrant_host="localhost", qdrant_port=6333)


@pytest.fixture
def rag(registry):
    return registry.get("cards").rag


class TestBuildFilters:
    def test_fixed_filter_always_applies(self, rag_tool, rag):
        f = rag_tool.build_filters(rag, None)
        assert [(c.key, c.match.value) for c in f.must] == [("product", "Credit card")]

    def test_request_filters_are_added(self, rag_tool, rag):
        f = rag_tool.build_filters(rag, {"company": "JPMORGAN CHASE & CO."})
        assert {(c.key, c.match.value) for c in f.must} == {
            ("company", "JPMORGAN CHASE & CO."),
            ("product", "Credit card"),
        }

    def test_fixed_filter_wins_over_request(self, rag_tool, rag):
        f = rag_tool.build_filters(rag, {"product": "Mortgage"})
        assert [(c.key, c.match.value) for c in f.must] == [("product", "Credit card")]

    def test_ignores_empty_and_unknown_fields(self, rag_tool, rag):
        f = rag_tool.build_filters(rag, {"company": "", "narrative": "x"})
        assert [c.key for c in f.must] == ["product"]

    def test_no_filters_at_all(self, rag_tool):
        assert rag_tool.build_filters(RagConfig("c", "text", "id"), {"x": "y"}) is None


class TestSearch:
    def test_formats_results(self, rag_tool, rag, monkeypatch):
        monkeypatch.setattr(rag_tool, "embed_query", lambda q: [0.0] * 384)
        point = SimpleNamespace(
            score=0.83,
            payload={
                "complaint_id": "123",
                "narrative": "They charged me twice",
                "company": "CITIBANK, N.A.",
                "product": "Credit card",
                "issue": "Fees or interest",
                "state": "NY",
                "date_received": "2024-01-02",
            },
        )
        captured = {}

        def fake_query_points(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(points=[point])

        monkeypatch.setattr(rag_tool.qdrant_client, "query_points", fake_query_points)

        result = rag_tool.search("double charge", rag, filters={"state": "NY"})

        doc = result["documents"][0]
        assert result["count"] == 1
        assert doc["id"] == "123"
        assert doc["text"] == "They charged me twice"
        assert doc["fields"]["company"] == "CITIBANK, N.A."
        assert doc["fields"]["sub_issue"] is None
        assert doc["score"] == 0.83
        assert captured["collection_name"] == "complaint_narratives"
        assert {c.key for c in captured["query_filter"].must} == {"state", "product"}

    def test_errors_are_returned_not_raised(self, rag_tool, rag, monkeypatch):
        monkeypatch.setattr(rag_tool, "embed_query", lambda q: [0.0])

        def boom(**kwargs):
            raise ConnectionError("qdrant down")

        monkeypatch.setattr(rag_tool.qdrant_client, "query_points", boom)
        result = rag_tool.search("x", rag)
        assert result["documents"] == []
        assert "qdrant down" in result["error"]


def test_lob_without_rag_returns_error(registry):
    result = execute_rag_query("q", registry.get("transactions"))
    assert "no text search" in result["error"]
    assert result["documents"] == []


@pytest.mark.integration
class TestRAGLive:
    """Downloads the embedding model; search needs Qdrant with embedded data."""

    def test_embed_query(self, rag_tool):
        embedding = rag_tool.embed_query("fraud complaints")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_search_live(self):
        result = RAGTool().search("unauthorized charges on my card", get_lob("credit_card").rag)
        assert "error" not in result, result.get("error")
        assert result["count"] > 0
