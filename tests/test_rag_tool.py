"""Tests for the RAG tool."""

from types import SimpleNamespace

import pytest

from app.rag_tool import RAGTool


@pytest.fixture
def rag_tool():
    # QdrantClient does not connect on construction and the embedding model is lazy,
    # so this needs neither Qdrant nor a model download.
    return RAGTool(qdrant_host="localhost", qdrant_port=6333, collection_name="test_complaints")


class TestBuildFilters:
    def test_none(self, rag_tool):
        assert rag_tool.build_filters(None) is None

    def test_company(self, rag_tool):
        f = rag_tool.build_filters({"company": "JPMORGAN CHASE & CO."})
        assert len(f.must) == 1
        assert f.must[0].key == "company"
        assert f.must[0].match.value == "JPMORGAN CHASE & CO."

    def test_ignores_empty_and_unknown_fields(self, rag_tool):
        assert rag_tool.build_filters({"company": "", "narrative": "x"}) is None


class TestSearch:
    def test_formats_results(self, rag_tool, monkeypatch):
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

        result = rag_tool.search("double charge", filters={"product": "Credit card"})

        assert result["count"] == 1
        assert result["documents"][0]["complaint_id"] == "123"
        assert result["documents"][0]["score"] == 0.83
        assert captured["query_filter"].must[0].key == "product"
        assert captured["collection_name"] == "test_complaints"

    def test_errors_are_returned_not_raised(self, rag_tool, monkeypatch):
        monkeypatch.setattr(rag_tool, "embed_query", lambda q: [0.0])

        def boom(**kwargs):
            raise ConnectionError("qdrant down")

        monkeypatch.setattr(rag_tool.qdrant_client, "query_points", boom)
        result = rag_tool.search("x")
        assert result["documents"] == []
        assert "qdrant down" in result["error"]


@pytest.mark.integration
class TestRAGLive:
    """Downloads the embedding model; search needs Qdrant with embedded data."""

    def test_embed_query(self, rag_tool):
        embedding = rag_tool.embed_query("fraud complaints")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_search_live(self):
        result = RAGTool().search("unauthorized charges on my card")
        assert "error" not in result, result.get("error")
        assert result["count"] > 0
