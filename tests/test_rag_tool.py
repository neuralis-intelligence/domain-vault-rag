"""
Tests for RAG tool.

Tests semantic search and filtering.
"""

import pytest
from app.rag_tool import RAGTool


class TestRAGTool:
    """Test cases for RAG tool."""

    @pytest.fixture
    def rag_tool(self):
        """Create RAG tool instance for testing."""
        # Note: This requires Qdrant to be running
        return RAGTool(
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="test_complaints"
        )

    def test_embed_query(self, rag_tool):
        """Test query embedding generation."""
        query = "fraud complaints"
        embedding = rag_tool.embed_query(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_build_filters_empty(self, rag_tool):
        """Test filter building with no filters."""
        filters = rag_tool.build_filters(None)
        assert filters is None

    def test_build_filters_company(self, rag_tool):
        """Test filter building with company filter."""
        filter_dict = {"company": "CHASE"}
        filters = rag_tool.build_filters(filter_dict)
        assert filters is not None

    def test_search_returns_results(self, rag_tool):
        """Test that search returns properly formatted results."""
        # This test requires actual data in Qdrant
        # Skip if not available
        pytest.skip("Requires Qdrant with test data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
