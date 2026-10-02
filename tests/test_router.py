"""
Tests for query router.

Tests routing logic for sql/rag/hybrid classification.
"""

import pytest
from app.router import route_query


class TestRouter:
    """Test cases for query routing."""

    def test_sql_route_counting(self):
        """Test that counting questions route to SQL."""
        question = "How many complaints did Bank of America receive?"
        route = route_query(question)
        assert route == "sql", f"Expected 'sql', got '{route}'"

    def test_sql_route_aggregation(self):
        """Test that aggregation questions route to SQL."""
        question = "What is the most common issue at Chase?"
        route = route_query(question)
        assert route == "sql", f"Expected 'sql', got '{route}'"

    def test_rag_route_narrative(self):
        """Test that narrative questions route to RAG."""
        question = "What are customers saying about fraud?"
        route = route_query(question)
        assert route == "rag", f"Expected 'rag', got '{route}'"

    def test_rag_route_examples(self):
        """Test that example requests route to RAG."""
        question = "Show me examples of foreclosure complaints"
        route = route_query(question)
        assert route == "rag", f"Expected 'rag', got '{route}'"

    def test_hybrid_route(self):
        """Test that hybrid questions route correctly."""
        question = "What's the top issue at Wells Fargo and what are people saying?"
        route = route_query(question)
        assert route == "hybrid", f"Expected 'hybrid', got '{route}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
