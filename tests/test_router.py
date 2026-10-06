"""Tests for the sql/rag/hybrid query router."""

import pytest

from app.router import parse_route, route_query


class TestParseRoute:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("sql", "sql"),
            ("SQL", "sql"),
            ("  rag\n", "rag"),
            ("hybrid", "hybrid"),
            ("The answer is: sql.", "sql"),
            ("**RAG**", "rag"),
            ("sql and rag", "hybrid"),
            ("I'm not sure", "sql"),
            ("", "sql"),
        ],
    )
    def test_parse_route(self, response, expected):
        assert parse_route(response) == expected

    def test_substrings_do_not_count(self):
        # "drag" contains "rag" but is not the word "rag"
        assert parse_route("drag") == "sql"


class TestRouteQuery:
    def test_uses_llm_with_zero_temperature(self, fake_llm):
        fake_llm.responses = ["rag"]
        assert route_query("What are customers saying about fraud?") == "rag"
        assert fake_llm.calls[0]["temperature"] == 0.0
        assert "What are customers saying about fraud?" in fake_llm.calls[0]["prompt"]


@pytest.mark.integration
class TestRouterLive:
    """Requires Ollama running with OLLAMA_MODEL pulled."""

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("How many complaints did Bank of America receive?", "sql"),
            ("What is the most common issue at Chase?", "sql"),
            ("What are customers saying about fraud?", "rag"),
            ("Show me examples of foreclosure complaints", "rag"),
            ("What's the top issue at Wells Fargo and what are people saying?", "hybrid"),
        ],
    )
    def test_live_routing(self, question, expected):
        assert route_query(question) == expected
