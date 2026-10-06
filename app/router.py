"""
Query router for classifying questions as sql/rag/hybrid.

Uses the LLM to analyze the question and determine the best retrieval strategy.
"""

import logging
import re
from typing import Literal

from app.llm import get_llm_client
from app.prompts.router import get_router_prompt

logger = logging.getLogger(__name__)

RouteType = Literal["sql", "rag", "hybrid"]
DEFAULT_ROUTE: RouteType = "sql"


def parse_route(response: str) -> RouteType:
    """
    Map a raw LLM classification response to a route.

    Prefers an exact one-word answer, then falls back to keyword matching.
    Ambiguous responses default to SQL.
    """
    words = set(re.findall(r"[a-z]+", response.lower()))

    if "hybrid" in words or {"sql", "rag"} <= words:
        return "hybrid"
    if "sql" in words:
        return "sql"
    if "rag" in words:
        return "rag"

    logger.warning("Ambiguous route classification %r; defaulting to %s", response, DEFAULT_ROUTE)
    return DEFAULT_ROUTE


def route_query(question: str) -> RouteType:
    """Classify a question into the sql/rag/hybrid route."""
    response = get_llm_client().generate(get_router_prompt(question), temperature=0.0)
    return parse_route(response)
