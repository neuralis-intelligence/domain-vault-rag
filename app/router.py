"""
Query router for classifying questions as sql/rag/hybrid.

Uses LLM to analyze the question and determine the best retrieval strategy.
"""

from typing import Literal
import logging

from app.llm import get_llm_client
from app.prompts.router import get_router_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RouteType = Literal["sql", "rag", "hybrid"]


def route_query(question: str) -> RouteType:
    """
    Classify a question into sql/rag/hybrid route.

    Args:
        question: Natural language query

    Returns:
        "sql" for aggregation/counting queries
        "rag" for semantic/narrative queries
        "hybrid" for mixed queries
    """
    llm = get_llm_client()
    prompt = get_router_prompt(question)

    response = llm.generate(prompt, temperature=0.0)

    # Parse response
    response_lower = response.strip().lower()

    if "sql" in response_lower and "rag" not in response_lower:
        return "sql"
    elif "rag" in response_lower and "sql" not in response_lower:
        return "rag"
    elif "hybrid" in response_lower or ("sql" in response_lower and "rag" in response_lower):
        return "hybrid"
    else:
        # Default to SQL for ambiguous cases
        logger.warning(f"Ambiguous route classification: {response}. Defaulting to SQL.")
        return "sql"
