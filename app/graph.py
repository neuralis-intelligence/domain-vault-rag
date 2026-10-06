"""
LangGraph workflow for routing and orchestrating SQL/RAG/hybrid queries.

1. Router: classify the query as sql/rag/hybrid
2. SQL tool: text-to-SQL on BigQuery
3. RAG tool: semantic search on Qdrant
4. Hybrid: SQL first, then RAG filtered by the SQL result
5. Synthesis: generate the final answer with the LLM
"""

import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.llm import synthesize_answer
from app.rag_tool import FILTERABLE_FIELDS, execute_rag_query
from app.router import route_query
from app.sql_tool import execute_sql_query

logger = logging.getLogger(__name__)

RouteType = Literal["sql", "rag", "hybrid"]


class GraphState(TypedDict):
    """State passed through the LangGraph workflow."""

    question: str
    route: RouteType
    sql_result: dict[str, Any] | None
    rag_result: dict[str, Any] | None
    answer: str
    metadata: dict[str, Any]


def filters_from_sql_result(sql_result: dict[str, Any]) -> dict[str, str]:
    """
    Derive RAG metadata filters from a SQL result.

    Uses exact-match conditions in the SQL (e.g. company = 'WELLS FARGO & COMPANY')
    and, where present, filterable columns in the top result row (e.g. the top issue).
    """
    if not sql_result or sql_result.get("error"):
        return {}

    filters: dict[str, str] = {}
    for field, value in re.findall(r"\b(\w+)\s*=\s*'([^']*)'", sql_result.get("sql", "")):
        if field.lower() in FILTERABLE_FIELDS:
            filters[field.lower()] = value

    rows = sql_result.get("results") or []
    if rows:
        for field in FILTERABLE_FIELDS:
            value = rows[0].get(field)
            if isinstance(value, str) and value:
                filters[field] = value

    return filters


def route_node(state: GraphState) -> dict[str, Any]:
    """Route the query to sql/rag/hybrid."""
    route = route_query(state["question"])
    logger.info("Route selected: %s", route)
    return {"route": route, "metadata": {**state["metadata"], "route": route}}


def sql_node(state: GraphState) -> dict[str, Any]:
    """Execute the SQL path."""
    return {"sql_result": execute_sql_query(state["question"])}


def rag_node(state: GraphState) -> dict[str, Any]:
    """Execute the RAG path."""
    return {"rag_result": execute_rag_query(state["question"])}


def hybrid_node(state: GraphState) -> dict[str, Any]:
    """Execute SQL, then RAG filtered by what SQL found."""
    sql_result = execute_sql_query(state["question"])
    filters = filters_from_sql_result(sql_result)
    logger.info("Hybrid RAG filters: %s", filters)

    rag_result = execute_rag_query(state["question"], filters=filters or None)
    if filters and not rag_result.get("documents"):
        logger.info("No narratives matched the SQL filters; retrying unfiltered")
        rag_result = execute_rag_query(state["question"])

    return {"sql_result": sql_result, "rag_result": rag_result}


def synthesize_node(state: GraphState) -> dict[str, Any]:
    """Synthesize the final answer using the LLM."""
    answer = synthesize_answer(
        question=state["question"],
        sql_result=state.get("sql_result"),
        rag_result=state.get("rag_result"),
        route=state["route"],
    )
    return {"answer": answer}


def build_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("route", route_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("hybrid", hybrid_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"sql": "sql", "rag": "rag", "hybrid": "hybrid"},
    )
    for node in ("sql", "rag", "hybrid"):
        workflow.add_edge(node, "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


class ComplaintInsightsGraph:
    """Thin wrapper around the compiled LangGraph workflow."""

    def __init__(self):
        self.graph = build_graph()

    async def run(self, question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the full workflow and return the answer, route, and retrieved data."""
        initial_state = GraphState(
            question=question,
            route="sql",  # overwritten by the router
            sql_result=None,
            rag_result=None,
            answer="",
            metadata=dict(context or {}),
        )

        final_state = await self.graph.ainvoke(initial_state)

        metadata = dict(final_state["metadata"])
        if final_state.get("sql_result") is not None:
            metadata["sql_result"] = final_state["sql_result"]
        if final_state.get("rag_result") is not None:
            metadata["rag_result"] = final_state["rag_result"]

        return {
            "answer": final_state["answer"],
            "route": final_state["route"],
            "metadata": metadata,
        }


_graph_instance: ComplaintInsightsGraph | None = None


def get_graph() -> ComplaintInsightsGraph:
    """Get or create the shared graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ComplaintInsightsGraph()
    return _graph_instance
