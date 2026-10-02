"""
LangGraph workflow for routing and orchestrating SQL/RAG/hybrid queries.

This module implements the agent logic:
1. Router: Classify query as sql/rag/hybrid
2. SQL Tool: Execute text-to-SQL on BigQuery
3. RAG Tool: Semantic search on Qdrant
4. Hybrid: Combine SQL (filter) + RAG (narratives)
5. Synthesis: Generate final answer with LLM
"""

from typing import TypedDict, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, END
import logging

from app.router import route_query
from app.sql_tool import execute_sql_query
from app.rag_tool import execute_rag_query
from app.llm import synthesize_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State passed through the LangGraph workflow."""
    question: str
    route: Literal["sql", "rag", "hybrid"]
    sql_result: Optional[Dict[str, Any]]
    rag_result: Optional[Dict[str, Any]]
    answer: str
    metadata: Dict[str, Any]


class ComplaintInsightsGraph:
    """
    LangGraph workflow for complaint insights queries.
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("route", self._route_node)
        workflow.add_node("sql", self._sql_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("hybrid", self._hybrid_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("route")

        # Conditional routing based on query classification
        workflow.add_conditional_edges(
            "route",
            lambda state: state["route"],
            {
                "sql": "sql",
                "rag": "rag",
                "hybrid": "hybrid"
            }
        )

        # All paths lead to synthesis
        workflow.add_edge("sql", "synthesize")
        workflow.add_edge("rag", "synthesize")
        workflow.add_edge("hybrid", "synthesize")

        # End after synthesis
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _route_node(self, state: GraphState) -> GraphState:
        """Route the query to sql/rag/hybrid."""
        logger.info(f"Routing query: {state['question']}")
        route = route_query(state["question"])
        state["route"] = route
        state["metadata"]["route"] = route
        logger.info(f"Route selected: {route}")
        return state

    def _sql_node(self, state: GraphState) -> GraphState:
        """Execute SQL query on BigQuery."""
        logger.info("Executing SQL path")
        sql_result = execute_sql_query(state["question"])
        state["sql_result"] = sql_result
        return state

    def _rag_node(self, state: GraphState) -> GraphState:
        """Execute RAG query on Qdrant."""
        logger.info("Executing RAG path")
        rag_result = execute_rag_query(state["question"])
        state["rag_result"] = rag_result
        return state

    def _hybrid_node(self, state: GraphState) -> GraphState:
        """Execute hybrid SQL + RAG query."""
        logger.info("Executing hybrid path")

        # First, get SQL results for filtering
        sql_result = execute_sql_query(state["question"])
        state["sql_result"] = sql_result

        # Then, use SQL results to filter RAG search
        rag_result = execute_rag_query(
            state["question"],
            filters=sql_result.get("filters", {})
        )
        state["rag_result"] = rag_result

        return state

    def _synthesize_node(self, state: GraphState) -> GraphState:
        """Synthesize final answer using LLM."""
        logger.info("Synthesizing answer")

        answer = synthesize_answer(
            question=state["question"],
            sql_result=state.get("sql_result"),
            rag_result=state.get("rag_result"),
            route=state["route"]
        )

        state["answer"] = answer
        return state

    async def run(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the full graph workflow.

        Args:
            question: Natural language query
            context: Optional context for the query

        Returns:
            Dict with answer, route, and metadata
        """
        initial_state = GraphState(
            question=question,
            route="sql",  # Will be set by router
            sql_result=None,
            rag_result=None,
            answer="",
            metadata=context or {}
        )

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        return {
            "answer": final_state["answer"],
            "route": final_state["route"],
            "metadata": final_state["metadata"]
        }


# Singleton instance
_graph_instance = None


def get_graph() -> ComplaintInsightsGraph:
    """Get or create the graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ComplaintInsightsGraph()
    return _graph_instance
