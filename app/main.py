"""
FastAPI backend for the Complaint Insights Agent.

Endpoints:
- GET  /         service info
- GET  /health   connectivity to Ollama, Qdrant and BigQuery
- POST /ask      answer a natural-language question (routes sql/rag/hybrid)
- POST /feedback record user feedback on an answer
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import requests
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.config import settings
from app.graph import ComplaintInsightsGraph, get_graph
from app.llm import get_llm_client
from app.rag_tool import get_rag_tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting Complaint Insights API v%s", __version__)
    yield


app = FastAPI(
    title="Complaint Insights API",
    description="Answer natural language questions about CFPB bank complaints",
    version=__version__,
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    context: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str  # "sql", "rag", or "hybrid"
    metadata: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5)
    comments: str | None = None


@app.get("/")
async def root():
    """Service info."""
    return {"service": "Complaint Insights API", "status": "operational", "version": __version__}


@app.get("/health")
async def health_check():
    """Report connectivity to each backing service."""
    checks = {
        "ollama": "connected" if get_llm_client().is_available() else "unavailable",
        "qdrant": "connected" if get_rag_tool().is_available() else "unavailable",
        "bigquery": "configured" if settings.bigquery_project_id else "not configured",
    }
    healthy = checks["ollama"] == "connected" and checks["qdrant"] == "connected"
    return {"status": "healthy" if healthy else "degraded", **checks}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    graph: ComplaintInsightsGraph = Depends(get_graph),
):
    """Route the question, retrieve data, and synthesize an answer."""
    logger.info("Received question: %s", request.question)
    try:
        result = await graph.run(request.question, context=request.context)
    except requests.ConnectionError as e:
        logger.error("Ollama unreachable at %s: %s", settings.ollama_host, e)
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unreachable at {settings.ollama_host}. Is Ollama running?",
        ) from e
    except Exception as e:
        logger.exception("Error processing question")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        route=result["route"],
        metadata=result["metadata"],
    )


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Record feedback on a response (logged for now)."""
    logger.info(
        "Feedback received: rating=%d, question=%s", feedback.rating, feedback.question[:50]
    )
    return {"status": "feedback recorded", "thank_you": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
