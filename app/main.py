"""
FastAPI backend for the Complaint Insights Agent.

Endpoints:
- GET  /         service info
- GET  /health   connectivity to Ollama, Qdrant and DuckDB
- GET  /models   installed local models the user can choose from
- GET  /lobs     lines of business the user can ask about, and whether their data is loaded
- POST /ask      answer a question about one LOB with one model (routes sql/rag/hybrid)
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
from app.llm import ModelNotAvailableError, get_llm_client, list_models, resolve_model
from app.lob import UnknownLOBError, get_registry
from app.rag_tool import get_rag_tool
from app.sql_tool import get_sql_tool

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
    lob: str | None = Field(
        None, description="Line of business id from GET /lobs (default LOB if omitted)"
    )
    model: str | None = Field(
        None, description="Model name from GET /models (OLLAMA_MODEL if omitted)"
    )
    context: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str  # "sql", "rag", or "hybrid"
    lob: str
    model: str
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
        "duckdb": "connected" if get_sql_tool().is_available() else "unavailable",
    }
    healthy = all(status == "connected" for status in checks.values())
    return {"status": "healthy" if healthy else "degraded", **checks}


@app.get("/models")
async def get_models():
    """Installed local models the user can pick (smallest first) and the default one."""
    try:
        models = list_models()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unreachable at {settings.ollama_host}. Is Ollama running?",
        ) from e
    return {"default": settings.ollama_model, "models": models}


@app.get("/lobs")
async def get_lobs():
    """Lines of business from config/lobs.toml, with how many rows each has loaded."""
    registry = get_registry()
    sql_tool, rag_tool = get_sql_tool(), get_rag_tool()
    lobs = []
    for lob in registry.lobs.values():
        rows = sql_tool.row_count(lob.table)
        lobs.append(
            {
                "id": lob.id,
                "name": lob.name,
                "description": lob.description,
                "rows": rows or 0,
                "sql_available": bool(rows),
                "rag_available": bool(lob.rag) and rag_tool.collection_exists(lob.rag.collection),
                "example_questions": list(lob.example_questions),
            }
        )
    return {"default": registry.default, "lobs": lobs}


@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    graph: ComplaintInsightsGraph = Depends(get_graph),
):
    """Route the question, retrieve data from the chosen LOB, and answer with the chosen model."""
    try:
        lob = get_registry().get(request.lob)
    except UnknownLOBError as e:
        raise HTTPException(status_code=400, detail=str(e.args[0])) from e
    try:
        model = resolve_model(request.model)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM service unreachable at {settings.ollama_host}. Is Ollama running?",
        ) from e

    logger.info("Received question (lob=%s, model=%s): %s", lob.id, model, request.question)
    try:
        result = await graph.run(request.question, lob=lob, model=model, context=request.context)
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
        lob=lob.id,
        model=model,
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
