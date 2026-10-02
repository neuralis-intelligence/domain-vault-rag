"""
FastAPI backend for Complaint Insights Agent.

Provides:
- POST /ask endpoint for natural language queries
- Query routing between SQL, RAG, and hybrid approaches
- Response synthesis using local LLM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from app.graph import ComplaintInsightsGraph
from app.llm import get_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Complaint Insights API",
    description="Answer natural language questions about CFPB bank complaints",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str  # "sql", "rag", or "hybrid"
    metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Complaint Insights API...")
    # TODO: Initialize LangGraph, LLM, BigQuery, and Qdrant clients
    # global graph
    # graph = ComplaintInsightsGraph()
    logger.info("API ready to serve requests")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Complaint Insights API",
        "status": "operational",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    # TODO: Check connectivity to Ollama, Qdrant, BigQuery
    return {
        "status": "healthy",
        "ollama": "connected",
        "qdrant": "connected",
        "bigquery": "connected"
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Process a natural language question about CFPB complaints.

    The system will:
    1. Route the question to SQL, RAG, or hybrid approach
    2. Execute the appropriate retrieval strategy
    3. Synthesize an answer using the local LLM
    4. Return the answer with metadata
    """
    try:
        logger.info(f"Received question: {request.question}")

        # TODO: Implement LangGraph workflow
        # result = await graph.run(request.question, context=request.context)

        # Placeholder response
        return QueryResponse(
            question=request.question,
            answer="This endpoint is under development. Coming soon!",
            route="unknown",
            metadata={}
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(question: str, answer: str, rating: int, comments: Optional[str] = None):
    """
    Submit feedback on query responses for continuous improvement.
    """
    # TODO: Log feedback for analysis
    logger.info(f"Feedback received: rating={rating}, question={question[:50]}...")
    return {"status": "feedback recorded", "thank_you": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
