"""
RAG tool for semantic search on complaint narratives.

Implements:
- Embedding generation for queries
- Qdrant semantic search with metadata filters
- Result formatting with citations
"""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# Payload fields that can be used as exact-match filters (indexed by embed_narratives.py).
FILTERABLE_FIELDS = ("company", "product", "issue", "state")


class RAGTool:
    """Semantic search tool for complaint narratives."""

    def __init__(
        self,
        embedding_model: str | None = None,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        collection_name: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self._embedding_model: SentenceTransformer | None = None

        self.qdrant_client = QdrantClient(
            host=qdrant_host or settings.qdrant_host,
            port=qdrant_port or settings.qdrant_port,
        )
        self.collection_name = collection_name or settings.qdrant_collection
        self.top_k = top_k or settings.rag_top_k
        self.score_threshold = (
            settings.rag_score_threshold if score_threshold is None else score_threshold
        )

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Sentence-transformers model, loaded on first use (download is ~130 MB)."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized embedding for a search query."""
        embedding = self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.tolist()

    @staticmethod
    def build_filters(filters: dict[str, Any] | None = None) -> Filter | None:
        """
        Build a Qdrant filter from metadata, e.g.
        {"company": "JPMORGAN CHASE & CO.", "product": "Credit card", "state": "CA"}.

        Unknown fields and empty values are ignored.
        """
        if not filters:
            return None

        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
            if field in FILTERABLE_FIELDS and value
        ]
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Semantic search over complaint narratives, optionally filtered by metadata."""
        try:
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=self.embed_query(query),
                query_filter=self.build_filters(filters),
                limit=top_k or self.top_k,
                score_threshold=self.score_threshold
                if score_threshold is None
                else score_threshold,
                with_payload=True,
            )

            documents = [
                {
                    "complaint_id": point.payload.get("complaint_id"),
                    "narrative": point.payload.get("narrative"),
                    "company": point.payload.get("company"),
                    "product": point.payload.get("product"),
                    "issue": point.payload.get("issue"),
                    "state": point.payload.get("state"),
                    "date_received": point.payload.get("date_received"),
                    "score": point.score,
                }
                for point in response.points
            ]
            return {
                "query": query,
                "filters": filters or {},
                "documents": documents,
                "count": len(documents),
            }

        except Exception as e:  # qdrant/network errors should not crash the graph
            logger.error("RAG search failed: %s", e)
            return {"error": str(e), "query": query, "documents": [], "count": 0}

    def is_available(self) -> bool:
        """Return True if Qdrant is reachable and the collection exists."""
        try:
            return self.qdrant_client.collection_exists(self.collection_name)
        except Exception:
            return False


_rag_tool: RAGTool | None = None


def get_rag_tool() -> RAGTool:
    """Get or create the shared RAG tool."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGTool()
    return _rag_tool


def execute_rag_query(question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run semantic search for `question` with optional metadata filters."""
    results = get_rag_tool().search(question, filters=filters)
    logger.info("RAG search returned %d results", results.get("count", 0))
    return results
