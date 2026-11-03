"""
RAG tool for semantic search over a line of business's embedded text.

Implements:
- Embedding generation for queries
- Qdrant semantic search with metadata filters (each LOB has its own collection
  and filter fields, see config/lobs.toml)
- Result formatting with citations
"""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.lob import LineOfBusiness, RagConfig, get_lob

logger = logging.getLogger(__name__)


class RAGTool:
    """Semantic search tool; one instance serves every LOB."""

    def __init__(
        self,
        embedding_model: str | None = None,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self._embedding_model: SentenceTransformer | None = None

        self.qdrant_client = QdrantClient(
            host=qdrant_host or settings.qdrant_host,
            port=qdrant_port or settings.qdrant_port,
        )
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
    def build_filters(rag: RagConfig, filters: dict[str, Any] | None = None) -> Filter | None:
        """
        Build a Qdrant filter from the LOB's fixed filter plus request filters, e.g.
        {"company": "JPMORGAN CHASE & CO.", "state": "CA"}.

        Fields not in the LOB's `filter_fields` and empty values are ignored. The fixed
        filter (e.g. product = 'Mortgage' in a shared collection) always applies.
        """
        merged = {
            field: value
            for field, value in (filters or {}).items()
            if field in rag.filter_fields and value
        }
        merged.update(rag.fixed_filter)
        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in merged.items()
        ]
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query: str,
        rag: RagConfig,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Semantic search over the LOB's collection, optionally filtered by metadata."""
        try:
            response = self.qdrant_client.query_points(
                collection_name=rag.collection,
                query=self.embed_query(query),
                query_filter=self.build_filters(rag, filters),
                limit=top_k or self.top_k,
                score_threshold=self.score_threshold
                if score_threshold is None
                else score_threshold,
                with_payload=True,
            )

            documents = [
                {
                    "id": point.payload.get(rag.id_column),
                    "text": point.payload.get(rag.text_key),
                    "fields": {f: point.payload.get(f) for f in rag.payload_fields},
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
        """Return True if Qdrant is reachable."""
        try:
            self.qdrant_client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self, collection: str) -> bool:
        """Return True if Qdrant is reachable and `collection` exists."""
        try:
            return self.qdrant_client.collection_exists(collection)
        except Exception:
            return False


_rag_tool: RAGTool | None = None


def get_rag_tool() -> RAGTool:
    """Get or create the shared RAG tool."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGTool()
    return _rag_tool


def execute_rag_query(
    question: str,
    lob: LineOfBusiness | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run semantic search for `question` in the LOB's collection, with optional filters."""
    lob = lob or get_lob()
    if lob.rag is None:
        return {"error": f"'{lob.name}' has no text search configured", "documents": [], "count": 0}
    results = get_rag_tool().search(question, lob.rag, filters=filters)
    logger.info("RAG search returned %d results", results.get("count", 0))
    return results
