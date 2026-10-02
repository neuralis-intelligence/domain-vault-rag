"""
RAG tool for semantic search on complaint narratives.

Implements:
- Embedding generation for queries
- Qdrant semantic search with metadata filters
- Result formatting with citations
"""

import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGTool:
    """Semantic search tool for complaint narratives."""

    def __init__(
        self,
        embedding_model: str = None,
        qdrant_host: str = None,
        qdrant_port: int = None,
        collection_name: str = None,
        top_k: int = 5,
        score_threshold: float = 0.7
    ):
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = qdrant_port or int(os.getenv("QDRANT_PORT", 6333))

        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION_NAME", "complaint_narratives"
        )
        self.top_k = top_k
        self.score_threshold = score_threshold

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def build_filters(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
        """
        Build Qdrant filters from metadata.

        Example filters:
        {
            "company": "JPMORGAN CHASE & CO.",
            "product": "Credit card",
            "state": "CA"
        }
        """
        if not filters:
            return None

        conditions = []

        for field, value in filters.items():
            if value:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search on complaint narratives.

        Args:
            query: Natural language search query
            filters: Optional metadata filters (company, product, issue, state)
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            Search results with narratives and metadata
        """
        try:
            # Generate query embedding
            query_vector = self.embed_query(query)

            # Build filters
            qdrant_filter = self.build_filters(filters)

            # Search Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k or self.top_k,
                score_threshold=score_threshold or self.score_threshold,
                query_filter=qdrant_filter
            )

            # Format results
            documents = []
            for result in search_results:
                documents.append({
                    "complaint_id": result.payload.get("complaint_id"),
                    "narrative": result.payload.get("narrative"),
                    "company": result.payload.get("company"),
                    "product": result.payload.get("product"),
                    "issue": result.payload.get("issue"),
                    "state": result.payload.get("state"),
                    "score": result.score
                })

            return {
                "query": query,
                "documents": documents,
                "count": len(documents)
            }

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return {
                "error": str(e),
                "query": query,
                "documents": []
            }


# Singleton instance
_rag_tool = None


def get_rag_tool() -> RAGTool:
    """Get or create RAG tool instance."""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = RAGTool()
    return _rag_tool


def execute_rag_query(
    question: str,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute RAG search pipeline.

    Args:
        question: Natural language question
        filters: Optional metadata filters

    Returns:
        Search results with relevant narratives
    """
    tool = get_rag_tool()
    results = tool.search(question, filters=filters)
    logger.info(f"RAG search returned {results.get('count', 0)} results")
    return results
