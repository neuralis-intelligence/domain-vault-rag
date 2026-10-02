"""
Narrative embedding generation and Qdrant loading.

This script:
- Loads complaint narratives from cleaned data
- Generates embeddings using sentence-transformers
- Stores embeddings in Qdrant with metadata filters
- Supports batch processing for large datasets
"""

import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NarrativeEmbedder:
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "complaint_narratives"
    ):
        self.model = SentenceTransformer(embedding_model)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists.")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection {self.collection_name}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def prepare_points(
        self,
        df: pd.DataFrame,
        batch_size: int = 100
    ) -> List[PointStruct]:
        """
        Prepare Qdrant points from DataFrame.

        Expected DataFrame columns:
        - complaint_id (unique identifier)
        - consumer_complaint_narrative (text to embed)
        - company, product, issue, state, date_received (metadata for filtering)
        """
        points = []

        # Filter rows with valid narratives
        df_valid = df[df['consumer_complaint_narrative'].notna()].copy()
        logger.info(f"Processing {len(df_valid)} narratives with valid text")

        for i in tqdm(range(0, len(df_valid), batch_size), desc="Embedding batches"):
            batch = df_valid.iloc[i:i+batch_size]

            # Generate embeddings
            embeddings = self.embed_batch(batch['consumer_complaint_narrative'].tolist())

            # Create points with metadata
            for idx, (_, row) in enumerate(batch.iterrows()):
                point = PointStruct(
                    id=hash(row['complaint_id']) % (10**9),  # Convert to numeric ID
                    vector=embeddings[idx],
                    payload={
                        "complaint_id": row['complaint_id'],
                        "company": row.get('company', ''),
                        "product": row.get('product', ''),
                        "issue": row.get('issue', ''),
                        "state": row.get('state', ''),
                        "date_received": str(row.get('date_received', '')),
                        "narrative": row['consumer_complaint_narrative'][:500],  # First 500 chars
                    }
                )
                points.append(point)

        return points

    def upload_to_qdrant(self, points: List[PointStruct], batch_size: int = 100):
        """Upload points to Qdrant in batches."""
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i+batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        logger.info(f"Uploaded {len(points)} points to Qdrant")


def main():
    """Main embedding pipeline."""
    logger.info("Starting narrative embedding pipeline...")

    # Get configuration from environment
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "complaint_narratives")

    # Initialize embedder
    embedder = NarrativeEmbedder(
        embedding_model=embedding_model,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name
    )

    # Create collection
    embedder.create_collection()

    # TODO: Load cleaned data
    # data_path = Path(__file__).parent.parent / "data" / "cleaned_complaints.csv"
    # if data_path.exists():
    #     logger.info(f"Loading data from {data_path}")
    #     df = pd.read_csv(data_path)
    #
    #     # Prepare and upload points
    #     points = embedder.prepare_points(df)
    #     embedder.upload_to_qdrant(points)
    # else:
    #     logger.warning("No cleaned data found. Run clean.py and load_bigquery.py first.")

    logger.info("Embedding pipeline completed.")
    logger.info(f"Collection '{collection_name}' is ready for semantic search.")


if __name__ == "__main__":
    main()
