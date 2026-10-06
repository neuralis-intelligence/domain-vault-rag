"""
Embed complaint narratives and load them into Qdrant.

- Reads data/processed/complaints.parquet (output of clean.py)
- Embeds consumer_complaint_narrative with sentence-transformers
- Upserts vectors + metadata into Qdrant in batches (idempotent: point id = complaint id)
- Creates keyword payload indexes for the metadata filters used by app/rag_tool.py

Usage (from the project root, with Qdrant running):
    python -m ingestion.embed_narratives --limit 2000   # quick local test
    python -m ingestion.embed_narratives                # all narratives
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import CLEANED_COMPLAINTS_PATH, settings
from app.rag_tool import FILTERABLE_FIELDS

logger = logging.getLogger(__name__)

PAYLOAD_FIELDS = ("company", "product", "issue", "sub_issue", "state")


def _clean(value: Any) -> Any:
    """Convert pandas missing values to None so payloads are valid JSON."""
    return None if pd.isna(value) else value


def build_point(row: pd.Series, vector: list[float]) -> PointStruct:
    """Build a Qdrant point for one complaint row."""
    payload = {field: _clean(row.get(field)) for field in PAYLOAD_FIELDS}
    payload["complaint_id"] = str(row["complaint_id"])
    payload["date_received"] = str(row["date_received"])
    payload["narrative"] = row["consumer_complaint_narrative"]
    return PointStruct(id=int(row["complaint_id"]), vector=vector, payload=payload)


class NarrativeEmbedder:
    """Embeds narratives and writes them to a Qdrant collection."""

    def __init__(
        self,
        embedding_model: str = settings.embedding_model,
        qdrant_host: str = settings.qdrant_host,
        qdrant_port: int = settings.qdrant_port,
        collection_name: str = settings.qdrant_collection,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection (and payload indexes) if it doesn't exist."""
        exists = self.qdrant_client.collection_exists(self.collection_name)
        if exists and recreate:
            self.qdrant_client.delete_collection(self.collection_name)
            exists = False

        if exists:
            logger.info("Collection %s already exists", self.collection_name)
            return

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )
        for field in FILTERABLE_FIELDS:
            self.qdrant_client.create_payload_index(
                self.collection_name, field_name=field, field_schema=PayloadSchemaType.KEYWORD
            )
        logger.info("Created collection %s", self.collection_name)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (normalized for cosine similarity)."""
        embeddings = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_and_upload(self, df: pd.DataFrame, batch_size: int = 256) -> int:
        """Embed narratives in `df` and upsert them batch by batch. Returns points written."""
        df = df[df["consumer_complaint_narrative"].notna()]
        logger.info("Embedding %d narratives", len(df))

        for start in tqdm(range(0, len(df), batch_size), desc="Embedding + upserting"):
            batch = df.iloc[start : start + batch_size]
            vectors = self.embed_batch(batch["consumer_complaint_narrative"].tolist())
            points = [
                build_point(row, vector)
                for (_, row), vector in zip(batch.iterrows(), vectors, strict=True)
            ]
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        return len(df)


def main(input_path: Path = CLEANED_COMPLAINTS_PATH) -> None:
    parser = argparse.ArgumentParser(description="Embed complaint narratives into Qdrant")
    parser.add_argument("--limit", type=int, help="Only embed the first N narratives")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection")
    args = parser.parse_args()

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found. Run `python -m ingestion.clean` first.")

    df = pd.read_parquet(input_path)
    df = df[df["consumer_complaint_narrative"].notna()]
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)

    embedder = NarrativeEmbedder()
    embedder.create_collection(recreate=args.recreate)
    count = embedder.embed_and_upload(df)
    logger.info("Collection '%s' now has %d new/updated points", embedder.collection_name, count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
