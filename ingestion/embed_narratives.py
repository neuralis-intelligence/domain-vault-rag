"""
Embed each line of business's free text and load it into Qdrant.

- Reads the LOB's `source` Parquet file (with its `where` filter) from config/lobs.toml
- Embeds the LOB's `rag.text_column` with sentence-transformers
- Upserts vectors + payload into the LOB's collection in batches. Point ids are stable
  (the record id), so re-runs don't duplicate, and records already in the collection are
  skipped unless --force is given
- Creates keyword payload indexes for the LOB's `rag.filter_fields`

Usage (from the project root, with Qdrant running):
    python -m ingestion.embed_narratives                          # every LOB with a [rag] block
    python -m ingestion.embed_narratives --lob student_loan       # one LOB
    python -m ingestion.embed_narratives --lob mortgage --limit 2000   # quick test sample
"""

import argparse
import datetime
import logging
import math
import uuid
from typing import Any

import duckdb
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import settings
from app.lob import LineOfBusiness, RagConfig, get_registry

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy/date values to plain JSON types (missing -> None)."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (datetime.date, datetime.datetime, pd.Timestamp)):
        return str(value.date() if isinstance(value, pd.Timestamp) else value)
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def point_id(collection: str, record_id: Any) -> int | str:
    """Integer ids are used as-is; anything else becomes a stable UUID."""
    text = str(record_id)
    return (
        int(text) if text.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}/{text}"))
    )


def build_point(row: pd.Series, vector: list[float], rag: RagConfig) -> PointStruct:
    """Build a Qdrant point for one record."""
    payload = {field: _json_safe(row.get(field)) for field in rag.payload_fields}
    payload[rag.id_column] = str(row[rag.id_column])
    payload[rag.text_key] = row[rag.text_column]
    return PointStruct(
        id=point_id(rag.collection, row[rag.id_column]), vector=vector, payload=payload
    )


def load_texts(lob: LineOfBusiness, limit: int | None = None) -> pd.DataFrame:
    """Rows of the LOB's source that have non-empty text to embed."""
    rag = lob.rag
    where = f"({lob.where}) AND " if lob.where else ""
    with duckdb.connect() as con:
        available = {
            row[0]
            for row in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(lob.source)]
            ).fetchall()
        }
        wanted = list(dict.fromkeys([rag.id_column, rag.text_column, *rag.payload_fields]))
        missing = [c for c in wanted if c not in available]
        if rag.id_column in missing or rag.text_column in missing:
            raise SystemExit(f"{lob.id}: {lob.source} has no column(s) {missing}")
        if missing:
            logger.warning(
                "%s: payload field(s) %s not in the data, stored as null", lob.id, missing
            )

        columns = ", ".join(f'"{c}"' for c in wanted if c in available)
        query = (
            f"SELECT {columns} FROM read_parquet(?) "
            f"WHERE {where}{rag.text_column} IS NOT NULL AND trim({rag.text_column}) <> ''"
        )
        df = con.execute(query, [str(lob.source)]).df()
    for column in missing:
        df[column] = None
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return df


class TextEmbedder:
    """Embeds text and writes it to Qdrant collections."""

    def __init__(
        self,
        embedding_model: str = settings.embedding_model,
        qdrant_host: str = settings.qdrant_host,
        qdrant_port: int = settings.qdrant_port,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_collection(self, rag: RagConfig, recreate: bool = False) -> None:
        """Create the collection if needed, and make sure its filter fields are indexed."""
        exists = self.qdrant_client.collection_exists(rag.collection)
        if exists and recreate:
            self.qdrant_client.delete_collection(rag.collection)
            exists = False

        if not exists:
            self.qdrant_client.create_collection(
                collection_name=rag.collection,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )
            logger.info("Created collection %s", rag.collection)

        indexed = self.qdrant_client.get_collection(rag.collection).payload_schema or {}
        for field in rag.filter_fields:
            if field not in indexed:
                self.qdrant_client.create_payload_index(
                    rag.collection, field_name=field, field_schema=PayloadSchemaType.KEYWORD
                )

    def drop_existing(
        self, df: pd.DataFrame, rag: RagConfig, batch_size: int = 1000
    ) -> pd.DataFrame:
        """Remove rows whose point is already in the collection."""
        ids = [point_id(rag.collection, v) for v in df[rag.id_column]]
        existing: set = set()
        for start in range(0, len(ids), batch_size):
            points = self.qdrant_client.retrieve(
                rag.collection, ids=ids[start : start + batch_size], with_payload=False
            )
            existing.update(p.id for p in points)
        keep = [pid not in existing for pid in ids]
        return df[keep]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (normalized for cosine similarity)."""
        embeddings = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_and_upload(self, df: pd.DataFrame, rag: RagConfig, batch_size: int = 256) -> int:
        """Embed the texts in `df` and upsert them batch by batch. Returns points written."""
        for start in tqdm(range(0, len(df), batch_size), desc=f"Embedding -> {rag.collection}"):
            batch = df.iloc[start : start + batch_size]
            vectors = self.embed_batch(batch[rag.text_column].tolist())
            points = [
                build_point(row, vector, rag)
                for (_, row), vector in zip(batch.iterrows(), vectors, strict=True)
            ]
            self.qdrant_client.upsert(collection_name=rag.collection, points=points)
        return len(df)


def main() -> None:
    registry = get_registry()
    rag_lobs = [lob.id for lob in registry.lobs.values() if lob.rag]

    parser = argparse.ArgumentParser(description="Embed LOB text into Qdrant")
    parser.add_argument(
        "--lob", nargs="+", choices=rag_lobs, default=rag_lobs, help="LOBs to embed (default: all)"
    )
    parser.add_argument("--limit", type=int, help="Only embed a random sample of N rows per LOB")
    parser.add_argument("--force", action="store_true", help="Re-embed rows already in Qdrant")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collections")
    args = parser.parse_args()

    lobs = [registry.lobs[lob_id] for lob_id in args.lob]
    if args.recreate:
        # Dropping a shared collection would delete other LOBs' points too.
        for lob in lobs:
            sharing = [
                other.id
                for other in registry.lobs.values()
                if other.rag
                and other.rag.collection == lob.rag.collection
                and other.id not in args.lob
            ]
            if sharing:
                raise SystemExit(
                    f"--recreate would drop '{lob.rag.collection}', which also holds "
                    f"{', '.join(sharing)}. Add them to --lob or leave out --recreate."
                )

    embedder = TextEmbedder()
    created: set[str] = set()
    for lob in lobs:
        if not lob.source.exists() and "*" not in str(lob.source):
            logger.warning("%s: skipped, %s not found", lob.id, lob.source)
            continue

        recreate = args.recreate and lob.rag.collection not in created
        embedder.create_collection(lob.rag, recreate=recreate)
        created.add(lob.rag.collection)

        df = load_texts(lob, args.limit)
        total = len(df)
        if not args.force:
            df = embedder.drop_existing(df, lob.rag)
        logger.info(
            "%s: %d texts, %d already embedded, %d to embed",
            lob.id,
            total,
            total - len(df),
            len(df),
        )
        if len(df):
            embedder.embed_and_upload(df, lob.rag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
