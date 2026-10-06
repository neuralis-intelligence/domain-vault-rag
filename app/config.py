"""Central configuration: project paths and settings loaded from the environment / .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_COMPLAINTS_PATH = PROCESSED_DIR / "complaints.parquet"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Defaults match .env.example."""

    bigquery_project_id: str | None = os.getenv("BIGQUERY_PROJECT_ID") or None
    bigquery_dataset: str = os.getenv("BIGQUERY_DATASET", "cfpb_complaints")
    bigquery_table: str = os.getenv("BIGQUERY_TABLE", "complaints")

    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION_NAME", "complaint_narratives")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

    sql_max_rows: int = int(os.getenv("SQL_MAX_ROWS", "1000"))
    sql_timeout_seconds: int = int(os.getenv("SQL_TIMEOUT_SECONDS", "30"))


settings = Settings()
