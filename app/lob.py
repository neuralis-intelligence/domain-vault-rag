"""
Line-of-business (LOB) registry, loaded from config/lobs.toml.

A line of business is one dataset the user can ask about: a DuckDB table built from a
Parquet file (SQL path) and, optionally, a Qdrant collection of embedded text (RAG path).
The user picks the LOB per question; a classifier agent can pick it later.

Entries can name a `profile`, a set of defaults shared by similar datasets (e.g. every
CFPB complaint product). Values in the LOB entry override the profile's.
"""

import copy
import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, settings

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Shared defaults, in the same shape as a [lobs.<id>] entry in lobs.toml.
PROFILES: dict[str, dict[str, Any]] = {
    "cfpb_complaints": {
        "columns": {
            "complaint_id": "Unique complaint identifier",
            "date_received": "Date the CFPB received the complaint",
            "date_sent_to_company": "Date forwarded to the company",
            "product": "CFPB product category",
            "sub_product": "Product subtype",
            "issue": "Main issue category",
            "sub_issue": "Issue detail",
            "consumer_complaint_narrative": "Customer's written complaint (often NULL)",
            "company": "Company legal name, UPPER or mixed case",
            "state": "U.S. state abbreviation",
            "zip_code": "5-digit ZIP code",
            "submitted_via": "Submission channel",
            "company_response": "Company's response to the consumer",
            "timely_response": "Whether the company responded on time",
            "company_public_response": "Company's optional public statement",
            "tags": "e.g. 'Older American', 'Servicemember'",
        },
        # High-cardinality columns whose most common values are still worth showing.
        "value_columns": ["company"],
        "prompt_notes": (
            "Company names are legal names (e.g. 'JPMORGAN CHASE & CO.'). When the user gives "
            "a short or informal name, match with UPPER(company) LIKE '%KEYWORD%' instead of "
            "exact equality."
        ),
        "sql_examples": [
            {
                "question": "What is the most common issue at Bank of America?",
                "sql": (
                    "SELECT issue, COUNT(*) AS complaint_count\nFROM {table}\n"
                    "WHERE UPPER(company) LIKE '%BANK OF AMERICA%'\nGROUP BY issue\n"
                    "ORDER BY complaint_count DESC\nLIMIT 1"
                ),
            },
            {
                "question": "How many complaints were received in 2024?",
                "sql": (
                    "SELECT COUNT(*) AS total_complaints\nFROM {table}\n"
                    "WHERE EXTRACT(YEAR FROM date_received) = 2024"
                ),
            },
            {
                "question": "Top 5 companies by complaint volume",
                "sql": (
                    "SELECT company, COUNT(*) AS complaint_count\nFROM {table}\n"
                    "GROUP BY company\nORDER BY complaint_count DESC\nLIMIT 5"
                ),
            },
            {
                "question": (
                    "Among the top 5 companies by complaint volume, which has the most "
                    "untimely responses?"
                ),
                "sql": (
                    "WITH top_companies AS (\n  SELECT company\n  FROM {table}\n"
                    "  GROUP BY company\n  ORDER BY COUNT(*) DESC\n  LIMIT 5\n)\n"
                    "SELECT company, COUNT(*) AS untimely_responses\nFROM {table}\n"
                    "WHERE timely_response = 'No' "
                    "AND company IN (SELECT company FROM top_companies)\n"
                    "GROUP BY company\nORDER BY untimely_responses DESC\nLIMIT 1"
                ),
            },
        ],
        "rag": {
            "collection": "complaint_narratives",
            "text_column": "consumer_complaint_narrative",
            "id_column": "complaint_id",
            "text_key": "narrative",
            "doc_label": "Complaint",
            "payload_fields": [
                "company",
                "product",
                "issue",
                "sub_issue",
                "state",
                "date_received",
            ],
            "filter_fields": ["company", "product", "issue", "state"],
        },
    },
}


class UnknownLOBError(KeyError):
    """Raised when a request names a line of business that isn't in the registry."""


@dataclass(frozen=True)
class RagConfig:
    """Where a LOB's embedded text lives in Qdrant and how its payload is laid out."""

    collection: str
    text_column: str  # column in the source table that gets embedded
    id_column: str  # unique record id; stored in the payload under the same name
    text_key: str = "text"  # payload key holding the embedded text
    doc_label: str = "Record"  # how citations are labelled, e.g. "[Complaint 123]"
    payload_fields: tuple[str, ...] = ()  # extra columns stored and shown with each hit
    filter_fields: tuple[str, ...] = ()  # payload fields indexed for exact-match filters
    fixed_filter: dict[str, str] = field(default_factory=dict)  # always applied


@dataclass(frozen=True)
class LineOfBusiness:
    """One queryable dataset."""

    id: str
    name: str
    description: str
    source: Path  # Parquet file (or glob) the table is built from
    table: str  # DuckDB table name
    where: str | None = None  # optional row filter applied when building the table
    columns: dict[str, str] = field(default_factory=dict)  # column -> description for prompts
    value_columns: tuple[str, ...] = ()
    prompt_notes: str = ""
    sql_examples: tuple[tuple[str, str], ...] = ()  # (question, sql with {table} placeholder)
    example_questions: tuple[str, ...] = ()  # shown in the UI
    rag: RagConfig | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _identifier(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.match(value):
        raise ValueError(f"{what} must be a plain identifier (letters, digits, _), got {value!r}")
    return value


def _parse_rag(lob_id: str, raw: dict[str, Any] | None) -> RagConfig | None:
    if not raw:
        return None
    missing = [key for key in ("collection", "text_column", "id_column") if not raw.get(key)]
    if missing:
        raise ValueError(f"LOB {lob_id!r}: [rag] is missing {', '.join(missing)}")
    return RagConfig(
        collection=raw["collection"],
        text_column=_identifier(raw["text_column"], f"LOB {lob_id!r} rag.text_column"),
        id_column=_identifier(raw["id_column"], f"LOB {lob_id!r} rag.id_column"),
        text_key=raw.get("text_key", "text"),
        doc_label=raw.get("doc_label", "Record"),
        payload_fields=tuple(raw.get("payload_fields", ())),
        filter_fields=tuple(raw.get("filter_fields", ())),
        fixed_filter={str(k): str(v) for k, v in raw.get("filter", {}).items()},
    )


def _parse_lob(lob_id: str, raw: dict[str, Any], root: Path) -> LineOfBusiness:
    profile_name = raw.get("profile")
    if profile_name:
        if profile_name not in PROFILES:
            raise ValueError(f"LOB {lob_id!r}: unknown profile {profile_name!r}")
        raw = _deep_merge(PROFILES[profile_name], raw)

    for key in ("name", "source", "table"):
        if not raw.get(key):
            raise ValueError(f"LOB {lob_id!r} is missing {key!r}")

    return LineOfBusiness(
        id=_identifier(lob_id, "LOB id"),
        name=raw["name"],
        description=raw.get("description", ""),
        source=root / raw["source"],
        table=_identifier(raw["table"], f"LOB {lob_id!r} table"),
        where=raw.get("where"),
        columns=dict(raw.get("columns", {})),
        value_columns=tuple(raw.get("value_columns", ())),
        prompt_notes=raw.get("prompt_notes", ""),
        sql_examples=tuple((ex["question"], ex["sql"]) for ex in raw.get("sql_examples", ())),
        example_questions=tuple(raw.get("example_questions", ())),
        rag=_parse_rag(lob_id, raw.get("rag")),  # `rag = false` turns off a profile's RAG
    )


@dataclass(frozen=True)
class Registry:
    lobs: dict[str, LineOfBusiness]
    default: str

    def get(self, lob_id: str | None = None) -> LineOfBusiness:
        """Return the LOB with `lob_id`, or the default LOB when it's None/empty."""
        key = lob_id or self.default
        try:
            return self.lobs[key]
        except KeyError:
            raise UnknownLOBError(
                f"Unknown line of business {key!r}. Available: {', '.join(self.lobs)}"
            ) from None


def load_registry(path: Path, root: Path = PROJECT_ROOT) -> Registry:
    """Parse and validate a lobs.toml file. Relative `source` paths resolve against `root`."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    entries = raw.get("lobs", {})
    if not entries:
        raise ValueError(f"{path} defines no [lobs.<id>] entries")
    lobs = {lob_id: _parse_lob(lob_id, entry, root) for lob_id, entry in entries.items()}

    tables = [lob.table for lob in lobs.values()]
    if len(tables) != len(set(tables)):
        raise ValueError(f"{path}: two LOBs use the same table name")

    default = raw.get("default_lob") or next(iter(lobs))
    if default not in lobs:
        raise ValueError(f"{path}: default_lob {default!r} is not defined")
    return Registry(lobs=lobs, default=default)


@lru_cache(maxsize=1)
def get_registry() -> Registry:
    """The registry from LOB_REGISTRY_PATH, loaded once per process."""
    return load_registry(settings.lob_registry_path)


def get_lob(lob_id: str | None = None) -> LineOfBusiness:
    """Shortcut for get_registry().get(lob_id)."""
    return get_registry().get(lob_id)
