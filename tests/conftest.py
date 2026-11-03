"""Shared fixtures."""

import textwrap

import pandas as pd
import pytest

from app.lob import load_registry


class FakeLLM:
    """Stands in for OllamaClient; returns queued responses and records prompts."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.models: list[str | None] = []  # model requested for each get_llm_client() call

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0) if self.responses else ""

    def is_available(self) -> bool:
        return True


@pytest.fixture
def fake_llm(monkeypatch):
    """Patch every module that calls get_llm_client() to use a FakeLLM."""
    llm = FakeLLM()

    def get_client(model=None):
        llm.models.append(model)
        return llm

    for module in ("app.llm", "app.router", "app.sql_tool"):
        monkeypatch.setattr(f"{module}.get_llm_client", get_client)
    return llm


REGISTRY_TOML = """
default_lob = "cards"

[lobs.cards]
name = "Credit cards"
description = "Card complaints"
profile = "cfpb_complaints"
source = "complaints.parquet"
where = "product = 'Credit card'"
table = "card_complaints"
rag.filter = { product = "Credit card" }
example_questions = ["Top issue?"]

[lobs.mortgage]
name = "Mortgages"
profile = "cfpb_complaints"
source = "complaints.parquet"
where = "product = 'Mortgage'"
table = "mortgage_complaints"
rag.filter = { product = "Mortgage" }

[lobs.transactions]
name = "Transactions"
source = "transactions.parquet"
table = "transactions"
value_columns = ["merchant"]

[lobs.transactions.columns]
amount = "Amount in USD"

[lobs.not_pulled]
name = "Not pulled yet"
source = "missing.parquet"
table = "not_pulled"
"""


@pytest.fixture
def data_dir(tmp_path):
    """A folder with small CFPB-style and transaction Parquet files plus a lobs.toml."""
    pd.DataFrame(
        {
            "complaint_id": ["1", "2", "3", "4"],
            "date_received": pd.to_datetime(
                ["2024-01-05", "2024-02-10", "2024-02-11", "2025-03-01"]
            ).date,
            "product": ["Credit card", "Credit card", "Credit card", "Mortgage"],
            "issue": ["Fees or interest", "Fees or interest", "Closing your account", "Escrow"],
            "company": ["CITIBANK, N.A.", "CITIBANK, N.A.", "DISCOVER BANK", "ROCKET"],
            "state": ["NY", "CA", "NY", "TX"],
            "timely_response": ["Yes", "No", "Yes", "Yes"],
            "consumer_complaint_narrative": ["late fee", None, "closed my card", "escrow wrong"],
        }
    ).to_parquet(tmp_path / "complaints.parquet", index=False)
    pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "merchant": ["ACME", "ACME", "BOLT"],
            "amount": [10.5, 20.0, 7.25],
        }
    ).to_parquet(tmp_path / "transactions.parquet", index=False)
    (tmp_path / "lobs.toml").write_text(textwrap.dedent(REGISTRY_TOML))
    return tmp_path


@pytest.fixture
def registry(data_dir):
    return load_registry(data_dir / "lobs.toml", root=data_dir)


@pytest.fixture
def built_db(registry, data_dir):
    """Path to a DuckDB file built from the test registry."""
    from ingestion.load_duckdb import build_database

    db_path = data_dir / "test.duckdb"
    build_database(registry, db_path)
    return db_path
