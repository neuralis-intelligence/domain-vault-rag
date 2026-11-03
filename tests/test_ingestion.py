"""Tests for the cleaning pipeline and Qdrant point building."""

import datetime

import duckdb
import pandas as pd

from ingestion.clean import FINAL_COLUMNS, clean_complaints
from ingestion.embed_narratives import build_point, load_texts, point_id
from ingestion.load_duckdb import COMPANY_ALIASES
from ingestion.pull_cfpb import loses_narratives


def raw_row(**overrides):
    row = {
        "Date received": "2024-03-05T14:00:00.000Z",
        "Product": "Credit card",
        "Sub-product": "Store credit card",
        "Issue": "Fees or interest",
        "Sub-issue": None,
        "Consumer complaint narrative": "  I was charged a late fee.  ",
        "Company public response": None,
        "Company": "CITIBANK, N.A.",
        "State": "NY",
        "ZIP code": "10001-1234",
        "Tags": None,
        "Submitted via": "Web",
        "Date sent to company": "2024-03-06T00:00:00.000Z",
        "Company response to consumer": "Closed with explanation",
        "Timely response?": "Yes",
        "Complaint ID": "111",
    }
    row.update(overrides)
    return row


def test_clean_complaints_schema_and_values():
    df = clean_complaints(pd.DataFrame([raw_row()]))

    assert list(df.columns) == FINAL_COLUMNS
    row = df.iloc[0]
    assert row["complaint_id"] == "111"
    assert row["date_received"] == datetime.date(2024, 3, 5)
    assert row["consumer_complaint_narrative"] == "I was charged a late fee."
    assert row["zip_code"] == "10001"


def test_clean_complaints_nulls_dedup_and_zip_masks():
    df = clean_complaints(
        pd.DataFrame(
            [
                raw_row(**{"Complaint ID": "1", "ZIP code": "123XX", "Issue": "   "}),
                raw_row(**{"Complaint ID": "1"}),  # duplicate id
                raw_row(**{"Complaint ID": "2", "Date received": "not a date"}),  # bad date
                raw_row(**{"Complaint ID": "3", "ZIP code": "123"}),
            ]
        )
    )

    assert df["complaint_id"].tolist() == ["1", "3"]
    assert pd.isna(df.iloc[0]["zip_code"])
    assert pd.isna(df.iloc[0]["issue"])
    assert pd.isna(df.iloc[1]["zip_code"])


def test_build_point_payload_is_json_safe(registry):
    rag = registry.get("cards").rag
    row = clean_complaints(pd.DataFrame([raw_row(**{"Sub-issue": None})])).iloc[0]
    point = build_point(row, [0.1, 0.2], rag)

    assert point.id == 111
    assert point.payload["complaint_id"] == "111"
    assert point.payload["sub_issue"] is None
    assert point.payload["date_received"] == "2024-03-05"
    assert point.payload["narrative"] == "I was charged a late fee."
    assert set(point.payload) == {*rag.payload_fields, "complaint_id", "narrative"}


def test_point_id_is_stable():
    assert point_id("c", "123") == 123
    assert point_id("c", "tx-1") == point_id("c", "tx-1") != point_id("other", "tx-1")


def test_load_texts_applies_lob_filter_and_skips_empty(registry):
    df = load_texts(registry.get("cards"))
    assert sorted(df["complaint_id"]) == ["1", "3"]  # 2 has no narrative, 4 is a mortgage


def test_build_database_one_table_per_lob(registry, built_db):
    assert not built_db.with_name(built_db.name + ".tmp").exists()

    with duckdb.connect(str(built_db), read_only=True) as con:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        assert tables == {
            "card_complaints",
            "mortgage_complaints",
            "transactions",
            "company_aliases",
        }
        assert con.execute("SELECT COUNT(*) FROM card_complaints").fetchone() == (3,)
        assert con.execute("SELECT DISTINCT product FROM mortgage_complaints").fetchall() == [
            ("Mortgage",)
        ]
        assert con.execute("SELECT MIN(date_received) FROM card_complaints").fetchone() == (
            datetime.date(2024, 1, 5),
        )
        (aliases,) = con.execute("SELECT COUNT(*) FROM company_aliases").fetchone()
        assert aliases == len(COMPANY_ALIASES)


def test_pull_never_overwrites_narratives(tmp_path):
    old = tmp_path / "credit_card.csv"
    pd.DataFrame({"Complaint ID": ["1"], "Consumer complaint narrative": ["text"]}).to_csv(old)
    without = pd.DataFrame({"Complaint ID": ["1"]})
    with_text = pd.DataFrame({"Complaint ID": ["1"], "Consumer complaint narrative": ["x"]})

    assert loses_narratives(old, without) is True
    assert loses_narratives(old, with_text) is False
    assert loses_narratives(tmp_path / "new.csv", without) is False
