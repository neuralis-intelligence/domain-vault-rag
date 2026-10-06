"""Tests for the cleaning pipeline and Qdrant point building."""

import datetime

import pandas as pd

from ingestion.clean import FINAL_COLUMNS, clean_complaints
from ingestion.embed_narratives import build_point


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


def test_build_point_payload_is_json_safe():
    row = clean_complaints(pd.DataFrame([raw_row(**{"Sub-issue": None})])).iloc[0]
    point = build_point(row, [0.1, 0.2])

    assert point.id == 111
    assert point.payload["complaint_id"] == "111"
    assert point.payload["sub_issue"] is None
    assert point.payload["date_received"] == "2024-03-05"
    assert point.payload["narrative"] == "I was charged a late fee."
