"""Tests for the evaluation scoring helpers."""

import datetime

import pytest

from eval.scoring import (
    cites_sources,
    expected_values,
    fraction_mentioned,
    keyword_rate,
    result_matches,
    value_mentioned,
)


class TestValueMentioned:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            ("CAPITAL ONE FINANCIAL CORPORATION", "Capital One leads"),
            ("BANK OF AMERICA, NATIONAL ASSOCIATION", "bank of america had the most"),
            ("JPMORGAN CHASE & CO.", "JPMorgan Chase received"),
            ("CITIBANK, N.A.", "Citibank is second"),
            ("Problem with a purchase shown on your statement", "purchases shown on statements"),
            (27768, "with 27,768 complaints"),
            (97.38, "97.4% were timely"),
            (0.9738, "97.4% were timely"),
            (100.0, "99.9% of responses"),
            ("CA", "California (CA) is first"),
            (datetime.date(2024, 1, 5), "since 2024-01-05"),
        ],
    )
    def test_matches(self, value, text):
        assert value_mentioned(value, text)

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (27768, "about 27,000 complaints"),
            (1956, "19,560 complaints"),
            ("CA", "Oscar"),
            ("Problem with a purchase shown on your statement", "problems with fees"),
            ("Getting a credit card", "getting approved"),
            (None, "anything"),
        ],
    )
    def test_non_matches(self, value, text):
        assert not value_mentioned(value, text)


def test_expected_values():
    rows = [{"company": "A", "n": 3}, {"company": "B", "n": 2}]
    assert expected_values(rows) == ["A", 3]
    assert expected_values(rows, "first_column") == ["A", "B"]
    assert expected_values([]) == []


def test_fraction_mentioned():
    assert fraction_mentioned(["Capital One", "Citibank"], "Capital One is first") == 0.5
    assert fraction_mentioned([], "x") is None


def test_result_matches_ignores_column_names_and_order():
    expected = ["CAPITAL ONE FINANCIAL CORPORATION", 27768]
    rows = [{"n": 27768, "bank": "CAPITAL ONE FINANCIAL CORPORATION"}]
    assert result_matches(expected, rows) is True
    assert result_matches(expected, [{"bank": "CITIBANK, N.A.", "n": 24488}]) is False
    assert result_matches([], rows) is None


def test_cites_sources():
    docs = [{"id": "20068111"}, {"id": "123"}]
    assert cites_sources("As complaint 20068111 says...", docs) is True
    assert cites_sources("Customers are unhappy.", docs) is False
    assert cites_sources("x", []) is None


def test_keyword_rate():
    assert keyword_rate("Fraud at Chase", ["fraud", "customers"]) == 0.5
    assert keyword_rate("x", []) is None


def test_score_row_and_rescore_agree(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from eval import run_eval

    q = {
        "question": "Top company?",
        "lob": "credit_card",
        "expected_route": "sql",
        "expected_answer_contains": ["company"],
    }
    raw = {
        "question_id": 0,
        "model": "m",
        "actual_route": "sql",
        "answer": "Capital One has the most complaints (27,768).",
        "sql_rows": json.dumps([{"company": "CAPITAL ONE FINANCIAL CORPORATION", "n": 27768}]),
        "source_ids": "[]",
    }
    expected = ["CAPITAL ONE FINANCIAL CORPORATION", 27768]
    scored = run_eval.score_row(dict(raw), q, expected)
    assert scored["routing_correct"] is True
    assert scored["sql_result_correct"] is True
    assert scored["answer_correct"] == 1.0
    assert scored["cites_sources"] is None

    path = tmp_path / "run.csv"
    pd.DataFrame([raw]).to_csv(path, index=False)
    questions = tmp_path / "q.jsonl"
    questions.write_text(json.dumps(q) + "\n")
    monkeypatch.setattr(run_eval, "ground_truth", lambda qs, db: {0: expected})
    rescored = run_eval.rescore(path, questions).iloc[0]
    assert rescored["answer_correct"] == 1.0
    assert bool(rescored["sql_result_correct"]) is True
