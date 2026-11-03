"""Tests for text-to-SQL generation and validation."""

import dataclasses

import duckdb
import pytest

from app.sql_tool import SQLTool, SQLValidationError, execute_sql_query


@pytest.fixture
def sql_tool(tmp_path):
    """Tool pointed at a path with no database, so nothing touches the real data."""
    return SQLTool(db_path=tmp_path / "missing.duckdb", max_rows=100)


@pytest.fixture
def db_tool(built_db):
    """Tool backed by the small database built from the test registry."""
    return SQLTool(db_path=built_db, max_rows=100)


@pytest.fixture
def cards(registry):
    return registry.get("cards")


class TestValidateSQL:
    def test_select_with_limit_is_unchanged(self, sql_tool):
        sql = "SELECT * FROM complaints LIMIT 10"
        assert sql_tool.validate_sql(sql) == sql

    def test_with_cte_allowed(self, sql_tool):
        sql = "WITH t AS (SELECT 1 AS x) SELECT x FROM t LIMIT 1"
        assert sql_tool.validate_sql(sql) == sql

    def test_missing_limit_is_added(self, sql_tool):
        assert sql_tool.validate_sql("SELECT issue FROM complaints").endswith("LIMIT 100")

    def test_large_limit_is_capped(self, sql_tool):
        assert sql_tool.validate_sql("SELECT issue FROM complaints LIMIT 50000").endswith(
            "LIMIT 100"
        )

    def test_stacked_limits_collapsed_to_smallest(self, sql_tool):
        sql = "SELECT company, COUNT(*) AS n FROM complaints GROUP BY company\nLIMIT 5\nLIMIT 1"
        assert sql_tool.validate_sql(sql) == (
            "SELECT company, COUNT(*) AS n FROM complaints GROUP BY company\nLIMIT 1"
        )

    def test_limit_inside_subquery_untouched(self, sql_tool):
        sql = "WITH t AS (SELECT company FROM complaints LIMIT 5) SELECT * FROM t LIMIT 1"
        assert sql_tool.validate_sql(sql) == sql

    def test_trailing_semicolon_stripped(self, sql_tool):
        assert sql_tool.validate_sql("SELECT 1 LIMIT 1;") == "SELECT 1 LIMIT 1"

    def test_column_names_containing_keywords_allowed(self, sql_tool):
        sql = "SELECT created_at, last_updated FROM complaints LIMIT 5"
        assert sql_tool.validate_sql(sql) == sql

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE complaints",
            "DELETE FROM complaints WHERE company = 'Chase'",
            "INSERT INTO complaints VALUES ('test')",
            "UPDATE complaints SET issue = 'x'",
            "SELECT 1; DROP TABLE complaints",
            "SELECT * FROM complaints WHERE 1=1 AND EXISTS (DELETE FROM x)",
        ],
    )
    def test_unsafe_queries_rejected(self, sql_tool, sql):
        with pytest.raises(SQLValidationError):
            sql_tool.validate_sql(sql)


class TestGenerateSQL:
    def test_strips_markdown_fences(self, db_tool, cards, fake_llm):
        fake_llm.responses = ["```sql\nSELECT 1 LIMIT 1\n```"]
        assert db_tool.generate_sql("q", cards) == "SELECT 1 LIMIT 1"

    def test_prompt_is_built_for_the_lob(self, db_tool, cards, fake_llm):
        fake_llm.responses = ["SELECT 1"]
        db_tool.generate_sql("q", cards, model="llama3.2")
        prompt = fake_llm.calls[0]["prompt"]
        assert "Table: card_complaints" in prompt
        assert "Credit cards: Card complaints" in prompt
        assert "FROM card_complaints" in prompt  # profile example with {table} filled in
        assert "UPPER(company) LIKE" in prompt  # profile notes
        assert fake_llm.models == ["llama3.2"]


class TestDescribeTable:
    def test_lists_types_descriptions_and_values(self, db_tool, cards):
        schema = db_tool.describe_table(cards)
        assert "- issue (VARCHAR): Main issue category. Values: 'Fees or interest'" in schema
        assert "'Closing your account'" in schema
        assert "Escrow" not in schema  # mortgage row is filtered out of this LOB's table
        assert "- date_received (DATE)" in schema
        assert "[from 2024-01-05 to 2024-02-11]" in schema

    def test_skips_the_rag_text_column(self, db_tool, cards):
        schema = db_tool.describe_table(cards)
        assert "- consumer_complaint_narrative (VARCHAR)" in schema
        assert "late fee" not in schema

    def test_all_null_columns_are_left_out(self, tmp_path, registry):
        db_path = tmp_path / "nulls.duckdb"
        with duckdb.connect(str(db_path)) as con:
            con.execute("CREATE TABLE t AS SELECT 'Fees' AS issue, NULL::VARCHAR AS notes")
        lob = dataclasses.replace(registry.get("transactions"), table="t", columns={})

        schema = SQLTool(db_path=db_path).describe_table(lob)
        assert "- issue (VARCHAR)" in schema
        assert "notes" not in schema

    def test_generic_dataset(self, db_tool, registry):
        schema = db_tool.describe_table(registry.get("transactions"))
        assert "- amount (DOUBLE): Amount in USD" in schema
        assert "- merchant (VARCHAR). Values: 'ACME', 'BOLT'" in schema

    def test_high_cardinality_columns_omitted_unless_listed(self, db_tool, cards, monkeypatch):
        monkeypatch.setattr("app.sql_tool.MAX_DISTINCT_FOR_VALUES", 1)
        schema = db_tool.describe_table(cards)
        assert "'NY'" not in schema  # state: 2 distinct > 1, not a value column
        assert "'CITIBANK, N.A.'" in schema  # company is in the profile's value_columns


class TestDuckDB:
    def test_missing_database_is_unavailable(self, sql_tool):
        assert sql_tool.is_available() is False
        result = sql_tool.execute_query("SELECT 1")
        assert "not found" in result["error"]
        assert "ingestion.load_duckdb" in result["error"]

    def test_available_and_row_counts(self, db_tool):
        assert db_tool.is_available() is True
        assert db_tool.row_count("card_complaints") == 3
        assert db_tool.row_count("not_pulled") is None

    def test_execute_returns_rows_as_dicts(self, db_tool):
        result = db_tool.execute_query(
            "SELECT issue, COUNT(*) AS n FROM card_complaints GROUP BY issue ORDER BY n DESC"
        )
        assert result["results"] == [
            {"issue": "Fees or interest", "n": 2},
            {"issue": "Closing your account", "n": 1},
        ]
        assert result["row_count"] == 2

    def test_explain_reports_bad_column(self, db_tool):
        assert "nope" in db_tool.explain("SELECT nope FROM card_complaints")
        assert db_tool.explain("SELECT issue FROM card_complaints") is None

    def test_connection_is_read_only(self, db_tool):
        assert "read-only" in db_tool.execute_query("DELETE FROM card_complaints")["error"]

    def test_local_files_are_blocked(self, db_tool, tmp_path):
        secret = tmp_path / "secret.csv"
        secret.write_text("a\n1\n")
        result = db_tool.execute_query(f"SELECT * FROM read_csv('{secret}')")
        assert "disabled by configuration" in result["error"]

    def test_query_timeout(self, db_tool):
        db_tool.timeout_seconds = 0.01
        result = db_tool.execute_query(
            "SELECT COUNT(*) FROM range(100000000) a, range(1000) b WHERE a.range % 7 = b.range"
        )
        assert "timed out" in result["error"]


class TestExecuteSQLQuery:
    @pytest.fixture(autouse=True)
    def _tool(self, monkeypatch, db_tool):
        monkeypatch.setattr("app.sql_tool.get_sql_tool", lambda: db_tool)

    def test_missing_database_returns_error_without_llm(
        self, monkeypatch, sql_tool, cards, fake_llm
    ):
        monkeypatch.setattr("app.sql_tool.get_sql_tool", lambda: sql_tool)
        result = execute_sql_query("q", cards)
        assert "not found" in result["error"]
        assert fake_llm.calls == []

    def test_lob_without_loaded_table(self, registry, fake_llm):
        result = execute_sql_query("q", registry.get("not_pulled"))
        assert "not_pulled" in result["error"]
        assert "load_duckdb" in result["error"]
        assert fake_llm.calls == []

    def test_runs_against_the_lob_table(self, cards, fake_llm):
        fake_llm.responses = ["SELECT COUNT(*) AS n FROM card_complaints"]
        result = execute_sql_query("q", cards, model="qwen")
        assert result["results"] == [{"n": 3}]
        assert result["sql"].endswith("LIMIT 100")
        assert fake_llm.models == ["qwen"]

    def test_retries_with_error_feedback(self, cards, fake_llm):
        fake_llm.responses = ["DROP TABLE t", "SELECT 1 AS x LIMIT 1"]
        result = execute_sql_query("q", cards)
        assert len(fake_llm.calls) == 2
        assert "Only SELECT queries are allowed" in fake_llm.calls[1]["prompt"]
        assert result["results"] == [{"x": 1}]

    def test_explain_error_is_fed_back(self, cards, fake_llm):
        fake_llm.responses = [
            "SELECT nope FROM card_complaints",
            "SELECT COUNT(*) AS n FROM card_complaints",
        ]
        result = execute_sql_query("q", cards)
        assert "nope" in fake_llm.calls[1]["prompt"]
        assert result["results"] == [{"n": 3}]

    def test_gives_up_after_max_attempts(self, cards, fake_llm):
        fake_llm.responses = ["DROP TABLE t", "DELETE FROM t"]
        result = execute_sql_query("q", cards)
        assert result["error"].startswith("SQL generation failed")
        assert result["results"] == []
