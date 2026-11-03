"""Tests for text-to-SQL generation and validation."""

import duckdb
import pytest

from app.sql_tool import SQLTool, SQLValidationError, execute_sql_query


@pytest.fixture
def sql_tool(tmp_path):
    """Tool pointed at a path with no database, so nothing touches the real data."""
    return SQLTool(db_path=tmp_path / "missing.duckdb", table="complaints", max_rows=100)


@pytest.fixture
def db_tool(tmp_path):
    """Tool backed by a tiny real DuckDB database."""
    db_path = tmp_path / "test.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE complaints (company VARCHAR, issue VARCHAR, date_received DATE)")
        con.execute(
            "INSERT INTO complaints VALUES "
            "('CITIBANK, N.A.', 'Fees or interest', '2024-01-05'), "
            "('CITIBANK, N.A.', 'Fees or interest', '2024-02-10'), "
            "('CITIBANK, N.A.', 'Closing your account', '2024-02-11')"
        )
    return SQLTool(db_path=db_path, table="complaints", max_rows=100)


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
    def test_strips_markdown_fences(self, sql_tool, fake_llm):
        fake_llm.responses = ["```sql\nSELECT 1 LIMIT 1\n```"]
        assert sql_tool.generate_sql("q") == "SELECT 1 LIMIT 1"

    def test_prompt_mentions_table(self, sql_tool, fake_llm):
        fake_llm.responses = ["SELECT 1"]
        sql_tool.generate_sql("q")
        prompt = fake_llm.calls[0]["prompt"]
        assert "Table: complaints" in prompt
        assert "DuckDB" in prompt


class TestDuckDB:
    def test_missing_database_is_unavailable(self, sql_tool):
        assert sql_tool.is_available() is False
        result = sql_tool.execute_query("SELECT 1")
        assert "not found" in result["error"]
        assert "ingestion.load_duckdb" in result["error"]

    def test_available_with_table(self, db_tool):
        assert db_tool.is_available() is True

    def test_execute_returns_rows_as_dicts(self, db_tool):
        result = db_tool.execute_query(
            "SELECT issue, COUNT(*) AS n FROM complaints GROUP BY issue ORDER BY n DESC"
        )
        assert result["results"] == [
            {"issue": "Fees or interest", "n": 2},
            {"issue": "Closing your account", "n": 1},
        ]
        assert result["row_count"] == 2

    def test_explain_reports_bad_column(self, db_tool):
        assert "nope" in db_tool.explain("SELECT nope FROM complaints")
        assert db_tool.explain("SELECT issue FROM complaints") is None

    def test_connection_is_read_only(self, db_tool):
        assert "read-only" in db_tool.execute_query("DELETE FROM complaints")["error"]

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
    def _tool(self, monkeypatch, sql_tool):
        monkeypatch.setattr("app.sql_tool.get_sql_tool", lambda: sql_tool)

    def test_missing_database_returns_error(self, fake_llm):
        fake_llm.responses = ["SELECT issue FROM t"]
        result = execute_sql_query("q")
        assert "not found" in result["error"]
        assert result["sql"].endswith("LIMIT 100")

    def test_explain_error_is_fed_back(self, monkeypatch, db_tool, fake_llm):
        monkeypatch.setattr("app.sql_tool.get_sql_tool", lambda: db_tool)
        fake_llm.responses = ["SELECT nope FROM complaints", "SELECT COUNT(*) AS n FROM complaints"]
        result = execute_sql_query("q")
        assert "nope" in fake_llm.calls[1]["prompt"]
        assert result["results"] == [{"n": 3}]

    def test_retries_with_error_feedback(self, fake_llm):
        fake_llm.responses = ["DROP TABLE t", "SELECT 1 LIMIT 1"]
        result = execute_sql_query("q")
        assert len(fake_llm.calls) == 2
        assert "Only SELECT queries are allowed" in fake_llm.calls[1]["prompt"]
        assert result["sql"] == "SELECT 1 LIMIT 1"

    def test_gives_up_after_max_attempts(self, fake_llm):
        fake_llm.responses = ["DROP TABLE t", "DELETE FROM t"]
        result = execute_sql_query("q")
        assert result["error"].startswith("SQL generation failed")
        assert result["results"] == []
