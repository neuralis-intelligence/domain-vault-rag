"""Tests for text-to-SQL generation and validation."""

import pytest

from app.sql_tool import SQLTool, SQLValidationError, execute_sql_query


@pytest.fixture
def sql_tool():
    # No project id -> no BigQuery client, so no credentials are needed.
    return SQLTool(project_id=None, dataset_id="test_dataset", table_id="complaints", max_rows=100)


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
        assert "`test_dataset.complaints`" in fake_llm.calls[0]["prompt"]


class TestExecuteSQLQuery:
    @pytest.fixture(autouse=True)
    def _tool(self, monkeypatch, sql_tool):
        monkeypatch.setattr("app.sql_tool.get_sql_tool", lambda: sql_tool)

    def test_unconfigured_bigquery_returns_error(self, fake_llm):
        fake_llm.responses = ["SELECT issue FROM t"]
        result = execute_sql_query("q")
        assert "not configured" in result["error"]
        assert result["sql"].endswith("LIMIT 100")

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
