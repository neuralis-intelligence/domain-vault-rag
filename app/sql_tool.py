"""
Text-to-SQL tool for BigQuery.

Implements:
- LLM-based SQL generation from natural language
- Schema injection with examples
- Query validation (SELECT-only, single statement, LIMIT enforcement)
- BigQuery dry-run validation
- One retry with the error fed back to the LLM
"""

import logging
import re
from typing import Any

from google.cloud import bigquery

from app.config import settings
from app.llm import get_llm_client
from app.prompts.sql import get_sql_prompt

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = (
    "DROP",
    "DELETE",
    "INSERT",
    "UPDATE",
    "MERGE",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


class SQLValidationError(ValueError):
    """Raised when generated SQL fails the safety checks."""


class SQLTool:
    """Text-to-SQL query executor."""

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str | None = None,
        table_id: str | None = None,
        max_rows: int | None = None,
    ):
        self.project_id = project_id or settings.bigquery_project_id
        self.dataset_id = dataset_id or settings.bigquery_dataset
        self.table_id = table_id or settings.bigquery_table
        self.max_rows = max_rows or settings.sql_max_rows
        self._client: bigquery.Client | None = None
        self.client_error = "BigQuery not configured (set BIGQUERY_PROJECT_ID)"

    @property
    def client(self) -> bigquery.Client | None:
        """
        BigQuery client, created on first use so construction needs no credentials.
        Returns None (and sets `client_error`) if it is not configured or cannot authenticate.
        """
        if self._client is None and self.project_id:
            try:
                self._client = bigquery.Client(project=self.project_id)
            except Exception as e:  # e.g. google.auth DefaultCredentialsError
                self.client_error = f"BigQuery client could not be created: {e}"
                logger.error(self.client_error)
        return self._client

    def generate_sql(self, question: str, previous_error: str | None = None) -> str:
        """Generate a SQL query for `question`, stripping any markdown fences."""
        prompt = get_sql_prompt(question, self.dataset_id, self.table_id, previous_error)
        sql_query = get_llm_client().generate(prompt, temperature=0.0)
        return _CODE_FENCE_RE.sub("", sql_query.strip()).strip()

    def validate_sql(self, sql_query: str) -> str:
        """
        Check generated SQL for safety and return the (possibly amended) query.

        - Only a single SELECT / WITH statement is allowed
        - No DDL/DML keywords
        - A LIMIT clause is enforced and capped at `max_rows`

        Raises:
            SQLValidationError: if the query is unsafe.
        """
        sql = sql_query.strip().rstrip(";").strip()

        if ";" in sql:
            raise SQLValidationError("Multiple SQL statements are not allowed")

        if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise SQLValidationError("Only SELECT queries are allowed")

        match = _FORBIDDEN_RE.search(sql)
        if match:
            raise SQLValidationError(f"Forbidden keyword found: {match.group(1).upper()}")

        limit = _LIMIT_RE.search(sql)
        if limit is None:
            sql = f"{sql}\nLIMIT {self.max_rows}"
        elif int(limit.group(1)) > self.max_rows:
            sql = _LIMIT_RE.sub(f"LIMIT {self.max_rows}", sql)

        return sql

    def dry_run(self, sql_query: str) -> str | None:
        """Validate query syntax with a BigQuery dry run. Returns an error message or None."""
        if not self.client:
            logger.warning("BigQuery client not configured, skipping dry run")
            return None

        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = self.client.query(sql_query, job_config=job_config)
            logger.info("Dry run OK; query will process %s bytes", job.total_bytes_processed)
            return None
        except Exception as e:  # google.api_core raises many exception types
            logger.error("Dry run failed: %s", e)
            return str(e)

    def execute_query(self, sql_query: str) -> dict[str, Any]:
        """Execute SQL on BigQuery and return rows plus metadata."""
        if not self.client:
            return {"error": self.client_error, "sql": sql_query, "results": []}

        try:
            job = self.client.query(sql_query)
            rows = [dict(row) for row in job.result(timeout=settings.sql_timeout_seconds)]
            return {
                "sql": sql_query,
                "results": rows,
                "row_count": len(rows),
                "bytes_processed": job.total_bytes_processed,
            }
        except Exception as e:
            logger.error("Query execution failed: %s", e)
            return {"error": str(e), "sql": sql_query, "results": []}


_sql_tool: SQLTool | None = None


def get_sql_tool() -> SQLTool:
    """Get or create the shared SQL tool."""
    global _sql_tool
    if _sql_tool is None:
        _sql_tool = SQLTool()
    return _sql_tool


def execute_sql_query(question: str, max_attempts: int = 2) -> dict[str, Any]:
    """Run the text-to-SQL pipeline: generate -> validate -> dry run -> execute."""
    tool = get_sql_tool()
    error: str | None = None
    sql_query = ""

    for attempt in range(1, max_attempts + 1):
        sql_query = tool.generate_sql(question, previous_error=error)
        logger.info("Generated SQL (attempt %d): %s", attempt, sql_query)

        try:
            sql_query = tool.validate_sql(sql_query)
        except SQLValidationError as e:
            error = str(e)
            continue

        error = tool.dry_run(sql_query)
        if error is None:
            return tool.execute_query(sql_query)

    return {"error": f"SQL generation failed: {error}", "sql": sql_query, "results": []}
