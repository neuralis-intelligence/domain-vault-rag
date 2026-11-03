"""
Text-to-SQL tool for DuckDB.

Implements:
- LLM-based SQL generation from natural language
- Schema injection with examples
- Query validation (SELECT-only, single statement, LIMIT enforcement)
- EXPLAIN validation (catches syntax / unknown-column errors before running)
- One retry with the error fed back to the LLM

The database is opened read-only with external file/network access disabled, a second
safety layer on top of the keyword guardrails. Build it with `python -m ingestion.load_duckdb`.
"""

import logging
import re
import threading
from pathlib import Path
from typing import Any

import duckdb

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
_TRAILING_LIMITS_RE = re.compile(r"(?:\s*\bLIMIT\s+\d+)+\s*$", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)


# Generated SQL may only see the database itself: no reading local files or URLs
# (read_csv, read_parquet, ...), no installing/loading extensions, and the settings
# can't be changed back by a query.
_LOCKED_DOWN_CONFIG = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "lock_configuration": True,
}


class SQLValidationError(ValueError):
    """Raised when generated SQL fails the safety checks."""


class SQLTool:
    """Text-to-SQL query executor backed by a local DuckDB file."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        table: str | None = None,
        max_rows: int | None = None,
        timeout_seconds: float | None = None,
    ):
        self.db_path = Path(db_path) if db_path else settings.duckdb_path
        self.table = table or settings.sql_table
        self.max_rows = max_rows or settings.sql_max_rows
        self.timeout_seconds = timeout_seconds or settings.sql_timeout_seconds
        self._con: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.Lock()
        self.client_error = ""

    @property
    def connection(self) -> duckdb.DuckDBPyConnection | None:
        """
        Read-only DuckDB connection, opened on first use.
        Returns None (and sets `client_error`) if the database is missing or can't be opened.
        """
        with self._lock:
            if self._con is None:
                if not self.db_path.exists():
                    self.client_error = (
                        f"DuckDB database not found at {self.db_path}. "
                        "Run `python -m ingestion.load_duckdb`."
                    )
                    logger.error(self.client_error)
                    return None
                try:
                    self._con = duckdb.connect(
                        str(self.db_path), read_only=True, config=_LOCKED_DOWN_CONFIG
                    )
                except duckdb.Error as e:
                    self.client_error = f"DuckDB database could not be opened: {e}"
                    logger.error(self.client_error)
        return self._con

    def is_available(self) -> bool:
        """True if the database opens and the complaints table exists."""
        con = self.connection
        if con is None:
            return False
        try:
            con.cursor().execute(f"SELECT 1 FROM {self.table} LIMIT 1")
            return True
        except duckdb.Error as e:
            logger.warning("DuckDB health check failed: %s", e)
            return False

    def generate_sql(self, question: str, previous_error: str | None = None) -> str:
        """Generate a SQL query for `question`, stripping any markdown fences."""
        prompt = get_sql_prompt(question, self.table, previous_error)
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

        # Small models sometimes stack clauses ("... LIMIT 5\nLIMIT 1"), a syntax error.
        # Keep only the most restrictive one.
        stacked = _TRAILING_LIMITS_RE.search(sql)
        if stacked and len(found := re.findall(r"\d+", stacked.group())) > 1:
            sql = f"{sql[: stacked.start()]}\nLIMIT {min(map(int, found))}"

        limit = _LIMIT_RE.search(sql)
        if limit is None:
            sql = f"{sql}\nLIMIT {self.max_rows}"
        elif int(limit.group(1)) > self.max_rows:
            sql = _LIMIT_RE.sub(f"LIMIT {self.max_rows}", sql)

        return sql

    def explain(self, sql_query: str) -> str | None:
        """Validate the query with EXPLAIN (no execution). Returns an error message or None."""
        con = self.connection
        if con is None:
            logger.warning("DuckDB not available, skipping EXPLAIN")
            return None

        try:
            con.cursor().execute(f"EXPLAIN {sql_query}")
            return None
        except duckdb.Error as e:
            logger.error("EXPLAIN failed: %s", e)
            return str(e)

    def execute_query(self, sql_query: str) -> dict[str, Any]:
        """Execute SQL on DuckDB and return rows plus metadata."""
        con = self.connection
        if con is None:
            return {"error": self.client_error, "sql": sql_query, "results": []}

        # A cursor per query: DuckDB connections aren't safe to share across threads.
        cursor = con.cursor()
        timer = threading.Timer(self.timeout_seconds, cursor.interrupt)
        timer.start()
        try:
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
            return {"sql": sql_query, "results": rows, "row_count": len(rows)}
        except duckdb.InterruptException:
            error = f"Query timed out after {self.timeout_seconds}s"
            logger.error(error)
            return {"error": error, "sql": sql_query, "results": []}
        except duckdb.Error as e:
            logger.error("Query execution failed: %s", e)
            return {"error": str(e), "sql": sql_query, "results": []}
        finally:
            timer.cancel()
            cursor.close()


_sql_tool: SQLTool | None = None


def get_sql_tool() -> SQLTool:
    """Get or create the shared SQL tool."""
    global _sql_tool
    if _sql_tool is None:
        _sql_tool = SQLTool()
    return _sql_tool


def execute_sql_query(question: str, max_attempts: int = 2) -> dict[str, Any]:
    """Run the text-to-SQL pipeline: generate -> validate -> EXPLAIN -> execute."""
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

        error = tool.explain(sql_query)
        if error is None:
            return tool.execute_query(sql_query)

    return {"error": f"SQL generation failed: {error}", "sql": sql_query, "results": []}
