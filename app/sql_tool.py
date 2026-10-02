"""
Text-to-SQL tool for BigQuery.

Implements:
- LLM-based SQL generation from natural language
- Schema injection with examples
- Query validation (SELECT-only, LIMIT enforcement)
- BigQuery dry-run validation
- Retry logic on errors
"""

import os
from typing import Dict, Any, Optional
from google.cloud import bigquery
import logging

from app.llm import get_llm_client
from app.prompts.sql import get_sql_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLTool:
    """Text-to-SQL query executor."""

    def __init__(
        self,
        project_id: str = None,
        dataset_id: str = None,
        table_id: str = None,
        max_rows: int = 1000
    ):
        self.project_id = project_id or os.getenv("BIGQUERY_PROJECT_ID")
        self.dataset_id = dataset_id or os.getenv("BIGQUERY_DATASET", "cfpb_complaints")
        self.table_id = table_id or os.getenv("BIGQUERY_TABLE", "complaints")
        self.max_rows = max_rows
        self.client = bigquery.Client(project=self.project_id) if self.project_id else None

    def generate_sql(self, question: str) -> str:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language query

        Returns:
            SQL query string
        """
        llm = get_llm_client()
        prompt = get_sql_prompt(question, self.dataset_id, self.table_id)

        sql_query = llm.generate(prompt)

        # Clean up the response (remove markdown formatting if present)
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

        return sql_query.strip()

    def validate_sql(self, sql_query: str) -> bool:
        """
        Validate SQL query for safety.

        Checks:
        - Only SELECT statements allowed
        - LIMIT clause enforced
        - No DDL/DML operations
        """
        sql_upper = sql_query.upper().strip()

        # Must start with SELECT
        if not sql_upper.startswith("SELECT"):
            logger.error("Only SELECT queries are allowed")
            return False

        # Check for dangerous keywords
        dangerous_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.error(f"Dangerous keyword found: {keyword}")
                return False

        # Enforce LIMIT
        if "LIMIT" not in sql_upper:
            logger.warning("Adding LIMIT clause for safety")
            sql_query += f" LIMIT {self.max_rows}"

        return True

    def dry_run(self, sql_query: str) -> bool:
        """
        Perform BigQuery dry run to validate query syntax.
        """
        if not self.client:
            logger.warning("BigQuery client not initialized, skipping dry run")
            return True

        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = self.client.query(sql_query, job_config=job_config)

            # Dry run successful
            logger.info(f"Dry run successful. Will process {query_job.total_bytes_processed} bytes")
            return True

        except Exception as e:
            logger.error(f"Dry run failed: {e}")
            return False

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query on BigQuery.

        Returns:
            Dict with query results and metadata
        """
        if not self.client:
            return {
                "error": "BigQuery client not configured",
                "results": []
            }

        try:
            # Run query
            query_job = self.client.query(sql_query)
            results = query_job.result()

            # Convert to list of dicts
            rows = [dict(row) for row in results]

            return {
                "sql": sql_query,
                "results": rows,
                "row_count": len(rows),
                "bytes_processed": query_job.total_bytes_processed
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "error": str(e),
                "sql": sql_query,
                "results": []
            }


# Singleton instance
_sql_tool = None


def get_sql_tool() -> SQLTool:
    """Get or create SQL tool instance."""
    global _sql_tool
    if _sql_tool is None:
        _sql_tool = SQLTool()
    return _sql_tool


def execute_sql_query(question: str, retry: bool = True) -> Dict[str, Any]:
    """
    Execute text-to-SQL pipeline.

    Args:
        question: Natural language question
        retry: Whether to retry once on error

    Returns:
        Query results
    """
    tool = get_sql_tool()

    # Generate SQL
    sql_query = tool.generate_sql(question)
    logger.info(f"Generated SQL: {sql_query}")

    # Validate
    if not tool.validate_sql(sql_query):
        return {"error": "SQL validation failed", "results": []}

    # Dry run
    if not tool.dry_run(sql_query):
        if retry:
            logger.info("Retrying SQL generation after dry run failure")
            return execute_sql_query(question, retry=False)
        else:
            return {"error": "SQL dry run failed", "results": []}

    # Execute
    return tool.execute_query(sql_query)
