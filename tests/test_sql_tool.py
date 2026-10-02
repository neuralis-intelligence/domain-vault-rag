"""
Tests for SQL tool.

Tests text-to-SQL generation and validation.
"""

import pytest
from app.sql_tool import SQLTool


class TestSQLTool:
    """Test cases for SQL tool."""

    @pytest.fixture
    def sql_tool(self):
        """Create SQL tool instance for testing."""
        return SQLTool(
            project_id="test-project",
            dataset_id="test_dataset",
            table_id="complaints"
        )

    def test_validate_select_query(self, sql_tool):
        """Test that SELECT queries pass validation."""
        sql = "SELECT * FROM complaints LIMIT 10"
        assert sql_tool.validate_sql(sql) is True

    def test_reject_drop_query(self, sql_tool):
        """Test that DROP queries are rejected."""
        sql = "DROP TABLE complaints"
        assert sql_tool.validate_sql(sql) is False

    def test_reject_delete_query(self, sql_tool):
        """Test that DELETE queries are rejected."""
        sql = "DELETE FROM complaints WHERE company = 'Chase'"
        assert sql_tool.validate_sql(sql) is False

    def test_reject_insert_query(self, sql_tool):
        """Test that INSERT queries are rejected."""
        sql = "INSERT INTO complaints VALUES ('test')"
        assert sql_tool.validate_sql(sql) is False

    def test_enforce_limit(self, sql_tool):
        """Test that LIMIT clause is enforced."""
        # This test would check if LIMIT is added when missing
        # Implementation depends on whether validate_sql modifies the query
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
