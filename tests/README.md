# Tests

Unit and integration tests for the Complaint Insights Agent.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_router.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

## Test Structure

- `test_router.py` - Tests for query routing logic
- `test_sql_tool.py` - Tests for SQL generation and validation
- `test_rag_tool.py` - Tests for RAG search functionality
- `test_api.py` - Integration tests for FastAPI endpoints

## Requirements

Install test dependencies:
```bash
pip install pytest pytest-cov httpx
```

## Notes

Some tests require external services:
- RAG tests need Qdrant running
- SQL tests may need BigQuery credentials (or use mocks)
- API tests use FastAPI TestClient (no external services needed)

For CI/CD, consider using mocks or Docker containers for dependencies.
