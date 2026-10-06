# Tests

Unit and integration tests for the Complaint Insights Agent.

## Running Tests

```bash
pytest                                  # unit tests only (default, no services needed)
pytest tests/test_sql_tool.py -v        # one file
pytest -m integration                   # live tests: need Ollama, Qdrant (+ data), model download
pytest -m ""                            # everything
```

Install test dependencies with `pip install -r requirements-dev.txt`.

## Test Structure

| File | Covers |
|---|---|
| `test_router.py` | Route parsing (`sql`/`rag`/`hybrid`) and the router's LLM call |
| `test_sql_tool.py` | SQL guardrails (SELECT-only, LIMIT cap), fence stripping, retry with error feedback |
| `test_rag_tool.py` | Qdrant filter building, result formatting, error handling |
| `test_graph.py` | LangGraph workflow for each route, hybrid SQL→RAG filters and fallback |
| `test_llm.py` | Ollama client options, synthesis context building |
| `test_api.py` | FastAPI endpoints with the graph mocked via `dependency_overrides` |
| `test_ingestion.py` | Cleaning pipeline and Qdrant point payloads |

## Notes

- Unit tests mock the LLM (`fake_llm` fixture in `conftest.py`), Qdrant and BigQuery, so they
  run offline in a few seconds.
- Tests marked `@pytest.mark.integration` are skipped by default (see `pyproject.toml`).
  The live router tests need `ollama pull qwen2.5:7b-instruct`; the live search test needs
  narratives embedded with `python -m ingestion.embed_narratives`.
