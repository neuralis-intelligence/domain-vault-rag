# Complaint Insights Agent

A hybrid LLM agent that answers natural-language questions about financial data using local open-source models. The user picks a **line of business** (credit cards, mortgages, student loans, vehicle loans, checking & savings, or their own dataset) and a **local model**; the agent routes each question to text-to-SQL, RAG, or both, and returns a grounded answer.

**Example:** "What is the most recurring issue at JPMorgan?" → *"Problem with a purchase shown on your statement" (5,571 complaints)*

---

## Goal

Let non-technical users query ~322K CFPB complaint records (Aug 2023 – Aug 2026, 1,756 companies) in plain English. The system picks the best retrieval strategy (SQL aggregation, semantic search, or hybrid) and generates accurate, cited answers with a local LLM.

---

## Architecture

Two kinds of questions need different tools:

| Question Type | Example | Tool |
|---|---|---|
| Counting / ranking / trends | "Top issue at JPMorgan?" | **Text-to-SQL** on DuckDB |
| What customers say | "What do people complain about in Chase fraud cases?" | **RAG** on complaint narratives |
| Hybrid | "Top issue at JPMorgan and what are people saying?" | SQL first, then RAG filtered by the SQL result |

An LLM **router** (LangGraph) picks the path for each question. Every step runs against the **line of business (LOB)** and the **model** the user chose:

- **Line of business** → which DuckDB table the SQL is written for, which columns and values the SQL prompt describes, and which Qdrant collection/filter the RAG search uses. LOBs are defined in [`config/lobs.toml`](config/lobs.toml). For now the user picks the LOB; a classifier agent can pick it later (see the roadmap).
- **Model** → the Ollama model used for routing, SQL generation and the answer. The UI lists every installed chat model, like the model menu in a chat app.

```
User (question + LOB + model) → Streamlit UI → FastAPI /ask → LangGraph
                                      ├── route (LLM: sql | rag | hybrid)
                                      ├── sql:    LLM → SQL → guardrails → EXPLAIN → DuckDB
                                      ├── rag:    embed question → Qdrant search (+ metadata filters)
                                      ├── hybrid: sql, then rag filtered by company/issue from the SQL result
                                      └── synthesize (LLM answer citing numbers / complaint IDs)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Ollama + Qwen2.5 7B Instruct (or Llama 3.1 8B) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (sentence-transformers, 384-dim) |
| **Vector DB** | Qdrant (Docker) |
| **SQL Database** | DuckDB (local file, read-only at query time) |
| **Agent Framework** | LangGraph |
| **API** | FastAPI |
| **Frontend** | Streamlit + Plotly |
| **Containerization** | Docker Compose |
| **Quality** | pytest, ruff |

---

## Repository Structure

```
domain-vault-rag/
├── README.md
├── LICENSE
├── pyproject.toml              # pytest + ruff configuration
├── requirements.txt            # Runtime dependencies (pinned)
├── requirements-dev.txt        # + pytest, ruff, notebook libraries
├── Dockerfile                  # One image for the API and the UI
├── docker-compose.yml          # Qdrant (+ optional Ollama, API, UI profiles)
├── .env.example                # Environment variable template → copy to .env
│
├── app/                        # FastAPI backend + agent
│   ├── config.py               # Project paths + settings (loads .env)
│   ├── main.py                 # GET /, /health, /models, /lobs; POST /ask, /feedback
│   ├── lob.py                  # Line-of-business registry (loads config/lobs.toml)
│   ├── graph.py                # LangGraph: route → sql/rag/hybrid → synthesize
│   ├── router.py               # LLM query classification
│   ├── sql_tool.py             # DuckDB text-to-SQL with guardrails
│   ├── rag_tool.py             # Qdrant semantic search
│   ├── llm.py                  # Ollama client + answer synthesis
│   └── prompts/                # Prompt templates (router, sql, synthesis)
│
├── ingestion/                  # ETL pipeline, run in this order
│   ├── pull_cfpb.py            # 1. Download raw CSVs from the CFPB API
│   ├── clean.py                # 2. Clean + unify → data/processed/complaints.parquet
│   ├── load_duckdb.py          # 3. Build the DuckDB file: one table per LOB
│   └── embed_narratives.py     # 4. Embed each LOB's text into Qdrant
│
├── config/
│   └── lobs.toml               # Lines of business: data source, table, prompt hints, RAG settings
│
├── data/                       # Contents are gitignored (only .gitkeep is tracked)
│   ├── raw/                    # credit_card.csv, mortgage.csv
│   └── processed/              # complaints.parquet, complaints.duckdb (+ notebook CSV exports)
│
├── notebooks/                  # Exploratory data analysis
│   ├── credit_card_exploration.ipynb
│   └── mortgage_exploration.ipynb
│
├── ui/
│   └── streamlit_app.py        # Chat UI, SQL result charts, narrative viewer
│
├── eval/
│   ├── questions.jsonl         # 25 questions (13 sql, 9 rag, 3 hybrid) with expected routes
│   └── run_eval.py             # Routing accuracy / answer quality / latency
│
└── tests/                      # Offline unit tests + opt-in integration tests
```

---

## Recent Changes

### UI refresh
- The line-of-business and model pickers sit next to the message box, like a chat app's model menu.
- A new chat shows example questions as chips. Answers show badges (route, dataset, model, time), and the data (chart / table / SQL) and sources are in collapsible sections.
- Light theme in `.streamlit/config.toml`; it follows the system dark mode.

### Lines of business + model picker
- **Choose the dataset:** `config/lobs.toml` defines five CFPB lines of business (credit cards, mortgages, student loans, vehicle loans, checking & savings), each its own DuckDB table, plus a template for adding your own Parquet datasets with different columns. `/ask` takes `"lob"`; the UI has a picker.
- **Schema-aware SQL prompt:** the prompt now lists the chosen table's real columns, types, value lists (e.g. every `issue` in that LOB) and date ranges, read from DuckDB, instead of hard-coded credit card/mortgage values. That's what lets a new dataset work without writing a prompt.
- **Choose the model:** `GET /models` lists installed Ollama chat models (embedding models and anything over `MAX_MODEL_SIZE_GB` are hidden; `ALLOWED_MODELS` sets an allowlist). `/ask` takes `"model"`; the UI has a picker and shows which model answered.
- **RAG per LOB:** each LOB names its Qdrant collection and a fixed filter. The CFPB LOBs share `complaint_narratives` filtered by `product`, so the existing embeddings are reused; `embed_narratives --lob <id>` embeds a new LOB and skips points that already exist.
- `pull_cfpb --products ...` and `clean` handle any CFPB product listed in `CFPB_PRODUCTS`.
- Removed settings: `SQL_TABLE`, `QDRANT_COLLECTION_NAME` (both now per LOB in `lobs.toml`).

### SQL backend: BigQuery → DuckDB
The BigQuery sandbox deleted the loaded data after 60 days, and keeping it would have needed a billing account. The SQL path now runs on a local DuckDB file instead: free, no account, no expiry, and faster at this size.
- New `ingestion/load_duckdb.py` builds `data/processed/complaints.duckdb` (~320 MB) from the cleaned Parquet file in about 10 s, with the same schema and `company_aliases` table. It builds a temp file and swaps it in, so a running API isn't disturbed.
- `app/sql_tool.py` opens the database **read-only** with file/network access disabled (`enable_external_access=false`, configuration locked), so generated SQL can't read local files such as `read_csv('/etc/passwd')`. Queries are checked with `EXPLAIN` (replacing the BigQuery dry run) and interrupted after `SQL_TIMEOUT_SECONDS`.
- The SQL prompt now targets DuckDB syntax and the bare table name `complaints`.
- `/health` reports `duckdb: connected` only when the database opens and the table exists.
- All BigQuery code, dependencies (`google-cloud-bigquery`, `db-dtypes`) and settings were removed. Docker mounts `data/processed` read-only instead of a service-account key.

### Repo restructure

The project was reorganised from the old `banking-qa-agent/` folder (whose source files were empty stubs) into the layout above. Main changes:

**Structure & configuration**
- Notebooks moved to `notebooks/`, the download script to `ingestion/pull_cfpb.py`, and raw data to `data/raw/`.
- New `app/config.py`: every module now reads settings from `.env` and uses the shared data paths.
- `requirements.txt` lists all runtime dependencies (pinned); dev tools are in `requirements-dev.txt`.
- `.gitignore` now excludes the data files, `.env` and caches. Before this, the 286 MB of raw CSVs would have been committed.
- Added `pyproject.toml` (pytest + ruff), a `Dockerfile` and `.dockerignore`. `docker-compose.yml` used to reference Dockerfiles that didn't exist and an NVIDIA GPU block that fails on macOS.

**Pipeline**
- `ingestion/clean.py` is implemented, following the cleaning steps in the notebooks.
- The SQL loader is implemented (now `ingestion/load_duckdb.py`). Its schema matches the data: it adds `company_public_response` and `tags` and drops `consumer_disputed`, which isn't in the data, and it seeds the `company_aliases` table.
- `ingestion/embed_narratives.py` uses stable point IDs (the complaint ID), so re-runs don't create duplicates. It embeds and uploads batch by batch, makes payloads JSON-safe, adds keyword indexes for the filter fields, and has `--limit` and `--recreate` flags.
- Notebook cells were out of order, so "Run All" crashed. They're reordered and now read from and write to the correct `data/` folders.

**Agent & API**
- `/ask` now runs the LangGraph workflow instead of returning a placeholder.
- `/health` actually checks Ollama and Qdrant.
- An unreachable LLM returns a 503 with a clear message, and empty questions get a 422.
- Fixed bugs:
  - An explicit `temperature=0.0` was being replaced by 0.1.
  - The SQL `LIMIT` was added and then discarded.
  - The keyword guard rejected column names like `created_at`.
  - Hybrid mode never applied its filters.
  - The Qdrant `search()` call has been removed from the library; it now uses `query_points()`.
- The SQL prompt now uses the real company and product names. It previously said `'BANK OF AMERICA, N.A.'` and `'Credit card or prepaid card'`, so queries returned 0 rows. A failed query is retried once, with the error sent back to the LLM.
- The UI shows each service's health, allows longer timeouts for local LLMs, and no longer uses deprecated Streamlit arguments.

**Tests**
- 65 offline unit tests with the LLM and Qdrant mocked; SQL tests run against a throwaway DuckDB file. The live-service tests are marked `integration`.

---

## How to Run the Project

All commands run from the project root with the virtual environment active.

### Prerequisites
- Python 3.11+ (developed on 3.13)
- [Ollama](https://ollama.com/download), installed natively (on macOS this uses the GPU; Ollama in Docker can't)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)

### Step 1: Install
```bash
git clone git@github.com:neuralis-intelligence/domain-vault-rag.git
cd domain-vault-rag
python3 -m venv .myvenv
source .myvenv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # defaults work as-is
```

### Step 2: Check the install (no services needed)
```bash
pytest          # 65 offline unit tests, ~5 s
ruff check .    # lint
```

### Step 3: Start the services
```bash
docker compose up -d qdrant                 # Qdrant on :6333, dashboard at http://localhost:6333/dashboard
ollama serve                                # skip if the Ollama desktop app is running
ollama pull qwen2.5:7b-instruct             # ~4.7 GB, one-time
```

### Step 4: Run the data pipeline
```bash
python -m ingestion.pull_cfpb                       # optional: (re-)download data/raw/*.csv (slow, rate-limited)
                                                    #   one product: --products student_loan
python -m ingestion.clean                           # → data/processed/complaints.parquet (~5 s)
python -m ingestion.load_duckdb                     # → data/processed/complaints.duckdb, one table per LOB
python -m ingestion.embed_narratives --lob mortgage --limit 2000   # quick test sample
python -m ingestion.embed_narratives                # every LOB; skips texts already in Qdrant
```
`embed_narratives` options: `--lob <id> ...` to embed only some LOBs, `--force` to re-embed existing points, `--recreate` to drop and rebuild their collections (refused if another LOB shares the collection and isn't included).

**Running it a second time:** the pipeline only needs to run once. Qdrant keeps its data in a Docker volume, and the DuckDB file stays in `data/processed/`. Re-run it only after pulling new data, changing the cleaning logic, or deleting the Qdrant volume (`docker compose down -v`). After re-running `load_duckdb`, restart the API.

### Step 5: Run the app
```bash
uvicorn app.main:app --reload               # API + Swagger docs: http://localhost:8000/docs
streamlit run ui/streamlit_app.py           # UI: http://localhost:8501
```

Check that everything is connected:
```bash
curl localhost:8000/health
# {"status":"healthy","ollama":"connected","qdrant":"connected","duckdb":"connected"}

curl localhost:8000/lobs      # lines of business and how many rows each has loaded
curl localhost:8000/models    # installed models you can pick

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the most common issue at JPMorgan Chase?", "lob": "credit_card", "model": "llama3.2:latest"}'
```
`lob` and `model` are optional; they default to `default_lob` in `lobs.toml` and `OLLAMA_MODEL`.

### Step 6: Evaluate and run the live tests
```bash
python eval/run_eval.py                     # API must be running; writes eval/eval_results.csv
pytest -m integration                       # tests against live Ollama / Qdrant
```

### Docker alternative
```bash
docker compose --profile app up -d --build  # Qdrant + API + UI; uses Ollama on the host
```
The API container mounts `data/processed/` read-only, so build the DuckDB file (`python -m ingestion.load_duckdb`) on the host first.

### Notebooks
```bash
python -m ipykernel install --user --name domain-vault-rag-myvenv --display-name "Python (.myvenv: domain-vault-rag)"
pip install jupyterlab && jupyter lab notebooks/   # or open them in VS Code and pick that kernel
```
The notebooks are for exploration. The production cleaning step is `ingestion/clean.py`.

---

## Lines of Business

A line of business (LOB) is one dataset the user can ask about. Each `[lobs.<id>]` entry in [`config/lobs.toml`](config/lobs.toml) says:

| Key | Meaning |
|---|---|
| `name`, `description` | Shown in the UI and given to the LLM |
| `source` | Parquet file (or glob) the table is built from, relative to the project root |
| `where` | Optional row filter, e.g. `product = 'Mortgage'` |
| `table` | DuckDB table name |
| `profile` | Optional shared defaults; `cfpb_complaints` supplies column descriptions, SQL examples and RAG settings for CFPB data |
| `columns`, `value_columns`, `prompt_notes`, `sql_examples` | Optional hints for the SQL prompt |
| `example_questions` | Buttons in the UI sidebar |
| `[rag]` | Optional: `collection`, `text_column`, `id_column`, `payload_fields`, `filter_fields`, `filter`. Without it the LOB is SQL-only and the router is skipped |

The CFPB lines of business:

| id | Table | Rows | Text search |
|---|---|---|---|
| `credit_card` | `credit_card_complaints` | ~246K | ✓ 108K narratives |
| `mortgage` | `mortgage_complaints` | ~76K | ✓ 38K narratives |
| `student_loan` | `student_loan_complaints` | ~60K | SQL only |
| `vehicle_loan` | `vehicle_loan_complaints` | ~60K | SQL only |
| `checking_savings` | `checking_savings_complaints` | ~231K | SQL only |

All five come from `data/processed/complaints.parquet`. Credit cards and mortgages share the Qdrant collection `complaint_narratives`, each restricted to its own product.

> ⚠️ **The CFPB no longer publishes complaint narratives.** As of 2026-11-02 the narrative column is gone from both the public API and the bulk `complaints.csv.zip`, even for old complaints. That's why the three newer products are SQL-only (`rag = false`). The credit card and mortgage narratives exist only in your `data/raw/credit_card.csv`, `data/raw/mortgage.csv` and the Qdrant volume, so **back these up**. `pull_cfpb` won't overwrite a raw file that has narratives with one that doesn't; it saves the new download as `<key>.no_narratives.csv` instead.

### Adding a CFPB product
1. Add it to `CFPB_PRODUCTS` in `ingestion/pull_cfpb.py` (key → the product name as the CFPB API spells it).
2. Add a `[lobs.<key>]` entry with `profile = "cfpb_complaints"`, `where = "product = '...'"` and `rag.filter = { product = "..." }`.
3. Run `pull_cfpb --products <key>`, `clean`, `load_duckdb`, then restart the API. New downloads have no narratives, so set `rag = false` in the entry (and skip `embed_narratives`).

### Adding your own dataset
1. Save it as Parquet, e.g. `data/processed/card_transactions.parquet`.
2. Copy the commented template at the bottom of `lobs.toml` and fill in `name`, `source` and `table`. Everything else is optional: the SQL prompt already lists the table's columns, types and common values from the data. Column descriptions, `prompt_notes` and one or two `sql_examples` noticeably improve the SQL a small model writes.
3. Run `python -m ingestion.load_duckdb`. If you added a `[rag]` block, also run `python -m ingestion.embed_narratives --lob <id>`.
4. Restart the API. The new LOB appears in the UI picker.

LOBs whose source file doesn't exist yet are skipped by the loader and hidden in the UI (listed as "Not loaded yet").

### Choosing a model
The **Model** picker next to the message box lists every installed Ollama chat model, smallest first. Install more with e.g. `ollama pull llama3.2` (3B, fast) or `ollama pull qwen2.5:7b-instruct` (better SQL). `ALLOWED_MODELS` in `.env` restricts the list; `MAX_MODEL_SIZE_GB` (default 10) hides larger models. Smaller models are faster but route and write SQL less reliably; use `python eval/run_eval.py --model <name>` to compare them.

---

## SQL Data (DuckDB)

The SQL path (counts, rankings, trends) queries a local DuckDB file, `data/processed/complaints.duckdb`, with one table per line of business. The RAG path doesn't use it: it reads text from Qdrant, which `embed_narratives.py` fills from the same Parquet files.

### Build it
```bash
python -m ingestion.clean         # if data/processed/complaints.parquet doesn't exist yet
python -m ingestion.load_duckdb
```
This creates one table per LOB in `lobs.toml` (see [Lines of Business](#lines-of-business)) plus `company_aliases` (`alias → canonical_name`, e.g. `Chase → JPMORGAN CHASE & CO.`).

Re-running **replaces** the file, so it's safe to re-run after new data is pulled. Restart the API afterwards. The data files are gitignored; never commit them.

### Verify
```bash
curl -s localhost:8000/lobs       # rows per LOB
python -c "import duckdb; c = duckdb.connect('data/processed/complaints.duckdb', read_only=True); \
print(c.sql('SELECT COUNT(*) FROM credit_card_complaints'))"
# Expect: 246,212
```
Expected top issue at JPMorgan Chase (credit cards): *Problem with a purchase shown on your statement*, 5,571.

### How queries are kept safe
1. **Guardrails** in `validate_sql`: a single `SELECT`/`WITH` statement, no DDL/DML keywords, and one `LIMIT` capped at `SQL_MAX_ROWS` (stacked `LIMIT`s are collapsed).
2. **Read-only connection** with external access disabled and the configuration locked, so a query can't write, read local files or URLs, load extensions, or turn these settings back on.
3. **`EXPLAIN` check** before running; a failure (e.g. an unknown column) goes back to the LLM for one retry.
4. **Timeout**: queries running longer than `SQL_TIMEOUT_SECONDS` are interrupted.

The `where` filters and table names in `lobs.toml` are trusted configuration (table names must be plain identifiers); only the generated SQL is treated as untrusted.

### Sharing with teammates
The file is local. Teammates either run `clean` + `load_duckdb` themselves (they need `data/raw/*.csv`, or `pull_cfpb`), or you send them `complaints.parquet` and they run only `load_duckdb`. If you later need one shared copy in the cloud, MotherDuck (hosted DuckDB) runs the same SQL; the change is mostly the connection string in `app/sql_tool.py`.

---

## Usage

With everything running, open `http://localhost:8501`, pick a **line of business** and a **model** in the two menus next to the message box, and ask questions like:

- Credit cards: "What are the top 5 most common issues at Bank of America?" → SQL
- Credit cards: "What are customers saying about unauthorized charges at Capital One?" → RAG
- Mortgages: "What is the top issue at Wells Fargo and what are people saying about it?" → hybrid

A new chat shows example questions for the chosen line of business as clickable chips. Each answer shows badges for the route, line of business, model and response time, a collapsible **Data** section (chart, table and the generated SQL) and a **Sources** section with the matching texts and similarity scores. The sidebar has **New chat** and the status of Ollama, Qdrant and DuckDB. The theme (accent color, corners) is in `.streamlit/config.toml` and follows the system light/dark setting.

---

## Configuration Reference

All settings come from `.env` (template: `.env.example`) and are read in `app/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `LOB_REGISTRY_PATH` | `config/lobs.toml` | Lines of business (tables, prompt hints and Qdrant collections are configured there) |
| `DUCKDB_PATH` | `data/processed/complaints.duckdb` | Database file (relative to the project root) |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Vector store |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5:7b-instruct` | LLM server / default model |
| `ALLOWED_MODELS` / `MAX_MODEL_SIZE_GB` | empty / `10` | Models offered in the picker: an allowlist, or every installed chat model up to this size |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` | `0.1` / `1000` / `120` | LLM generation |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the model used for ingestion |
| `RAG_TOP_K` / `RAG_SCORE_THRESHOLD` | `5` / `0.5` | Number of narratives returned / minimum cosine similarity |
| `SQL_MAX_ROWS` / `SQL_TIMEOUT_SECONDS` | `1000` / `30` | SQL guardrails |

---

## Testing

```bash
pytest                    # unit tests (offline: LLM and Qdrant mocked, throwaway DuckDB)
pytest -m integration     # live tests against Ollama / Qdrant
pytest -m ""              # everything
```

See [tests/README.md](tests/README.md) for what each test file covers.

## Evaluation

With the API running:

```bash
python eval/run_eval.py   # --api-url, --questions, --output are optional
```

This measures:
- **Routing accuracy**: did the agent choose the expected tool?
- **Answer quality**: does the answer contain the expected terms?
- **Latency** and **error rate**, overall and per route

---

## Performance Considerations

- **SQL path**: 2 LLM calls (route + SQL) + 1 synthesis call. Expect several seconds on a local 7B model.
- **RAG path**: 1 LLM call + embedding + vector search + synthesis
- **Hybrid**: both of the above
- **Caching**: worth adding for repeated queries (stretch goal)

---

## Development Roadmap

### Phase 1: Data Engineering
- [x] Download CFPB credit card and mortgage complaint data (`ingestion/pull_cfpb.py`)
- [x] Exploratory data analysis (`notebooks/`)
- [x] Clean and normalize to a unified schema (`ingestion/clean.py`)
- [x] Load into DuckDB (`ingestion/load_duckdb.py`)
- [x] Company alias lookup table (loaded by `load_duckdb.py`)

### Phase 2: SQL Path
- [ ] Run Ollama locally with Qwen2.5 7B
- [x] Text-to-SQL with schema injection
- [x] Guardrails (SELECT-only, LIMIT enforcement, read-only sandboxed connection, EXPLAIN validation, retry)
- [ ] Validate on the test questions

### Phase 3: RAG Path
- [ ] Generate embeddings for the narratives (script ready, **not yet run**)
- [ ] Load into Qdrant with metadata filters
- [x] Filtered semantic search
- [x] Answer synthesis with citations

### Phase 4: Agent & API
- [x] LangGraph router (sql/rag/hybrid)
- [x] FastAPI `/ask` endpoint
- [x] Docker Compose setup (written, **not yet tested**)
- [ ] Run Qdrant and Ollama containers

### Phase 5: UI & Evaluation
- [x] Streamlit chat interface
- [x] Visualization of SQL results
- [ ] Run the evaluation on the question set
- [ ] Document performance metrics

---

## Stretch Goals

- [ ] Langfuse tracing for LLM observability
- [ ] Query result caching (Redis)
- [ ] Multi-turn conversations with context
- [ ] Support for additional CFPB product categories
- [ ] Groq free tier as a backup LLM provider

---

## Contributing

This is a learning/portfolio project. Suggestions and improvements are welcome via issues or PRs. Before opening a PR, run `ruff check . && ruff format --check . && pytest`.

---

## License

See the LICENSE file.

---

## Resume Summary

> Built a hybrid LLM agent (LangGraph, Ollama, DuckDB, Qdrant) that answers natural-language questions over 320K CFPB complaints, routing between text-to-SQL and RAG; achieved X% answer accuracy on a 25-question eval set.

---

## TODO: Remaining Work

Work through these roughly in order. Items 1–6 get the full system running end to end; the rest finish and polish it.

### 1. Housekeeping
- [ ] Delete the leftover Jupyter checkpoint folders. They're gitignored, but `data/raw/.ipynb_checkpoints/` holds 286 MB of duplicate CSVs:
      `find . -path ./.myvenv -prune -o -name .ipynb_checkpoints -type d -exec rm -rf {} +`
- [ ] Commit the restructure on `feature/repo-structure` (`git add -A` also records the removal of the old `banking-qa-agent/` folder), then open a PR to `main`.

### 2. Install the missing tools
- [ ] Docker Desktop and Ollama.
- [ ] `ollama pull qwen2.5:7b-instruct`

### 3. SQL data (see [SQL Data (DuckDB)](#sql-data-duckdb))
- [x] Run `python -m ingestion.load_duckdb` and check the row counts.

### 4. Qdrant / RAG
- [ ] `docker compose up -d qdrant`
- [ ] `python -m ingestion.embed_narratives --limit 2000`, test a few RAG questions, then run it on everything.
- [ ] Tune `RAG_SCORE_THRESHOLD`. Look at the scores the UI shows for good and bad matches: if too few narratives come back, lower it; if they're irrelevant, raise it.

### 5. First end-to-end run
- [ ] Start the API + UI, check that `/health` reports everything as connected, and try the example questions.
- [ ] `pytest -m integration`. The live router tests show whether the router prompt (`app/prompts/router.py`) needs tuning for your model.

### 6. Evaluation
- [ ] Improve `eval/questions.jsonl`. Several `expected_answer_contains` values are too generic to mean much (e.g. `["count", "number"]`). Replace them with real numbers or names from the SQL data (e.g. `"5571"`, `"Problem with a purchase shown on your statement"`).
- [ ] Add more hybrid questions (only 3 of 25 right now).
- [ ] Run `python eval/run_eval.py`, record the metrics in this README, and fill in the X% in the resume summary.

### 7. Feature gaps in the code
- [ ] **Line-of-business agent:** classify which LOB a question is about (e.g. an LLM call over the LOB names/descriptions in `lobs.toml`, or embedding similarity to each LOB's example questions) and pre-select it, keeping the user's pick as an override. `/ask` already accepts `lob`, so the classifier only has to fill it in.
- [ ] **Cross-LOB questions** ("credit card vs mortgage complaints at Chase"): today each question targets one table.
- [ ] **Use `company_aliases` in the SQL path.** It's loaded into DuckDB, but the SQL tool doesn't use it yet; the prompt relies on `UPPER(company) LIKE '%...%'` instead. Options: put the alias list in the SQL prompt, or teach the prompt to `JOIN` the alias table.
- [ ] **Hybrid filters:** `filters_from_sql_result` (`app/graph.py`) only picks up exact `company = '...'` conditions and the top result row. Queries that use `LIKE` produce no company filter.
- [ ] **`/feedback`** only logs. Store it somewhere (e.g. a separate local DuckDB/SQLite file or CSV; the main database is opened read-only).
- [ ] **Narratives in hybrid answers:** the SQL result rows and narratives go into one prompt. Check that long contexts don't overflow the model's context window (`MAX_SQL_ROWS_IN_CONTEXT` and `MAX_NARRATIVE_CHARS` in `app/llm.py`).
- [ ] **Streaming responses**, so the UI shows the answer as it's generated rather than after a long spinner.

### 8. Ops & quality
- [ ] Test `docker compose --profile app up --build` end to end. It's written but untested, because Docker wasn't installed.
- [ ] Add GitHub Actions CI running `ruff check .`, `ruff format --check .` and `pytest` on every PR.
- [ ] Re-run the notebooks so their saved outputs show the new paths (the outputs are still from before the restructure).
- [ ] Optionally make the notebooks import functions from `ingestion/clean.py` instead of duplicating the cleaning logic.
- [ ] Schedule `pull_cfpb → clean → load_duckdb → embed_narratives` to refresh the data (e.g. monthly).
