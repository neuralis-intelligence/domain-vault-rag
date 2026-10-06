# Complaint Insights Agent

A hybrid LLM agent that answers natural-language questions about CFPB bank complaints (credit cards and mortgages) using local open-source models. The agent routes each question to text-to-SQL, RAG, or both, and returns a grounded answer.

**Example:** "What is the most recurring issue at JPMorgan?" → *"Problem with a purchase shown on your statement" (5,571 complaints)*

---

## Goal

Let non-technical users query ~322K CFPB complaint records (Aug 2023 – Aug 2026, 1,756 companies) in plain English. The system picks the best retrieval strategy (SQL aggregation, semantic search, or hybrid) and generates accurate, cited answers with a local LLM.

---

## Architecture

Two kinds of questions need different tools:

| Question Type | Example | Tool |
|---|---|---|
| Counting / ranking / trends | "Top issue at JPMorgan?" | **Text-to-SQL** on BigQuery |
| What customers say | "What do people complain about in Chase fraud cases?" | **RAG** on complaint narratives |
| Hybrid | "Top issue at JPMorgan and what are people saying?" | SQL first, then RAG filtered by the SQL result |

An LLM **router** (LangGraph) picks the path for each question.

```
User → Streamlit UI → FastAPI /ask → LangGraph
                                      ├── route (LLM: sql | rag | hybrid)
                                      ├── sql:    LLM → SQL → guardrails → dry run → BigQuery
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
| **Data Warehouse** | BigQuery |
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
│   ├── main.py                 # GET /, GET /health, POST /ask, POST /feedback
│   ├── graph.py                # LangGraph: route → sql/rag/hybrid → synthesize
│   ├── router.py               # LLM query classification
│   ├── sql_tool.py             # BigQuery text-to-SQL with guardrails
│   ├── rag_tool.py             # Qdrant semantic search
│   ├── llm.py                  # Ollama client + answer synthesis
│   └── prompts/                # Prompt templates (router, sql, synthesis)
│
├── ingestion/                  # ETL pipeline, run in this order
│   ├── pull_cfpb.py            # 1. Download raw CSVs from the CFPB API
│   ├── clean.py                # 2. Clean + unify → data/processed/complaints.parquet
│   ├── load_bigquery.py        # 3. Load BigQuery table + company_aliases
│   └── embed_narratives.py     # 4. Embed narratives into Qdrant
│
├── data/                       # Contents are gitignored (only .gitkeep is tracked)
│   ├── raw/                    # credit_card.csv, mortgage.csv
│   └── processed/              # complaints.parquet (+ notebook CSV exports)
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

## Recent Changes (repo restructure)

The project was reorganised from the old `banking-qa-agent/` folder (whose source files were empty stubs) into the layout above. Main changes:

**Structure & configuration**
- Notebooks moved to `notebooks/`, the download script to `ingestion/pull_cfpb.py`, and raw data to `data/raw/`.
- New `app/config.py`: every module now reads settings from `.env` and uses the shared data paths.
- `requirements.txt` lists all runtime dependencies (pinned); dev tools are in `requirements-dev.txt`.
- `.gitignore` now excludes the data files, `.env`, `credentials/` and caches. Before this, the 286 MB of raw CSVs would have been committed.
- Added `pyproject.toml` (pytest + ruff), a `Dockerfile` and `.dockerignore`. `docker-compose.yml` used to reference Dockerfiles that didn't exist and an NVIDIA GPU block that fails on macOS.

**Pipeline**
- `ingestion/clean.py` is implemented, following the cleaning steps in the notebooks.
- `ingestion/load_bigquery.py` is implemented. Its schema now matches the data: it adds `company_public_response` and `tags` and drops `consumer_disputed`, which isn't in the data. The table is partitioned by month and clustered by company and product, and the `company_aliases` table is seeded.
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
- 55 offline unit tests with the LLM, Qdrant and BigQuery mocked. The live-service tests are marked `integration`.

---

## How to Run the Project

All commands run from the project root with the virtual environment active.

### Prerequisites
- Python 3.11+ (developed on 3.13)
- [Ollama](https://ollama.com/download), installed natively (on macOS this uses the GPU; Ollama in Docker can't)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- A Google Cloud project with BigQuery + the [gcloud CLI](https://cloud.google.com/sdk/docs/install) (SQL path only)

### Step 1: Install
```bash
git clone git@github.com:neuralis-intelligence/domain-vault-rag.git
cd domain-vault-rag
python3 -m venv .myvenv
source .myvenv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # edit BIGQUERY_PROJECT_ID (see the BigQuery section below)
```

### Step 2: Check the install (no services needed)
```bash
pytest          # 55 offline unit tests, ~5 s
ruff check .    # lint
```

### Step 3: Start the services
```bash
docker compose up -d qdrant                 # Qdrant on :6333, dashboard at http://localhost:6333/dashboard
ollama serve                                # skip if the Ollama desktop app is running
ollama pull qwen2.5:7b-instruct             # ~4.7 GB, one-time
gcloud auth application-default login       # BigQuery credentials for local runs
```

### Step 4: Run the data pipeline
```bash
python -m ingestion.pull_cfpb                       # optional: re-download data/raw/*.csv (slow, rate-limited)
python -m ingestion.clean                           # → data/processed/complaints.parquet (~5 s)
python -m ingestion.load_bigquery                   # → BigQuery (see the BigQuery section below)
python -m ingestion.embed_narratives --limit 2000   # quick test sample
python -m ingestion.embed_narratives                # all ~147K narratives (long on CPU)
```
Add `--recreate` to `embed_narratives` to drop and rebuild the Qdrant collection.

### Step 5: Run the app
```bash
uvicorn app.main:app --reload               # API + Swagger docs: http://localhost:8000/docs
streamlit run ui/streamlit_app.py           # UI: http://localhost:8501
```

Check that everything is connected:
```bash
curl localhost:8000/health
# {"status":"healthy","ollama":"connected","qdrant":"connected","bigquery":"configured"}

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What is the most common issue at JPMorgan Chase?"}'
```

### Step 6: Evaluate and run the live tests
```bash
python eval/run_eval.py                     # API must be running; writes eval/eval_results.csv
pytest -m integration                       # tests against live Ollama / Qdrant
```

### Docker alternative
```bash
docker compose --profile app up -d --build  # Qdrant + API + UI; uses Ollama on the host
```
The API container needs a service-account key at `credentials/bigquery-key.json` for BigQuery, because `gcloud` login credentials aren't available inside the container.

### Notebooks
```bash
python -m ipykernel install --user --name domain-vault-rag-myvenv --display-name "Python (.myvenv: domain-vault-rag)"
pip install jupyterlab && jupyter lab notebooks/   # or open them in VS Code and pick that kernel
```
The notebooks are for exploration. The production cleaning step is `ingestion/clean.py`.

---

## BigQuery Setup

### Do I need to push the cleaned data to BigQuery?
**Yes.** The SQL path (counts, rankings, trends) queries BigQuery directly, so the cleaned table must exist there. The RAG path doesn't need BigQuery: it reads narratives from Qdrant, which `embed_narratives.py` fills from the local Parquet file.

- Push the **full cleaned dataset** (~322K rows, ~110 MB). Aggregations are only correct over all rows.
- Load it with `python -m ingestion.load_bigquery`. **Don't** commit data to git or upload the CSVs by hand: the script sets the correct schema, partitioning and clustering.
- The data is public CFPB data, and the CFPB already scrubs personal information from the narratives.
- **Cost:** load jobs are free, and this size fits well inside BigQuery's free tier (10 GB storage, 1 TB of queries per month).

### One-time setup (project owner)
1. **Create or choose a GCP project** in the [Cloud Console](https://console.cloud.google.com/) and note the **project ID** (e.g. `complaint-insights-123456`).
2. **Enable billing** on the project.
   > ⚠️ Don't use the BigQuery *sandbox* (no billing). Sandbox tables and partitions expire after 60 days, and since this table is partitioned by `date_received`, older complaints would be deleted. With billing enabled you still stay in the free tier.
3. **Enable the BigQuery API:**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable bigquery.googleapis.com
   ```
4. **Authenticate:**
   ```bash
   gcloud auth application-default login
   ```
5. **Set `.env`:**
   ```bash
   BIGQUERY_PROJECT_ID=YOUR_PROJECT_ID
   BIGQUERY_DATASET=cfpb_complaints     # default
   BIGQUERY_TABLE=complaints            # default
   ```
6. **Load the data:**
   ```bash
   python -m ingestion.clean
   python -m ingestion.load_bigquery
   ```
   This creates:
   | Object | Details |
   |---|---|
   | Dataset `cfpb_complaints` | Location `US` |
   | Table `cfpb_complaints.complaints` | 16 columns, partitioned by month on `date_received`, clustered by `company, product` |
   | Table `cfpb_complaints.company_aliases` | `alias → canonical_name` (e.g. `Chase → JPMORGAN CHASE & CO.`) |

   Re-running the script **replaces** both tables (`WRITE_TRUNCATE`), so it's safe to re-run after new data is pulled.
7. **Verify** in the BigQuery console or with `bq`:
   ```bash
   bq query --use_legacy_sql=false \
     'SELECT product, COUNT(*) AS n FROM `cfpb_complaints.complaints` GROUP BY product'
   # Expect: Credit card ≈ 246,212 and Mortgage ≈ 75,632

   bq query --use_legacy_sql=false \
     'SELECT issue, COUNT(*) AS n FROM `cfpb_complaints.complaints`
      WHERE company = "JPMORGAN CHASE & CO." GROUP BY issue ORDER BY n DESC LIMIT 1'
   # Expect: Problem with a purchase shown on your statement, 5571
   ```

### Giving teammates access
The app runs queries as *the person or service account running it*, so each teammate needs two roles:

| Role | Where | Why |
|---|---|---|
| **BigQuery Job User** (`roles/bigquery.jobUser`) | Project | Allows running query jobs |
| **BigQuery Data Viewer** (`roles/bigquery.dataViewer`) | Dataset `cfpb_complaints` | Read-only access to the tables |
| *BigQuery Data Editor* (`roles/bigquery.dataEditor`) | Dataset | **Only** for someone who will re-run `load_bigquery.py` |

Grant them like this:
```bash
# Project-level: run query jobs
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:teammate@example.com" --role="roles/bigquery.jobUser"
```
For dataset-level read access: BigQuery console → `cfpb_complaints` → **Sharing → Permissions → Add principal** → `teammate@example.com` → role **BigQuery Data Viewer**.

Each teammate then:
1. Runs `gcloud auth application-default login` with the account you granted.
2. Sets `BIGQUERY_PROJECT_ID=YOUR_PROJECT_ID` in their `.env` (the **project that owns the table**, not their own project).
3. Skips `load_bigquery.py`, since the table already exists.

### Service account (for Docker or a deployed API)
```bash
gcloud iam service-accounts create complaint-insights-api
SA=complaint-insights-api@YOUR_PROJECT_ID.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:$SA" --role="roles/bigquery.jobUser"
# + grant BigQuery Data Viewer on the dataset to $SA (console → Sharing → Permissions)
mkdir -p credentials
gcloud iam service-accounts keys create credentials/bigquery-key.json --iam-account=$SA
```
Then set `GOOGLE_APPLICATION_CREDENTIALS=./credentials/bigquery-key.json` in `.env`. `credentials/` is gitignored, so **never commit the key**.

### Changing dataset or table names
Names are configured in **`.env` only**; nothing else needs editing:

| Setting | Used by |
|---|---|
| `BIGQUERY_PROJECT_ID` | Loader + SQL tool (the project queries run in) |
| `BIGQUERY_DATASET` | Loader (creates it) + SQL prompt (`` `dataset.table` ``) |
| `BIGQUERY_TABLE` | Loader + SQL prompt |

Hard-coded values you may want to change:
- The alias table name `company_aliases` and the alias list: `ingestion/load_bigquery.py` (`ALIASES_TABLE`, `COMPANY_ALIASES`).
- Dataset location `US`: `create_dataset()` in `ingestion/load_bigquery.py`.
- The schema description and example company names in the SQL prompt: `app/prompts/sql.py`. Update these if you add columns or products.
- If you change the cleaned columns, keep `FINAL_COLUMNS` in `ingestion/clean.py`, `COMPLAINTS_SCHEMA` in `ingestion/load_bigquery.py` and the prompt schema in sync.

---

## Usage

With everything running, open `http://localhost:8501` and ask questions like:

- "What are the top 5 most common issues at Bank of America?" → SQL
- "What are customers saying about unauthorized charges at Capital One?" → RAG
- "What is the top mortgage issue at Wells Fargo and what are people saying about it?" → hybrid

The UI shows which route was taken, a table and chart for SQL results, and the matching complaint narratives with similarity scores.

---

## Configuration Reference

All settings come from `.env` (template: `.env.example`) and are read in `app/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `BIGQUERY_PROJECT_ID` | none | GCP project for BigQuery; the SQL path is disabled if unset |
| `BIGQUERY_DATASET` / `BIGQUERY_TABLE` | `cfpb_complaints` / `complaints` | Table queried by the SQL tool |
| `GOOGLE_APPLICATION_CREDENTIALS` | none | Service-account key path (optional when using `gcloud` login) |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_COLLECTION_NAME` | `localhost` / `6333` / `complaint_narratives` | Vector store |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `qwen2.5:7b-instruct` | LLM |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT_SECONDS` | `0.1` / `1000` / `120` | LLM generation |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the model used for ingestion |
| `RAG_TOP_K` / `RAG_SCORE_THRESHOLD` | `5` / `0.5` | Number of narratives returned / minimum cosine similarity |
| `SQL_MAX_ROWS` / `SQL_TIMEOUT_SECONDS` | `1000` / `30` | SQL guardrails |

---

## Testing

```bash
pytest                    # unit tests (offline: LLM, Qdrant, BigQuery are mocked)
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
- [ ] Load into BigQuery with partitioning/clustering (script ready, **not yet run**)
- [x] Company alias lookup table (loaded by `load_bigquery.py`)

### Phase 2: SQL Path
- [ ] Run Ollama locally with Qwen2.5 7B
- [x] Text-to-SQL with schema injection
- [x] Guardrails (SELECT-only, LIMIT enforcement, dry-run validation, retry)
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

> Built a hybrid LLM agent (LangGraph, Ollama, BigQuery, Qdrant) that answers natural-language questions over 320K CFPB complaints, routing between text-to-SQL and RAG; achieved X% answer accuracy on a 25-question eval set.

---

## TODO: Remaining Work

Work through these roughly in order. Items 1–6 get the full system running end to end; the rest finish and polish it.

### 1. Housekeeping
- [ ] Delete the leftover Jupyter checkpoint folders. They're gitignored, but `data/raw/.ipynb_checkpoints/` holds 286 MB of duplicate CSVs:
      `find . -path ./.myvenv -prune -o -name .ipynb_checkpoints -type d -exec rm -rf {} +`
- [ ] Commit the restructure on `feature/repo-structure` (`git add -A` also records the removal of the old `banking-qa-agent/` folder), then open a PR to `main`.

### 2. Install the missing tools
- [ ] Docker Desktop, Ollama and the gcloud CLI (none are installed yet).
- [ ] `ollama pull qwen2.5:7b-instruct`

### 3. BigQuery (see [BigQuery Setup](#bigquery-setup))
- [ ] Create or choose the GCP project, enable billing (not the sandbox) and the BigQuery API.
- [ ] Set `BIGQUERY_PROJECT_ID` in `.env`.
- [ ] Run `python -m ingestion.load_bigquery` and check the row counts with the verification queries.
- [ ] Grant teammates Job User + Data Viewer; create a service account if you'll use Docker.

### 4. Qdrant / RAG
- [ ] `docker compose up -d qdrant`
- [ ] `python -m ingestion.embed_narratives --limit 2000`, test a few RAG questions, then run it on everything.
- [ ] Tune `RAG_SCORE_THRESHOLD`. Look at the scores the UI shows for good and bad matches: if too few narratives come back, lower it; if they're irrelevant, raise it.

### 5. First end-to-end run
- [ ] Start the API + UI, check that `/health` reports everything as connected, and try the example questions.
- [ ] `pytest -m integration`. The live router tests show whether the router prompt (`app/prompts/router.py`) needs tuning for your model.

### 6. Evaluation
- [ ] Improve `eval/questions.jsonl`. Several `expected_answer_contains` values are too generic to mean much (e.g. `["count", "number"]`). Replace them with real numbers or names from BigQuery (e.g. `"5571"`, `"Problem with a purchase shown on your statement"`).
- [ ] Add more hybrid questions (only 3 of 25 right now).
- [ ] Run `python eval/run_eval.py`, record the metrics in this README, and fill in the X% in the resume summary.

### 7. Feature gaps in the code
- [ ] **Use `company_aliases` in the SQL path.** It's loaded into BigQuery, but the SQL tool doesn't use it yet; the prompt relies on `UPPER(company) LIKE '%...%'` instead. Options: put the alias list in the SQL prompt, or teach the prompt to `JOIN` the alias table.
- [ ] **Hybrid filters:** `filters_from_sql_result` (`app/graph.py`) only picks up exact `company = '...'` conditions and the top result row. Queries that use `LIKE` produce no company filter.
- [ ] **`/health` for BigQuery** only checks that a project is configured. Add a real check, e.g. a dry-run `SELECT 1`.
- [ ] **`/feedback`** only logs. Store it somewhere (a BigQuery table or a local SQLite/CSV file).
- [ ] **Narratives in hybrid answers:** the SQL result rows and narratives go into one prompt. Check that long contexts don't overflow the model's context window (`MAX_SQL_ROWS_IN_CONTEXT` and `MAX_NARRATIVE_CHARS` in `app/llm.py`).
- [ ] **Streaming responses**, so the UI shows the answer as it's generated rather than after a long spinner.

### 8. Ops & quality
- [ ] Test `docker compose --profile app up --build` end to end. It's written but untested, because Docker wasn't installed.
- [ ] Add GitHub Actions CI running `ruff check .`, `ruff format --check .` and `pytest` on every PR.
- [ ] Re-run the notebooks so their saved outputs show the new paths (the outputs are still from before the restructure).
- [ ] Optionally make the notebooks import functions from `ingestion/clean.py` instead of duplicating the cleaning logic.
- [ ] Schedule `pull_cfpb → clean → load_bigquery → embed_narratives` to refresh the data (e.g. monthly).
