# Complaint Insights Agent

A hybrid LLM agent that answers natural-language questions about CFPB bank complaints (credit cards and mortgages) using local open-source models. The agent intelligently routes queries between text-to-SQL and RAG approaches for accurate, grounded responses.

**Example:** "What is the most recurring issue at JPMorgan?" → *"Problem with a purchase shown on your statement" (5,571 complaints)*

---

## Goal

Enable non-technical users to query ~320K CFPB complaint records (Aug 2023 – Aug 2026) using plain English questions, with the system automatically determining the best retrieval strategy (SQL aggregation, semantic search, or hybrid) and generating accurate, cited answers using local LLMs.

---

## Architecture

Two kinds of questions require different tools:

| Question Type | Example | Tool |
|---|---|---|
| Counting / ranking / trends | "Top issue at JPMorgan?" | **Text-to-SQL** on BigQuery |
| What customers say | "What do people complain about in Chase fraud cases?" | **RAG** on complaint narratives |
| Hybrid | "Top issue at JPMorgan and what are people saying?" | SQL first, then RAG filtered by result |

An LLM **router** (LangGraph) picks the optimal path for each question.

```
User → Streamlit UI → FastAPI Backend → Router (LangGraph)
                                         ├── Text-to-SQL → BigQuery
                                         └── Semantic Search → Qdrant
                                                  ↓
                                        Answer Synthesis (Ollama)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Ollama + Qwen2.5 7B Instruct or Llama 3.1 8B (backup: Groq free tier) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| **Vector DB** | Qdrant (Docker) |
| **Data Warehouse** | BigQuery |
| **Agent Framework** | LangGraph |
| **API** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker Compose |

---

## Repository Structure

```
domain-vault-rag/
├── README.md
├── LICENSE
├── pyproject.toml              # pytest + ruff configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # + tests, linting, notebooks
├── Dockerfile                  # Image for the API and the UI
├── docker-compose.yml          # Qdrant (+ optional Ollama, API, UI)
├── .env.example                # Environment variable template
│
├── app/                        # FastAPI backend + agent
│   ├── config.py               # Paths and settings (loads .env)
│   ├── main.py                 # /ask, /health, /feedback endpoints
│   ├── graph.py                # LangGraph router → sql/rag/hybrid → synthesis
│   ├── router.py               # LLM query classification
│   ├── sql_tool.py             # BigQuery text-to-SQL with guardrails
│   ├── rag_tool.py             # Qdrant semantic search
│   ├── llm.py                  # Ollama client + answer synthesis
│   └── prompts/                # Prompt templates
│
├── ingestion/                  # ETL pipeline (run in this order)
│   ├── pull_cfpb.py            # 1. Download raw CSVs from the CFPB API
│   ├── clean.py                # 2. Clean + unify → data/processed/complaints.parquet
│   ├── load_bigquery.py        # 3. Load BigQuery table + company aliases
│   └── embed_narratives.py     # 4. Embed narratives into Qdrant
│
├── data/                       # Gitignored contents
│   ├── raw/                    # credit_card.csv, mortgage.csv
│   └── processed/              # complaints.parquet
│
├── notebooks/                  # Exploratory data analysis
│   ├── credit_card_exploration.ipynb
│   └── mortgage_exploration.ipynb
│
├── ui/
│   └── streamlit_app.py        # Chat interface + visualizations
│
├── eval/
│   ├── questions.jsonl         # 25 test questions with expected routes
│   └── run_eval.py             # Routing accuracy / answer quality / latency
│
└── tests/                      # Unit tests (offline) + integration tests
```

---

## Development Roadmap

### Phase 1: Data Engineering
- [x] Download CFPB credit card and mortgage complaint data (`ingestion/pull_cfpb.py`)
- [x] Exploratory data analysis (see `notebooks/`)
- [x] Clean and normalize to unified schema (`ingestion/clean.py`)
- [ ] Load into BigQuery with partitioning/clustering (`ingestion/load_bigquery.py` implemented, not yet run)
- [x] Create company alias lookup table (loaded by `load_bigquery.py`)

### Phase 2: SQL Path
- [ ] Deploy Ollama locally with Qwen2.5 7B
- [x] Implement text-to-SQL with schema injection
- [x] Add guardrails (SELECT-only, LIMIT enforcement, dry-run validation)
- [ ] Validate on test questions

### Phase 3: RAG Path
- [ ] Generate embeddings for complaint narratives (`ingestion/embed_narratives.py` implemented, not yet run)
- [ ] Load into Qdrant with metadata filters
- [x] Implement filtered semantic search
- [x] Build answer synthesis with citation

### Phase 4: Agent & API
- [x] Build LangGraph router (sql/rag/hybrid)
- [x] Implement FastAPI `/ask` endpoint
- [x] Create Docker Compose setup
- [ ] Deploy Qdrant and Ollama containers

### Phase 5: UI & Evaluation
- [x] Build Streamlit chat interface
- [x] Add visualization for SQL results
- [ ] Run evaluation on question set
- [ ] Document performance metrics

---

## Quick Start

### Prerequisites
- Python 3.11+ (developed on 3.13)
- [Ollama](https://ollama.com/download) installed natively (recommended on macOS for GPU)
- Docker (for Qdrant)
- A Google Cloud project with BigQuery enabled + [gcloud CLI](https://cloud.google.com/sdk/docs/install) (SQL path only)

### 1. Install
```bash
git clone git@github.com:neuralis-intelligence/domain-vault-rag.git
cd domain-vault-rag
python3 -m venv .myvenv
source .myvenv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then set BIGQUERY_PROJECT_ID
```

All commands below run from the project root with the venv active.

### 2. Check the install (no services needed)
```bash
pytest          # 55 offline unit tests
ruff check .    # lint
```

### 3. Start services
```bash
docker compose up -d qdrant                     # Qdrant on :6333 (dashboard: /dashboard)
ollama serve                                    # skip if the Ollama app is running
ollama pull qwen2.5:7b-instruct
gcloud auth application-default login           # BigQuery credentials
```

### 4. Run the data pipeline
```bash
python -m ingestion.pull_cfpb                   # optional: re-download data/raw/*.csv (slow)
python -m ingestion.clean                       # → data/processed/complaints.parquet
python -m ingestion.load_bigquery               # → BigQuery cfpb_complaints.complaints
python -m ingestion.embed_narratives --limit 2000   # quick test; omit --limit for all ~147K
```

### 5. Run the app
```bash
uvicorn app.main:app --reload                   # API: http://localhost:8000/docs
streamlit run ui/streamlit_app.py               # UI:  http://localhost:8501
```

Check `curl localhost:8000/health`: it reports the status of Ollama, Qdrant and BigQuery.

### Docker alternative
```bash
docker compose --profile app up -d --build      # Qdrant + API + UI (uses host Ollama)
```

---

## Usage

Once running, navigate to `http://localhost:8501` and ask questions like:

- "What are the top 5 most common issues at Bank of America?"
- "Show me complaints about fraudulent charges at Chase"
- "What percentage of Wells Fargo mortgage complaints are about foreclosure?"

The system will automatically route to the appropriate tool and synthesize an answer.

---

## Evaluation

With the API running:

```bash
python eval/run_eval.py                          # writes eval/eval_results.csv
```

This measures:
- **Routing accuracy**: Did the agent choose the right tool?
- **Answer quality**: Does the response contain the expected terms?
- **Latency** and **error rate**

## Testing

```bash
pytest                    # unit tests (offline, mocked LLM/Qdrant/BigQuery)
pytest -m integration     # live tests against Ollama / Qdrant
```

See [tests/README.md](tests/README.md) for details.

---

## Performance Considerations

- **SQL path**: Fast (<2s), handles aggregation queries
- **RAG path**: Moderate speed (~5s), depends on embedding quality
- **Hybrid**: Combines both, useful for complex multi-part questions
- **Caching**: Consider implementing for repeated queries (stretch goal)

---

## Stretch Goals

- [ ] Langfuse tracing for LLM observability
- [ ] Query result caching (Redis)
- [ ] Multi-turn conversations with context
- [ ] Support for additional CFPB product categories

---

## Contributing

This is a learning/portfolio project. Suggestions and improvements welcome via issues or PRs.

---

## License

See LICENSE file.

---

## Resume Summary

> Built a hybrid LLM agent (LangGraph, Ollama, BigQuery, Qdrant) that answers natural-language questions over 320K CFPB complaints, routing between text-to-SQL and RAG; achieved X% answer accuracy on a 25-question eval set.
