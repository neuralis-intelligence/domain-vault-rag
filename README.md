# Complaint Insights Agent

A hybrid LLM agent that answers natural-language questions about CFPB bank complaints (credit cards and mortgages) using local open-source models. The agent intelligently routes queries between text-to-SQL and RAG approaches for accurate, grounded responses.

**Example:** "What is the most recurring issue at JPMorgan?" → *"Problem with a purchase shown on your statement" (5,571 complaints)*

---

## Goal

Enable non-technical users to query 250K+ CFPB complaint records using plain English questions, with the system automatically determining the best retrieval strategy (SQL aggregation, semantic search, or hybrid) and generating accurate, cited answers using local LLMs.

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
├── README.md                   # This file
├── LICENSE
├── requirements.txt            # Root dependencies
├── docker-compose.yml          # Multi-container orchestration
├── .env.example                # Environment variable template
├── .gitignore
│
├── banking-qa-agent/           # Existing implementation (preserved)
│   ├── app.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── raw/                # Raw CFPB CSVs
│   │   ├── processed/          # Chunked narratives
│   │   ├── eval/               # Golden test sets
│   │   ├── pull_cfpb.py        # Data download script
│   │   ├── data-exploration-credit_card.ipynb
│   │   └── data-exploration-mortgage.ipynb
│   ├── src/
│   │   ├── ingest.py           # Embedding pipeline
│   │   ├── graph.py            # LangGraph workflow
│   │   ├── router.py           # Query classification
│   │   ├── retrieve.py         # Vector search
│   │   └── eval.py             # Evaluation logic
│   └── chroma_db/              # Local vector store
│
├── data/                       # Planned: raw CSVs (gitignored)
│
├── ingestion/                  # Planned: ETL pipeline
│   ├── clean.py                # Data cleaning & schema normalization
│   ├── load_bigquery.py        # BigQuery loader
│   └── embed_narratives.py     # Batch embedding generation
│
├── app/                        # Planned: FastAPI backend
│   ├── main.py                 # POST /ask endpoint
│   ├── graph.py                # LangGraph router → sql/rag/hybrid
│   ├── llm.py                  # Ollama client
│   ├── sql_tool.py             # BigQuery text-to-SQL
│   ├── rag_tool.py             # Qdrant semantic search
│   └── prompts/                # Prompt templates
│
├── ui/                         # Planned: Streamlit frontend
│   └── streamlit_app.py        # Chat interface + visualizations
│
├── eval/                       # Planned: evaluation suite
│   ├── questions.jsonl         # Test questions with ground truth
│   └── run_eval.py             # Automated accuracy measurement
│
└── tests/                      # Planned: unit & integration tests
```

---

## Development Roadmap

### Phase 1: Data Engineering
- [x] Download CFPB credit card and mortgage complaint data
- [x] Exploratory data analysis (see notebooks in `banking-qa-agent/data/`)
- [ ] Clean and normalize to unified schema (snake_case, proper types)
- [ ] Load into BigQuery with partitioning/clustering
- [ ] Create company alias lookup table

### Phase 2: SQL Path
- [ ] Deploy Ollama locally with Qwen2.5 7B
- [ ] Implement text-to-SQL with schema injection
- [ ] Add guardrails (SELECT-only, LIMIT enforcement, dry-run validation)
- [ ] Validate on test questions

### Phase 3: RAG Path
- [ ] Generate embeddings for complaint narratives
- [ ] Load into Qdrant with metadata filters
- [ ] Implement filtered semantic search
- [ ] Build answer synthesis with citation

### Phase 4: Agent & API
- [ ] Build LangGraph router (sql/rag/hybrid)
- [ ] Implement FastAPI `/ask` endpoint
- [ ] Create Docker Compose setup
- [ ] Deploy Qdrant and Ollama containers

### Phase 5: UI & Evaluation
- [ ] Build Streamlit chat interface
- [ ] Add visualization for SQL results
- [ ] Run evaluation on question set
- [ ] Document performance metrics

---

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Ollama installed locally
- BigQuery project (for SQL path)

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd domain-vault-rag
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your BigQuery credentials and other settings
```

3. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. Start services:
```bash
docker-compose up -d
```

5. Pull LLM model:
```bash
ollama pull qwen2.5:7b-instruct
```

6. Run data ingestion (once data is prepared):
```bash
python ingestion/clean.py
python ingestion/load_bigquery.py
python ingestion/embed_narratives.py
```

7. Start the API:
```bash
uvicorn app.main:app --reload
```

8. Launch the UI:
```bash
streamlit run ui/streamlit_app.py
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

Run the evaluation suite to measure performance:

```bash
python eval/run_eval.py
```

This measures:
- **Routing accuracy**: Did the agent choose the right tool?
- **Answer accuracy**: Does the response match ground truth?

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

> Built a hybrid LLM agent (LangGraph, Ollama, BigQuery, Qdrant) that answers natural-language questions over 250K+ CFPB complaints, routing between text-to-SQL and RAG; achieved X% answer accuracy on a 25-question eval set.
