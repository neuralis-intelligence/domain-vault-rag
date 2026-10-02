# Complaint Insights Agent — Project Plan

Ask plain-English questions about CFPB bank complaints (credit card + mortgage) and get accurate answers from a local open-source LLM.

**Example:** "What is the most recurring issue at JPMorgan?" → *"Problem with a purchase shown on your statement" (5,571 complaints)*

---

## Key Idea

Two kinds of questions need two different tools:

| Question type | Example | Tool |
|---|---|---|
| Counting / ranking / trends | "Top issue at JPMorgan?" | **Text-to-SQL** on BigQuery |
| What customers say | "What do people complain about in Chase fraud cases?" | **RAG** on complaint narratives |
| Both | "Top issue at JPMorgan and what are people saying?" | SQL first, then RAG filtered by result |

An LLM **router** picks the path for each question.

```
User → Streamlit → FastAPI → Router (LangGraph)
                               ├── Text-to-SQL → BigQuery
                               └── Semantic search → Qdrant
                                        ↓
                              Answer synthesis (Ollama)
```

---

## Tech Stack

| Part | Tool |
|---|---|
| LLM | Ollama + Qwen2.5 7B Instruct or Llama 3.1 8B (backup: Groq free tier) |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector DB | Qdrant (Docker) |
| Data warehouse | BigQuery |
| Agent | LangGraph |
| API | FastAPI |
| UI | Streamlit |
| Packaging | Docker Compose |

---

## Repo Structure

```
complaint-insights-agent/
├── README.md
├── docker-compose.yml
├── .env.example
├── data/                    # raw CSVs (gitignored)
├── ingestion/
│   ├── clean.py
│   ├── load_bigquery.py
│   └── embed_narratives.py
├── app/
│   ├── main.py              # FastAPI POST /ask
│   ├── graph.py             # router → sql / rag / hybrid → answer
│   ├── llm.py
│   ├── sql_tool.py
│   ├── rag_tool.py
│   └── prompts/
├── ui/streamlit_app.py
├── eval/
│   ├── questions.jsonl
│   └── run_eval.py
└── tests/
```

---

## 5-Day Checklist

### Day 1 — Data
- [ ] Clean credit card + mortgage CSVs into one schema (snake_case, DATE types)
- [ ] Load into one BigQuery table `complaints` (partition by `date_received`, cluster by `company, product`)
- [ ] Create `company_aliases` table ("JP Morgan", "Chase" → `JPMORGAN CHASE & CO.`)
- [ ] Write ~25 test questions with correct answers (hand-written SQL)

### Day 2 — SQL Path
- [ ] Install Ollama, pull the model
- [ ] Write SQL prompt: schema + list of issues/products + 3–4 example Q→SQL pairs
- [ ] Add guardrails: SELECT only, force LIMIT, BigQuery dry run, retry once on error
- [ ] ✅ JPMorgan question answers correctly from a Python script

### Day 3 — RAG Path
- [ ] Embed narratives (use free Colab GPU, or a ~30K sample if short on time)
- [ ] Load into Qdrant with metadata: company, product, issue, state, date
- [ ] Build filtered search (metadata filter + semantic search)
- [ ] Answer prompt cites complaint IDs

### Day 4 — Agent + API
- [ ] LangGraph router: `sql` / `rag` / `hybrid`
- [ ] FastAPI `/ask` endpoint
- [ ] Docker Compose: ollama, qdrant, api, ui

### Day 5 — UI + Eval + Polish
- [ ] Streamlit chat; show a bar chart when SQL returns a table
- [ ] Run eval: routing accuracy, answer accuracy
- [ ] README: diagram, demo GIF, eval numbers

---

## If Time Runs Short
Cut RAG scope first (smaller sample). Keep the SQL path solid — it answers the main questions.

## Stretch Goals
- Langfuse tracing for LLM calls
- Cache repeated questions

---

## Resume Bullet (fill in your numbers)
> Built a hybrid LLM agent (LangGraph, Ollama, BigQuery, Qdrant) that answers natural-language questions over 250K+ CFPB complaints, routing between text-to-SQL and RAG; achieved X% answer accuracy on a 25-question eval set.
