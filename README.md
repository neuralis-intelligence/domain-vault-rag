# domain-vault-rag — End-to-End Build Plan

> **Project goal:** Turn `domain-vault-rag` from a portfolio RAG prototype into a production-style, LLM-powered financial intelligence system that demonstrates the skills employers are currently asking for: Python, LLM applications, RAG/retrieval, agents/tool calling, SQL/data engineering, evaluation, observability, APIs, Docker/CI/CD, cloud deployment, security/guardrails, and measurable system quality.

**Current date:** October 2026  
**Primary data:** CFPB Consumer Complaint Database / the `credit_card.csv` dataset already in this project  
**Initial warehouse:** BigQuery  
**Repository name:** `domain-vault-rag`

---

## 0. Executive decision: what we are building

We are **not** going to throw away the current credit-card complaint data.

We will build the project in layers:

```text
LEVEL 0
Data foundation
    ↓
LEVEL 1
Natural language → SQL analytics
    ↓
LEVEL 2
Semantic complaint search
    ↓
LEVEL 3
Hybrid SQL + semantic RAG
    ↓
LEVEL 4
Tool-using LLM agent
    ↓
LEVEL 5
Financial research / investigation workflow
    ↓
LEVEL 6
Evaluation + observability + guardrails + production deployment
    ↓
LEVEL 7
Model comparison + optional QLoRA fine-tuning
    ↓
LEVEL 8
Second domain / domain isolation / multi-source financial knowledge
```

The important principle is:

> **Every level must produce a working artifact before we move to the next level.**

Do not start with multi-agent orchestration, fine-tuning, or a vector database.

---

# 1. Why this project is worth building

Current U.S. AI Engineer postings strongly emphasize Python, LLMs, RAG, ML, cloud, APIs, CI/CD, observability, Docker, SQL, vector databases, evaluation, and agent frameworks. A September 2026 analysis of 1,056 live U.S. AI Engineer postings found Python in 68.4%, LLM in 52.8%, RAG in 34.7%, AWS in 27.2%, CI/CD in 16.7%, REST APIs in 16.7%, observability in 13.3%, vector databases in 12.7%, Docker in 11.5%, SQL in 10.8%, fine-tuning in 10.2%, and LangGraph in 9.3% of the analyzed postings. Only 3.5% of those postings were classified as open to entry-level candidates, so the portfolio needs to demonstrate practical depth rather than just familiarity with tools.

Source: September 2026 U.S. AI Engineer job-market analysis.

The current project architecture already contains several strong ideas: routing, domain isolation, retrieval, reasoning, an evaluator gate, evaluation metrics, observability, MLflow, DVC, FastAPI, Docker, and CI/CD. The original architecture is documented in the project's supplied README.

We will keep those ideas, but **reorder the implementation around the data we actually have**.

---

# 2. The central product idea

The project should eventually be described as:

> **Domain Vault RAG — an LLM-powered financial intelligence system for natural-language analytics, semantic complaint search, evidence-grounded reasoning, and safe tool-using investigation over consumer financial data.**

A user should eventually be able to ask questions such as:

### Analytical

- Which bank has had the most credit-card complaints related to timeouts?
- How many complaints did each company receive about payment processing problems in 2025?
- Which states had the highest complaint volume for credit cards?
- What percentage of complaints received a timely company response?
- How did complaint volume change year over year?

### Semantic

- Find complaints where consumers describe being charged after a payment was supposedly reversed.
- Show examples of complaints that sound like unauthorized recurring charges.
- What kinds of problems are consumers describing when they say a payment was not recognized?

### Hybrid

- Which banks have the highest number of complaints about payment processing, and what are consumers actually saying?
- For the three companies with the most complaints about late fees, summarize the common failure patterns from the narratives.
- A company has unusually high complaints about a specific issue. Show the count and provide representative evidence from the narratives.

### Research / investigation

- Investigate whether complaint volume about payment problems increased for Company X and explain the main themes.
- Compare Company A and Company B on complaint volume and narrative themes.
- Find quantitative evidence first, then retrieve supporting complaint narratives, then produce a cited explanation.

The system should know when to use:

```text
SQL
semantic retrieval
SQL + retrieval
or abstention
```

That routing decision is one of the most important parts of the project.

---

# 3. Data decision: do NOT hunt for datasets yet

## 3.1 The current dataset is sufficient for Levels 0–7

The existing credit-card complaint data is already large enough to demonstrate:

- data cleaning
- schema design
- BigQuery
- SQL analytics
- natural-language-to-SQL
- semantic search
- embeddings
- RAG
- hybrid retrieval
- LLM reasoning
- tool calling
- evaluation
- observability
- guardrails
- production API design
- model comparison
- optional fine-tuning

The CFPB itself publishes the Consumer Complaint Database and makes the data freely available for analysis and tool building. The database is generally updated daily. The CFPB also explicitly warns that complaint counts are not a statistical sample of consumer experiences and should be interpreted with company size/market share and other context.

Source: CFPB Consumer Complaint Database.

## 3.2 Why the existing data is especially useful

The dataset contains both:

**Structured information**

- date
- product
- sub-product
- issue
- sub-issue
- company
- state
- response
- timely response
- submission method
- tags
- identifiers

and:

**Unstructured information**

- consumer complaint narratives

That combination is ideal for demonstrating an LLM system because the same user question can require either structured analytics or language understanding.

For example:

> "Which bank has had the most issues with timeouts in credit-card complaints?"

requires structured aggregation if "timeouts" maps cleanly to an issue/category.

But:

> "What are consumers describing when they say the bank's system timed out?"

requires semantic retrieval over narratives.

And:

> "Which bank has the most timeout complaints and what are customers saying about them?"

requires both.

This gives the project a real reason to combine SQL and RAG instead of adding RAG just because it is fashionable.

---

# 4. Dataset expansion rule

We will **not add another dataset just to make the project look bigger**.

Before adding any source, run this checklist:

1. Does it answer a question the current CFPB complaint data cannot answer?
2. Is it authoritative or clearly reputable?
3. Is the license/usage policy clear?
4. Is the schema compatible with our current system?
5. Can it support a concrete product feature?
6. Can we evaluate questions over it?
7. Can we ingest it without weeks of preprocessing?
8. Does it strengthen the financial-domain story?

If the answer is not clearly yes, **do not add it**.

---

# 5. First external source to add later: CFPB public material

The first expansion should be from the **same organization**, not a random dataset.

The CFPB public-data inventory includes additional public financial datasets and documents, and the CFPB complaint program provides official material describing complaint processes and financial products.

This is preferable to immediately adding an unrelated Kaggle dataset because:

- same institution
- same financial domain
- same credibility level
- easier provenance story
- easier citation story
- fewer data-model surprises
- stronger "financial intelligence" narrative

Potential later sources:

- CFPB complaint documentation
- CFPB consumer-finance educational material
- CFPB public reports
- CFPB enforcement/public research documents
- other official financial-regulator documents where licensing/access is clear

Do not ingest these yet.

First finish the complaint-data system.

---

# 6. BigQuery decision

## We keep BigQuery.

Do **not** move the complaint data into PostgreSQL just because the original README proposed PostgreSQL + pgvector.

The current system benefits from separating:

```text
BigQuery
    = analytical source of truth

Vector/search layer
    = semantic retrieval
```

However, there is an important simplification:

### We do not need a separate vector database immediately.

BigQuery now supports embeddings and vector search, including semantic and hybrid search through `VECTOR_SEARCH`, vector indexes, and lexical search.

Therefore the first implementation should benchmark:

```text
BigQuery SQL
+
BigQuery vector search
```

before introducing Qdrant or pgvector.

This makes the architecture cheaper and simpler.

Later, we can benchmark:

```text
BigQuery vector search
vs
Qdrant
vs
pgvector
```

as an engineering experiment.

That benchmark itself becomes portfolio material.

---

# 7. Target architecture

## Phase A — first working system

```text
                         ┌───────────────────────┐
                         │      User Question    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Query Understanding   │
                         │ intent + entities     │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
              ┌──────────┐    ┌──────────────┐   ┌─────────────┐
              │ SQL path │    │ Search path  │   │ Hybrid path │
              └────┬─────┘    └──────┬───────┘   └──────┬──────┘
                   │                 │                  │
                   ▼                 ▼                  ▼
              BigQuery SQL     Semantic search    SQL + retrieval
                   │                 │                  │
                   └─────────────────┼──────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Evidence Builder      │
                         │ results + citations   │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Open-weight LLM       │
                         │ reasoning + response  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Evaluator / Guardrail │
                         └───────────┬───────────┘
                                     │
                           ┌─────────┴─────────┐
                           ▼                   ▼
                        PASS              ABSTAIN/ESCALATE
                           │
                           ▼
                    Answer + evidence
```

## Phase B — later

Add:

```text
planner
tool-calling
memory
multi-step investigation
domain router
document sources
human review
feedback
model routing
observability
CI/CD
```

Do not build all of this at once.

---

# 8. Repository structure

Change the current structure gradually into:

```text
domain-vault-rag/
│
├── README.md
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── docker-compose.yml
│
├── src/
│   └── domain_vault/
│       ├── config/
│       ├── data/
│       │   ├── ingestion/
│       │   ├── validation/
│       │   └── profiling/
│       │
│       ├── sql/
│       │   ├── schema.py
│       │   ├── generator.py
│       │   ├── validator.py
│       │   └── executor.py
│       │
│       ├── retrieval/
│       │   ├── embeddings.py
│       │   ├── semantic.py
│       │   ├── lexical.py
│       │   ├── hybrid.py
│       │   └── reranker.py
│       │
│       ├── agents/
│       │   ├── planner.py
│       │   ├── tools.py
│       │   ├── investigator.py
│       │   └── graph.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   ├── prompts/
│       │   ├── structured_output.py
│       │   └── model_router.py
│       │
│       ├── guardrails/
│       │   ├── scope.py
│       │   ├── prompt_injection.py
│       │   ├── pii.py
│       │   └── abstention.py
│       │
│       ├── evidence/
│       │   ├── builder.py
│       │   └── citations.py
│       │
│       ├── evaluation/
│       │   ├── datasets/
│       │   ├── metrics/
│       │   ├── runners/
│       │   └── reports/
│       │
│       ├── observability/
│       │   ├── tracing.py
│       │   ├── metrics.py
│       │   └── cost.py
│       │
│       └── api/
│           └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── schemas/
│   └── synthetic/
│
├── eval/
│   ├── golden_questions.jsonl
│   ├── sql_cases.jsonl
│   ├── retrieval_cases.jsonl
│   ├── safety_cases.jsonl
│   └── regression/
│
├── notebooks/
│   ├── 01_data_profile.ipynb
│   ├── 02_issue_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── sql/
│   ├── staging/
│   ├── marts/
│   └── examples/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/
│
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── data-dictionary.md
│   ├── setup.md
│   ├── evaluation.md
│   ├── security.md
│   ├── model-card.md
│   └── adr/
│
├── docker/
│
└── .github/
    └── workflows/
        ├── tests.yml
        ├── evaluation.yml
        └── build.yml
```

---

# 9. Phase 0 — Freeze the scope

## Goal

Prevent scope explosion.

## Do this first

Create:

```text
docs/project-scope.md
```

Write down:

### In scope

- CFPB credit-card complaints
- structured analytics
- complaint narrative search
- RAG
- natural-language-to-SQL
- hybrid SQL + retrieval
- LLM reasoning
- evaluation
- observability
- guardrails
- API
- Docker
- CI/CD
- optional fine-tuning
- optional second domain later

### Out of scope initially

- real customer data
- banking transactions
- live bank systems
- financial advice
- regulatory compliance certification
- autonomous financial decisions
- production customer-facing deployment
- huge multi-agent swarm
- random public datasets

## Definition of done

You have a one-page scope document and every future feature can be classified as:

```text
NOW
LATER
NOT THIS PROJECT
```

---

# 10. Phase 1 — Data foundation

## Goal

Make the current dataset trustworthy before using an LLM.

## Step 1. Inspect the CSV

Profile:

- row count
- column count
- data types
- null percentages
- unique values
- duplicate rows
- duplicate complaint IDs
- date range
- product distribution
- issue distribution
- company distribution
- state distribution
- narrative availability
- narrative length
- response distribution
- timely-response distribution

Create:

```text
notebooks/01_data_profile.ipynb
```

## Step 2. Create a formal schema

Create:

```text
docs/data-dictionary.md
data/schemas/complaints.yml
```

For every field document:

```text
field
type
nullable
meaning
example
allowed values
source
```

Do not rely on memory for the meaning of fields.

Use the CFPB field reference as the authoritative source for CFPB fields.

## Step 3. Clean the data

Create:

```text
src/domain_vault/data/validation/
src/domain_vault/data/ingestion/
```

Cleaning rules should include:

- normalize column names
- parse dates
- normalize null values
- trim whitespace
- preserve original text
- preserve original complaint ID
- remove exact duplicates
- validate categorical values
- flag malformed records
- calculate narrative length
- create ingestion timestamp
- create source metadata

Never silently delete suspicious rows.

Instead:

```text
valid records
quarantine records
```

## Step 4. Load BigQuery

Recommended tables:

```text
raw_complaints
stg_complaints
complaints
complaint_narratives
```

Keep the raw layer immutable.

## Step 5. Partition and cluster

For the main complaint table, evaluate:

```text
PARTITION BY date_received
```

and clustering around fields frequently used in queries, such as:

```text
product
issue
company
state
```

Benchmark query cost.

## Definition of done

You can run:

```sql
SELECT COUNT(*) FROM complaints;
```

and get the expected record count.

You can also reproduce:

- top companies
- top issues
- complaint counts by year
- complaint counts by product

from SQL.

---

# 11. Phase 2 — Build the analytical intelligence layer

## Goal

Answer quantitative questions without using the LLM to hallucinate arithmetic.

This is critical.

The LLM should **generate or select SQL**, not invent the numerical answer.

## Step 1. Create reusable SQL views

Examples:

```text
company_complaint_counts
issue_complaint_counts
company_issue_counts
monthly_complaint_counts
state_complaint_counts
response_metrics
```

## Step 2. Write canonical SQL queries

Start with 20–30.

Examples:

```text
top companies by complaint count
top issues
top sub-issues
complaints by year
complaints by month
complaints by state
company + issue combinations
timely response rate
narrative availability
```

## Step 3. Create a semantic data dictionary

Create:

```text
docs/data-dictionary.md
```

Add business-language mappings:

```text
"bank"
→ company

"complaints"
→ COUNT(DISTINCT complaint_id)

"most issues"
→ ORDER BY COUNT(*) DESC

"last year"
→ date filter

"credit card"
→ product filter

"timeout"
→ issue/sub-issue/narrative search depending on context
```

## Step 4. Implement a deterministic SQL baseline

Before using an LLM, create:

```text
question → intent → SQL template → result
```

This establishes a baseline.

## Definition of done

At least 20 analytical questions can be answered correctly without an LLM generating arbitrary SQL.

---

# 12. Phase 3 — Build the first LLM component: NL → SQL

## Goal

Introduce an LLM in a controlled way.

The LLM receives:

```text
question
+
schema
+
data dictionary
+
allowed SQL rules
```

It returns structured output:

```json
{
  "intent": "analytical",
  "sql": "...",
  "tables": ["complaints"],
  "filters": [],
  "confidence": 0.92
}
```

Do not allow free-form output.

## SQL safety rules

The generated SQL must:

- be read-only
- use approved datasets/tables
- reject DDL
- reject DML
- reject scripting
- enforce query limits where appropriate
- validate referenced columns
- validate referenced tables
- use parameterization where possible
- log generated SQL
- log execution time
- log bytes processed

## SQL validator

Build:

```text
src/domain_vault/sql/validator.py
```

Pipeline:

```text
LLM
 ↓
structured output validation
 ↓
SQL parser/validator
 ↓
allowlist check
 ↓
dry run / validation
 ↓
execute
 ↓
result
```

## Definition of done

The system can answer:

> Which bank has had the most credit-card complaints?

and return:

```text
answer
SQL used
row(s) supporting answer
execution metadata
```

---

# 13. Phase 4 — Build the evaluation dataset BEFORE complex RAG

This is one of the most important steps.

Create:

```text
eval/golden_questions.jsonl
```

Target:

```text
100 questions minimum
200–300 questions ideal
```

Do not create all questions manually.

Generate candidate questions from the schema and known analytical patterns, then manually verify the expected answer/SQL.

## Categories

### Category A — aggregation

- count
- sum
- average
- max
- min
- ranking

### Category B — filtering

- company
- state
- date
- issue
- sub-issue

### Category C — time

- yearly
- monthly
- trends
- before/after

### Category D — comparison

- company A vs B
- year A vs year B
- state A vs state B

### Category E — ambiguity

- "timeouts"
- "payment problems"
- "bank"
- "complaints about fees"

### Category F — impossible questions

Questions the system should reject or clarify.

### Category G — adversarial

Prompt injection attempts.

## Every evaluation case should include

```json
{
  "id": "sql_001",
  "question": "...",
  "expected_intent": "analytical",
  "expected_sql": "...",
  "expected_answer": "...",
  "acceptable_answer_variants": [],
  "difficulty": "medium"
}
```

## Definition of done

You have a reproducible evaluation set before optimizing the model.

---

# 14. Phase 5 — Choose the first open-weight LLM

Do not fine-tune yet.

The first objective is to build the application around a strong baseline model.

Evaluate 2–3 current open-weight instruction models that you can realistically run through:

```text
vLLM
or
Ollama
```

Do not select a model solely because it is popular.

Benchmark:

- SQL generation
- structured output reliability
- reasoning
- tool calling
- latency
- VRAM requirement
- context length
- license
- local deployment complexity

Record the decision in:

```text
docs/adr/0001-model-selection.md
```

## Required experiment

For each candidate:

```text
100 golden questions
→ run model
→ execute SQL
→ compare result to expected
```

Measure:

```text
SQL validity
SQL execution success
answer correctness
latency
tokens
cost
```

## Definition of done

You have a model-selection table and a documented baseline.

---

# 15. Phase 6 — Build semantic complaint search

Now we introduce embeddings.

## Goal

Search complaint narratives by meaning.

Example:

> "Find complaints where consumers describe being charged after a payment was supposedly reversed."

This is not a simple SQL aggregation.

## Step 1. Create narrative records

Create a retrieval table with:

```text
complaint_id
date_received
company
product
issue
sub_issue
state
narrative
source
metadata
embedding
```

## Step 2. Generate embeddings

First benchmark:

```text
BigQuery-native embeddings/vector search
```

because BigQuery already supports semantic and hybrid search.

Only add Qdrant/pgvector if the benchmark shows a reason.

## Step 3. Build retrieval API

Create:

```text
src/domain_vault/retrieval/semantic.py
```

Input:

```text
query
filters
top_k
```

Output:

```json
{
  "results": [
    {
      "complaint_id": "...",
      "score": 0.82,
      "narrative": "...",
      "metadata": {}
    }
  ]
}
```

## Definition of done

A user can enter a semantic question and retrieve relevant complaints with metadata.

---

# 16. Phase 7 — Evaluate retrieval

Do not declare RAG successful because the returned text "looks good."

Create retrieval test cases:

```text
eval/retrieval_cases.jsonl
```

Each case:

```json
{
  "id": "ret_001",
  "query": "...",
  "relevant_complaint_ids": ["...", "..."],
  "top_k": 10
}
```

Measure:

- Recall@K
- Precision@K
- MRR
- nDCG
- semantic relevance
- metadata filter correctness

## Create three retrieval baselines

### Baseline A

Keyword search.

### Baseline B

Dense vector search.

### Baseline C

Hybrid search.

Then compare.

This is much more valuable than saying:

> "I built a vector database."

---

# 17. Phase 8 — Add reranking

If hybrid retrieval produces noisy results:

```text
query
 ↓
candidate retrieval
 ↓
top 20–50 candidates
 ↓
reranker
 ↓
top 5–10
```

Benchmark whether reranking actually improves retrieval metrics.

Do not add a reranker if it does not measurably help.

Document the experiment.

---

# 18. Phase 9 — Build RAG answer generation

Now connect:

```text
query
→ retrieval
→ evidence
→ LLM
→ grounded answer
```

The LLM must receive:

```text
USER QUESTION

EVIDENCE
[1] complaint ...
[2] complaint ...
[3] complaint ...

INSTRUCTIONS
Answer only using evidence.
If evidence is insufficient, say so.
Do not invent facts.
```

## Output format

Use structured output:

```json
{
  "answer": "...",
  "citations": [
    {
      "complaint_id": "...",
      "reason": "..."
    }
  ],
  "confidence": 0.84,
  "abstained": false
}
```

## Definition of done

The system answers semantic questions and identifies the complaint records supporting the answer.

---

# 19. Phase 10 — The key project feature: Hybrid SQL + RAG

This is where the project becomes significantly more interesting.

Example:

> Which companies have the most complaints about payment problems, and what are consumers saying?

The system should execute:

```text
Step 1:
SQL identifies top companies.

Step 2:
Retriever searches narratives for those companies.

Step 3:
Evidence builder combines counts + representative narratives.

Step 4:
LLM explains the result.

Step 5:
Evaluator verifies that the explanation matches the evidence.
```

Architecture:

```text
                    USER
                     │
                     ▼
                PLANNER LLM
                     │
             ┌───────┴────────┐
             ▼                ▼
        SQL TOOL         SEARCH TOOL
             │                │
             └───────┬────────┘
                     ▼
              EVIDENCE OBJECT
                     │
                     ▼
                ANSWER LLM
                     │
                     ▼
                EVALUATOR
```

This should become the core of `domain-vault-rag`.

---

# 20. Phase 11 — Build the tool-using agent

Only now should we introduce LangGraph.

Do not build a swarm.

Start with one graph:

```text
START
 ↓
query_understanding
 ↓
planner
 ↓
tool_selection
 ↓
execute_tool
 ↓
evidence_builder
 ↓
answer
 ↓
evaluate
 ↓
END
```

Tools:

```text
run_sql
semantic_search
hybrid_search
get_complaint
get_company_summary
get_issue_summary
```

The LLM does not directly access databases.

It requests tools.

---

# 21. Phase 12 — Add investigation mode

Create two modes.

## Mode 1 — Quick answer

For simple questions:

```text
question
→ one tool
→ answer
```

## Mode 2 — Investigation

For complex questions:

```text
question
→ plan
→ SQL
→ search
→ compare
→ retrieve evidence
→ synthesize
→ evaluate
```

Example:

> Investigate whether Company X had an unusual increase in complaints about payment processing and summarize the main narrative themes.

Possible plan:

```text
1. Determine monthly complaint volume.
2. Identify relevant issue/sub-issue categories.
3. Detect periods of unusual change.
4. Retrieve representative narratives.
5. Cluster/theme the narratives.
6. Generate explanation.
7. Cite evidence.
```

This is the beginning of the "financial intelligence agent" story.

---

# 22. Phase 13 — Add guardrails

The original README correctly identifies prompt injection as a planned limitation.

Build it now.

## Guardrail categories

### 1. Scope guardrail

Reject:

```text
"Write me a poem."
```

if the application is in financial analytics mode.

### 2. SQL safety

Never allow:

```text
DROP
DELETE
UPDATE
INSERT
ALTER
CREATE
```

through the analytical tool.

### 3. Prompt injection

Treat complaint narratives as **untrusted data**.

A complaint may contain:

```text
Ignore previous instructions...
```

The model must treat this as complaint content, not instructions.

### 4. Data leakage

Prevent retrieved evidence from one future domain from appearing in another domain.

### 5. PII protection

Even though CFPB publishes complaint data without direct identifiers, implement a detection layer for sensitive strings.

### 6. Abstention

If evidence is insufficient:

```text
I don't have enough evidence to answer this reliably.
```

This is preferable to hallucination.

---

# 23. Phase 14 — Build the evaluator

The evaluator should inspect:

```text
question
plan
tool calls
SQL
retrieved evidence
draft answer
citations
```

Evaluate:

### Analytical

- SQL validity
- SQL execution
- numerical correctness
- filter correctness
- aggregation correctness

### Retrieval

- retrieval recall
- precision
- ranking quality

### Generation

- groundedness
- answer relevance
- citation accuracy
- unsupported claims

### Agent

- correct tool selection
- unnecessary tool calls
- workflow completion
- failure recovery

### Safety

- prompt injection
- scope violation
- data leakage
- unsafe SQL

### Production

- latency
- token usage
- cost
- error rate

---

# 24. Phase 15 — Observability

Pick one tracing platform first.

Good candidates:

```text
Langfuse
or
Arize Phoenix
```

Do not implement both initially.

Trace:

```text
request
 ├── planner
 ├── SQL generation
 ├── SQL execution
 ├── retrieval
 ├── reranking
 ├── LLM generation
 └── evaluator
```

Record:

```text
latency
tokens
model
prompt version
retrieval parameters
SQL
tool calls
evaluation score
failure type
```

Create a dashboard showing:

```text
requests
success rate
answer quality
retrieval quality
p50 latency
p95 latency
token usage
estimated cost
guardrail failures
abstentions
```

Current AI-engineering postings increasingly emphasize evaluation and observability rather than merely building a chatbot.

---

# 25. Phase 16 — Build FastAPI

Create:

```text
POST /query
POST /search
POST /analytics
POST /investigate
GET  /health
GET  /version
```

Example request:

```json
{
  "question": "Which bank had the most complaints about payment processing?",
  "mode": "auto"
}
```

Example response:

```json
{
  "answer": "...",
  "mode": "analytical",
  "evidence": [],
  "citations": [],
  "metrics": {
    "latency_ms": 1234
  }
}
```

The API should expose the system, not the internal implementation details.

---

# 26. Phase 17 — Build the UI

Use Streamlit initially.

The UI should show:

```text
Question
↓
Detected intent
↓
Plan
↓
Tools used
↓
SQL / retrieval evidence
↓
Answer
↓
Citations
↓
Evaluation result
```

Add a toggle:

```text
Simple answer
Investigation mode
```

Do not hide the evidence.

The portfolio demo should make the engineering visible.

---

# 27. Phase 18 — Dockerize

Create:

```text
docker/Dockerfile.api
docker/Dockerfile.worker
docker-compose.yml
```

At minimum:

```text
API
LLM service
observability service
```

BigQuery remains external.

## Definition of done

A new machine can run the application using documented setup commands.

---

# 28. Phase 19 — CI/CD

GitHub Actions should run:

```text
lint
 ↓
unit tests
 ↓
integration tests
 ↓
SQL validation tests
 ↓
security tests
 ↓
evaluation regression
 ↓
Docker build
```

The evaluation gate should fail if:

```text
SQL accuracy drops
groundedness drops
retrieval recall drops
safety score drops
```

Do not make arbitrary thresholds immediately.

First establish a baseline, then choose thresholds from observed performance.

---

# 29. Phase 20 — Fine-tuning

Only now.

Fine-tuning is optional.

The project should be impressive **without** fine-tuning.

Fine-tune only if error analysis shows a repeatable problem such as:

- poor SQL formatting
- poor structured outputs
- poor domain terminology
- weak tool selection
- poor classification
- poor refusal behavior

## Candidate approach

```text
LoRA / QLoRA
+
Hugging Face PEFT
+
TRL
+
Unsloth if appropriate
```

## Dataset

Use your own evaluated examples:

```text
question
schema
expected plan
expected tool
expected SQL
expected answer
```

Do not simply fine-tune on raw complaint narratives.

## Experiment design

Compare:

```text
base model
vs
fine-tuned model
```

on the **same held-out evaluation set**.

Measure:

```text
SQL accuracy
tool-selection accuracy
answer correctness
groundedness
latency
token usage
```

If fine-tuning does not improve the system, document that result.

A negative result is useful if the experiment is rigorous.

---

# 30. Phase 21 — Model comparison

Add a second model.

Create:

```text
docs/model-benchmark.md
```

Compare:

```text
Model A
Model B
Fine-tuned A
```

Dimensions:

| Dimension | Measurement |
|---|---|
| SQL correctness | % correct |
| Tool selection | % correct |
| Answer correctness | % correct |
| Groundedness | evaluator score |
| Retrieval | Recall@K |
| Latency | p50/p95 |
| Cost | $/query |
| Context | supported length |
| Deployment | VRAM/CPU |
| License | documented |
| Reliability | failure rate |

This turns the project into an engineering experiment rather than a demo.

---

# 31. Phase 22 — Add a second knowledge domain

Only after the first system is strong.

The second domain should preferably come from official financial/regulatory sources.

Recommended progression:

```text
CFPB complaints
      ↓
CFPB official documents/reports
      ↓
other authoritative financial documents
```

Do not immediately add health or auto insurance just because the original README used those examples.

The current dataset gives us a stronger story if we first become excellent at **consumer financial intelligence**.

---

# 32. Phase 23 — Reintroduce domain isolation

Once multiple sources exist, implement:

```text
financial_complaints
regulatory_documents
product_documents
```

The router decides which sources may be used.

For example:

```text
Question
 ↓
scope/domain classifier
 ↓
allowed source set
 ↓
retrieval
```

The key security property:

> The router defines the allowed evidence space before retrieval occurs.

Do not rely only on a prompt saying:

> "Don't use the other domain."

Enforce it in code.

---

# 33. Phase 24 — Knowledge graph / GraphRAG: optional advanced level

Only do this if a concrete query requires relationships that are awkward in SQL/vector search.

Potential entities:

```text
Company
Product
Issue
SubIssue
State
Date
RegulatoryDocument
Complaint
```

Potential relationships:

```text
Complaint → ABOUT → Issue
Complaint → AGAINST → Company
Complaint → LOCATED_IN → State
Document → DISCUSSES → Issue
Company → MENTIONED_IN → Document
```

Then compare:

```text
SQL
vs
vector search
vs
graph retrieval
vs
hybrid
```

This is an advanced extension, not a starting requirement.

---

# 34. Exact milestone sequence

## Milestone 1 — Data foundation

Deliver:

- cleaned CFPB complaint table
- BigQuery dataset
- data dictionary
- profiling notebook
- validation tests

**Stop here until reliable.**

---

## Milestone 2 — Analytics engine

Deliver:

- canonical SQL queries
- SQL views
- analytical API
- 20+ verified questions

---

## Milestone 3 — LLM SQL

Deliver:

- open-weight model
- structured SQL generation
- SQL validator
- execution
- 100-question evaluation set

---

## Milestone 4 — Semantic search

Deliver:

- embeddings
- vector search
- metadata filtering
- retrieval evaluation

---

## Milestone 5 — RAG

Deliver:

- evidence builder
- grounded answer generation
- citations
- RAG evaluation

---

## Milestone 6 — Hybrid intelligence

Deliver:

- SQL + retrieval
- multi-step reasoning
- representative evidence
- complex questions

---

## Milestone 7 — Agent

Deliver:

- LangGraph
- planner
- tools
- investigation mode
- failure recovery

---

## Milestone 8 — Safety + evaluation

Deliver:

- scope guardrail
- prompt-injection tests
- SQL security
- abstention
- evaluator
- regression suite

---

## Milestone 9 — Production engineering

Deliver:

- FastAPI
- Streamlit
- Docker
- observability
- CI/CD

---

## Milestone 10 — Model experiments

Deliver:

- model benchmark
- optional QLoRA
- before/after evaluation
- experiment report

---

## Milestone 11 — Domain expansion

Deliver:

- authoritative CFPB documents
- multi-source retrieval
- source-level isolation

---

## Milestone 12 — Advanced research

Optional:

- Qdrant vs BigQuery vector search
- pgvector benchmark
- reranking experiments
- GraphRAG
- model routing
- caching
- semantic query planning
- MCP/tool protocol support

---

# 35. Suggested 12-week execution schedule

## Week 1 — Data

- profile CSV
- inspect schema
- clean data
- load BigQuery
- document provenance
- build first SQL queries

**Deliverable:** trustworthy dataset.

## Week 2 — Analytics

- create marts/views
- canonical queries
- data dictionary
- analytical question set
- SQL tests

**Deliverable:** deterministic analytics engine.

## Week 3 — LLM → SQL

- choose baseline open model
- structured outputs
- SQL generator
- SQL validator
- execution pipeline

**Deliverable:** working natural-language analytics.

## Week 4 — Evaluation

- 100 golden questions
- automatic evaluation
- model benchmark
- failure analysis

**Deliverable:** measurable baseline.

## Week 5 — Embeddings

- create narrative retrieval table
- generate embeddings
- semantic search
- metadata filters

**Deliverable:** semantic complaint search.

## Week 6 — Retrieval quality

- keyword baseline
- dense baseline
- hybrid retrieval
- Recall@K / MRR / nDCG
- reranking experiment

**Deliverable:** measured retrieval system.

## Week 7 — RAG

- evidence builder
- grounded generation
- citations
- abstention

**Deliverable:** citation-grounded complaint assistant.

## Week 8 — Hybrid agent

- SQL tool
- search tool
- planner
- LangGraph
- investigation mode

**Deliverable:** financial intelligence agent.

## Week 9 — Safety

- SQL guardrails
- prompt injection defense
- scope control
- leakage tests
- safety evaluation

**Deliverable:** safe agent architecture.

## Week 10 — Production

- FastAPI
- Docker
- Streamlit
- observability
- structured logs

**Deliverable:** deployable application.

## Week 11 — CI/CD + model experiments

- GitHub Actions
- regression evaluation
- second model
- latency/cost benchmark
- optional QLoRA

**Deliverable:** engineering-grade ML lifecycle.

## Week 12 — Expansion + presentation

- add first authoritative CFPB document source
- multi-source retrieval
- improve UI
- write architecture report
- publish evaluation report
- record demo

**Deliverable:** portfolio-ready project.

---

# 36. Definition of "portfolio ready"

Do not call the project finished just because the chatbot answers questions.

It is portfolio ready when the repository demonstrates:

## Data engineering

- reproducible ingestion
- validation
- BigQuery
- schema documentation
- provenance

## LLM engineering

- open-weight LLM
- structured outputs
- prompt/context engineering
- tool calling

## RAG

- embeddings
- semantic search
- hybrid retrieval
- reranking
- retrieval evaluation

## Agent engineering

- LangGraph
- planning
- tool selection
- multi-step workflow
- recovery/abstention

## Evaluation

- golden dataset
- regression tests
- correctness
- groundedness
- retrieval metrics
- safety metrics

## Production engineering

- FastAPI
- Docker
- CI/CD
- logging
- tracing
- latency/cost monitoring

## Security

- SQL restrictions
- prompt-injection defense
- source isolation
- safe tool execution
- abstention

## ML engineering

- model benchmark
- experiment tracking
- optional fine-tuning
- reproducible evaluation

---

# 37. What NOT to do

## Do not start with:

### ❌ Fine-tuning

You don't yet know what behavior needs to improve.

### ❌ Five agents

A complex graph with no measurable benefit is not impressive.

### ❌ Qdrant + pgvector + BigQuery at the same time

You will spend time wiring infrastructure instead of building the product.

### ❌ Random datasets

More rows do not automatically make the project better.

### ❌ A giant prompt

System reliability should come from:

```text
tools
schemas
validators
retrieval
evaluation
guardrails
```

not one enormous system prompt.

### ❌ "RAG chatbot" positioning

The stronger story is:

```text
LLM-powered financial intelligence system
```

### ❌ Claims like "hallucination-free"

Instead report measurable evaluation results.

---

# 38. The first 10 things to do — literally

If starting today, do these in exactly this order:

### 1.

Create a branch:

```bash
git checkout -b feature/data-foundation
```

### 2.

Create:

```text
docs/project-scope.md
```

### 3.

Create:

```text
docs/data-dictionary.md
```

### 4.

Profile `credit_card.csv`.

Record:

```text
rows
columns
nulls
duplicates
dates
companies
issues
narrative lengths
```

### 5.

Create:

```text
notebooks/01_data_profile.ipynb
```

### 6.

Create the BigQuery dataset.

Recommended logical layers:

```text
raw
staging
analytics
retrieval
```

### 7.

Load the raw complaint data without changing it.

### 8.

Build:

```text
staging.complaints
```

with normalized types and column names.

### 9.

Build:

```text
analytics.company_issue_counts
```

### 10.

Answer this question deterministically:

> Which bank has had the most issues with timeouts in credit-card complaints?

Before using any LLM.

This becomes our first golden test.

---

# 39. First SQL/LLM benchmark

Create the following progression.

### Question 1

> How many credit-card complaints are in the dataset?

Expected path:

```text
SQL
```

### Question 2

> Which company has the most credit-card complaints?

Expected path:

```text
SQL
```

### Question 3

> Which company has the most complaints about [specific issue]?

Expected path:

```text
SQL
```

### Question 4

> Find complaints where customers describe a payment being reversed incorrectly.

Expected path:

```text
semantic retrieval
```

### Question 5

> Which company has the most complaints about payment problems, and what are customers saying?

Expected path:

```text
SQL + retrieval
```

### Question 6

> Investigate whether Company X's payment complaints increased and summarize the main themes.

Expected path:

```text
multi-step investigation
```

These six questions effectively define the architecture.

---

# 40. Evaluation scorecard

Maintain:

```text
docs/evaluation.md
```

Track:

| Metric | Initial target |
|---|---:|
| SQL execution success | >95% |
| SQL answer correctness | >90% |
| Retrieval Recall@10 | establish baseline first |
| Citation correctness | >95% |
| Grounded answer rate | >90% |
| Tool-selection accuracy | >90% |
| Prompt-injection detection | >95% on test set |
| Scope classification | >95% on test set |
| p95 latency | measure baseline |
| Cost/query | measure baseline |
| Regression failures | 0 on protected cases |

These are **engineering targets**, not claims about market standards. Adjust them after observing the baseline.

---

# 41. Data provenance requirements

Every retrieved item should be traceable to:

```text
source
dataset
table
record ID
ingestion version
timestamp
```

For example:

```json
{
  "source": "CFPB Consumer Complaint Database",
  "dataset_version": "2026-10-02",
  "complaint_id": "12345678",
  "table": "complaints",
  "retrieved_at": "..."
}
```

This becomes extremely important once the system contains multiple sources.

---

# 42. Citation strategy

The final answer should distinguish:

### Quantitative evidence

```text
BigQuery query result
```

from:

### Narrative evidence

```text
complaint record(s)
```

from:

### External knowledge

```text
official CFPB document
```

Do not mix them into an opaque paragraph.

Example response structure:

```text
Answer

Quantitative evidence
- Company A: 12,431 complaints
- Company B: 10,203 complaints

Narrative evidence
- Complaint #...
- Complaint #...

Method
- SQL aggregation + semantic retrieval

Caveat
- CFPB states complaint counts are not a statistical sample...
```

---

# 43. The eventual demo

The demo should open with:

```text
DOMAIN VAULT RAG

Financial Intelligence Agent
```

User enters:

> Which companies have the most complaints about payment processing, and what are consumers saying?

UI shows:

```text
Intent
HYBRID INVESTIGATION

Plan
1. Aggregate complaints
2. Identify top companies
3. Retrieve representative narratives
4. Summarize themes
5. Verify evidence

Tools
✓ BigQuery
✓ Hybrid Search
✓ Reranker
✓ LLM
✓ Evaluator

Answer
...

Evidence
...

Evaluation
✓ grounded
✓ citations verified
✓ SQL verified
```

This communicates the engineering much better than a generic chat window.

---

# 44. Resume/project positioning

Do not lead with:

> Built a RAG chatbot using LangChain.

Instead emphasize:

> Built an end-to-end LLM-powered financial intelligence platform over 240K+ consumer credit-card complaints, combining natural-language-to-SQL analytics in BigQuery with hybrid semantic retrieval, tool-using agents, evidence-grounded generation, automated evaluation, safety guardrails, observability, and containerized deployment.

Then quantify the system:

```text
X evaluation questions
X% SQL accuracy
X% retrieval Recall@K
X ms p95 latency
X models benchmarked
X% citation correctness
X guardrail test cases
```

Only use numbers after they are actually measured.

---

# 45. Final architecture after the full build

```text
                           USER
                            │
                            ▼
                    ┌───────────────┐
                    │ FastAPI API   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Scope / Intent│
                    │ Classifier    │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ LangGraph     │
                    │ Planner       │
                    └───────┬───────┘
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ▼               ▼                ▼
       ┌────────┐      ┌──────────┐     ┌─────────┐
       │ SQL    │      │ Semantic │     │ Docs    │
       │ Tool   │      │ Search   │     │ Search  │
       └───┬────┘      └────┬─────┘     └────┬────┘
           │                │                │
           ▼                ▼                ▼
       BigQuery        Vector Search    Official docs
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                    ┌───────────────┐
                    │ Evidence      │
                    │ Builder       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Open-weight   │
                    │ LLM           │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Evaluator     │
                    │ + Guardrails  │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
              ANSWER              ABSTAIN /
              + CITATIONS         HUMAN REVIEW
                 │
                 ▼
          ┌─────────────────┐
          │ Observability   │
          │ traces + evals  │
          └─────────────────┘
```

---

# 46. The project philosophy

The project should demonstrate five engineering principles:

## 1. Use the right tool for the question

```text
numbers → SQL
meaning → retrieval
complex investigation → agent
```

## 2. Measure before optimizing

```text
baseline
→ change
→ evaluation
→ decision
```

## 3. Treat LLMs as probabilistic components

Use:

```text
schemas
validators
tests
guardrails
evaluators
```

## 4. Evidence before explanation

The LLM should explain retrieved evidence rather than invent evidence.

## 5. Production thinking

A portfolio project becomes much stronger when it demonstrates:

```text
quality
security
latency
cost
observability
deployment
reproducibility
```

---

# 47. Final rule for adding new data

Before adding a new dataset, create:

```text
docs/data-sources.md
```

For every source:

```text
Source:
Owner:
Official URL:
Purpose:
Why current data cannot answer this:
Schema:
License:
Access method:
Update frequency:
Expected query types:
PII considerations:
Retention:
Evaluation plan:
```

If you cannot fill this out quickly and confidently:

**Do not add the dataset.**

---

# 48. What we should build first

The immediate target is deliberately small:

```text
CFPB credit-card complaints
        │
        ▼
BigQuery
        │
        ├── SQL analytics
        │
        └── narrative embeddings
                │
                ▼
          hybrid retrieval
                │
                ▼
           open-weight LLM
                │
                ▼
          grounded answer
                │
                ▼
             evaluator
```

Once this works, the project has a solid core.

Everything else is an extension of that core.

---

# 49. Current status checklist

## Data

- [ ] Profile `credit_card.csv`
- [ ] Document schema
- [ ] Validate fields
- [ ] Load BigQuery
- [ ] Build raw/staging/analytics layers
- [ ] Document CFPB provenance

## Analytics

- [ ] Canonical SQL queries
- [ ] SQL views
- [ ] Data dictionary
- [ ] 20 verified analytical questions
- [ ] SQL tests

## LLM

- [ ] Select open-weight baseline
- [ ] Structured outputs
- [ ] NL → SQL
- [ ] SQL validator
- [ ] Model benchmark

## Retrieval

- [ ] Narrative table
- [ ] Embeddings
- [ ] Semantic search
- [ ] Keyword search
- [ ] Hybrid search
- [ ] Retrieval evaluation
- [ ] Reranking experiment

## RAG

- [ ] Evidence builder
- [ ] Grounded generation
- [ ] Citations
- [ ] Abstention
- [ ] RAG evaluation

## Agent

- [ ] LangGraph
- [ ] Planner
- [ ] SQL tool
- [ ] Search tool
- [ ] Investigation mode
- [ ] Tool-call evaluation

## Safety

- [ ] Scope guardrail
- [ ] Prompt-injection defense
- [ ] SQL restrictions
- [ ] PII checks
- [ ] Source isolation
- [ ] Safety test set

## Production

- [ ] FastAPI
- [ ] Streamlit
- [ ] Docker
- [ ] Observability
- [ ] GitHub Actions
- [ ] Regression evaluation
- [ ] Deployment

## ML experimentation

- [ ] Second model
- [ ] Model benchmark
- [ ] Optional QLoRA
- [ ] Before/after evaluation
- [ ] MLflow
- [ ] Experiment report

## Expansion

- [ ] First official CFPB document source
- [ ] Multi-source retrieval
- [ ] Source-level isolation
- [ ] Optional Qdrant/pgvector benchmark
- [ ] Optional GraphRAG
- [ ] Optional MCP/tool protocol support

---

# 50. Bottom line

**Do not change the repository name.**

Keep:

```text
domain-vault-rag
```

But change the project's implementation order.

The project starts with one excellent dataset rather than many mediocre ones:

```text
CFPB credit-card complaints
```

Then progressively proves:

```text
data engineering
→ SQL intelligence
→ LLM SQL generation
→ semantic retrieval
→ RAG
→ hybrid reasoning
→ tool-using agents
→ evaluation
→ safety
→ observability
→ production deployment
→ model experimentation
→ authoritative financial knowledge
→ domain isolation
```

That progression is the actual portfolio story.

The strongest artifact is not the number of frameworks in `requirements.txt`.

It is being able to demonstrate, with measurements:

> **"I took messy real-world financial data, designed a system that knows when to use SQL versus retrieval, gave an LLM controlled tools, grounded its answers in evidence, evaluated the system automatically, defended it against common LLM failure modes, instrumented it, and deployed it as a real service."**

That is the standard this build should aim for.
