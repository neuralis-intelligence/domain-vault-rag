"""
Evaluate and compare the local models on the question set.

For every model and question it records:
- Routing: did the agent pick the expected route (sql / rag / hybrid)?
- SQL result correct: does the model's SQL return the same facts as `reference_sql`?
- Answer correct: does the written answer state those facts? (share of facts found)
- Cites sources: for text-search answers, does it mention a retrieved record id?
- Keyword match: the original rough check against `expected_answer_contains`
- Latency, success / error

Each model gets one warm-up question first, so loading it into memory isn't counted as
answer time (the load time is reported separately).

Usage (API running, from the project root):
    python -m eval.run_eval                                  # every model the UI offers
    python -m eval.run_eval --models llama3.2:latest qwen2.5:7b-instruct
    python -m eval.run_eval --models llama3.2:latest --limit 5    # quick smoke test
    python -m eval.run_eval --rescore eval/results/latest.csv     # re-score a saved run

Results go to eval/results/run_<timestamp>.csv and eval/results/latest.csv; the UI's
"Model evaluation" page reads latest.csv.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests

from app.config import settings
from app.lob import get_lob
from eval import scoring

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
WARMUP_QUESTION = "How many complaints are there?"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ground_truth(questions: list[dict[str, Any]], db_path: Path) -> dict[int, list[Any]]:
    """Run each question's reference_sql on DuckDB; returns {question index: facts}."""
    truth: dict[int, list[Any]] = {}
    with duckdb.connect(str(db_path), read_only=True) as con:
        for i, q in enumerate(questions):
            if not q.get("reference_sql"):
                continue
            sql = q["reference_sql"].format(table=get_lob(q.get("lob")).table)
            cursor = con.execute(sql)
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
            truth[i] = scoring.expected_values(rows, q.get("check", "first_row"))
    return truth


class EvaluationRunner:
    def __init__(self, api_url: str, timeout: int = 300):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def available_models(self) -> list[str]:
        response = requests.get(f"{self.api_url}/models", timeout=10)
        response.raise_for_status()
        return [m["name"] for m in response.json()["models"]]

    def ask(
        self, question: str, lob: str | None, model: str
    ) -> tuple[dict | None, str | None, float]:
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{self.api_url}/ask",
                json={"question": question, "lob": lob, "model": model},
                timeout=self.timeout,
            )
            elapsed = time.perf_counter() - start
            if not response.ok:
                return None, f"HTTP {response.status_code}: {response.text[:300]}", elapsed
            return response.json(), None, elapsed
        except requests.RequestException as e:
            return None, str(e), time.perf_counter() - start

    def warm_up(self, model: str) -> float:
        """Load the model into memory; returns the seconds it took."""
        logger.info("Warming up %s", model)
        _, error, seconds = self.ask(WARMUP_QUESTION, None, model)
        if error:
            logger.warning("Warm-up for %s failed: %s", model, error)
        return seconds

    def evaluate_model(
        self,
        model: str,
        questions: list[dict[str, Any]],
        truth: dict[int, list[Any]],
        run_id: str,
    ) -> list[dict[str, Any]]:
        load_seconds = self.warm_up(model)
        rows = []
        for i, q in enumerate(questions):
            logger.info("[%s] %d/%d %s", model, i + 1, len(questions), q["question"][:60])
            result, error, seconds = self.ask(q["question"], q.get("lob"), model)
            result = result or {}
            metadata = result.get("metadata") or {}
            sql_result = metadata.get("sql_result") or {}
            documents = (metadata.get("rag_result") or {}).get("documents") or []
            answer = result.get("answer", "")
            route = result.get("route", "")
            row = {
                "run_id": run_id,
                "model": model,
                "model_load_seconds": round(load_seconds, 2),
                "question_id": i,
                "question": q["question"],
                "lob": q.get("lob"),
                "expected_route": q.get("expected_route"),
                "actual_route": route,
                "latency_seconds": round(seconds, 2),
                "success": error is None,
                "error": error or sql_result.get("error") or "",
                "generated_sql": sql_result.get("sql", ""),
                "answer": answer,
                # Raw outputs, kept so `--rescore` can re-score without re-asking.
                "sql_rows": json.dumps((sql_result.get("results") or [])[:50], default=str),
                "source_ids": json.dumps([str(d.get("id")) for d in documents]),
            }
            rows.append(score_row(row, q, truth.get(i, [])))
        return rows


def score_row(row: dict[str, Any], q: dict[str, Any], expected: list[Any]) -> dict[str, Any]:
    """Add the score columns to one raw result row."""
    answer = row.get("answer") or ""
    route = row.get("actual_route") or ""
    documents = [{"id": doc_id} for doc_id in json.loads(row.get("source_ids") or "[]")]
    row.update(
        {
            "routing_correct": bool(route) and route == q.get("expected_route"),
            "expected_facts": json.dumps(expected, default=str) if expected else "",
            "sql_result_correct": scoring.result_matches(
                expected, json.loads(row.get("sql_rows") or "[]")
            ),
            "answer_correct": scoring.fraction_mentioned(expected, answer),
            "cites_sources": scoring.cites_sources(answer, documents)
            if route in ("rag", "hybrid")
            else None,
            "keyword_match": scoring.keyword_rate(answer, q.get("expected_answer_contains", [])),
        }
    )
    return row


def rescore(path: Path, questions_path: Path) -> pd.DataFrame:
    """Re-apply the current scoring (and ground truth) to a saved run."""
    questions = load_questions(questions_path)
    truth = ground_truth(questions, settings.duckdb_path)
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    rows = []
    for raw in df.to_dict("records"):
        raw = {k: ("" if isinstance(v, float) and pd.isna(v) else v) for k, v in raw.items()}
        i = int(raw["question_id"])
        rows.append(score_row(raw, questions[i], truth.get(i, [])))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model: the leaderboard."""

    def pct(series: pd.Series) -> float:
        values = series.dropna().astype(float)
        return round(100 * values.mean(), 1) if len(values) else float("nan")

    summary = df.groupby("model").apply(
        lambda g: pd.Series(
            {
                "questions": len(g),
                "routing_accuracy_%": pct(g["routing_correct"]),
                "sql_correct_%": pct(g["sql_result_correct"]),
                "answer_correct_%": pct(g["answer_correct"]),
                "cites_sources_%": pct(g["cites_sources"]),
                "keyword_match_%": pct(g["keyword_match"]),
                "success_%": pct(g["success"]),
                "median_latency_s": round(g["latency_seconds"].median(), 1),
                "p90_latency_s": round(g["latency_seconds"].quantile(0.9), 1),
                "load_s": g["model_load_seconds"].iloc[0],
            }
        ),
        include_groups=False,
    )
    return summary.sort_values(["answer_correct_%", "routing_accuracy_%"], ascending=False)


def print_report(df: pd.DataFrame) -> None:
    summary = summarize(df)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\n" + "=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)
        print(summary.to_string())
        print("\nRouting accuracy % by expected route:")
        by_route = df.pivot_table(
            index="model", columns="expected_route", values="routing_correct", aggfunc="mean"
        )
        print((100 * by_route).round(1).to_string())
        print("\nAnswer correct % by line of business (questions with ground truth):")
        by_lob = df.dropna(subset=["answer_correct"]).pivot_table(
            index="model", columns="lob", values="answer_correct", aggfunc="mean"
        )
        print((100 * by_lob).round(1).to_string())
        print("=" * 100)
        print(
            "sql_correct / answer_correct use reference SQL; cites_sources applies to "
            "text-search answers; keyword_match is the old rough check."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare local models")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names, or 'all'")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, help="Only the first N questions (smoke test)")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per question")
    parser.add_argument(
        "--rescore", type=Path, metavar="CSV", help="Re-score a saved run instead of running"
    )
    args = parser.parse_args()

    if args.rescore:
        df = rescore(args.rescore, args.questions)
        df.to_csv(args.rescore, index=False)
        print_report(df)
        logger.info("Re-scored %s", args.rescore)
        return

    runner = EvaluationRunner(args.api_url, timeout=args.timeout)
    available = runner.available_models()
    models = available if args.models == ["all"] else args.models
    unknown = [m for m in models if m not in available]
    if unknown:
        raise SystemExit(f"Not available: {unknown}. Available: {available}")

    questions = load_questions(args.questions)[: args.limit]
    truth = ground_truth(questions, settings.duckdb_path)
    logger.info(
        "%d questions (%d with ground truth) x %d models: %s",
        len(questions),
        len(truth),
        len(models),
        ", ".join(models),
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.extend(runner.evaluate_model(model, questions, truth, run_id))
        # Save after each model, so an interrupted run still leaves results.
        RESULTS_DIR.mkdir(exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"run_{run_id}.csv", index=False)
        df.to_csv(RESULTS_DIR / "latest.csv", index=False)

    print_report(df)
    logger.info("Saved %s and %s", RESULTS_DIR / f"run_{run_id}.csv", RESULTS_DIR / "latest.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
