"""
Evaluation script for Complaint Insights Agent.

Measures:
1. Routing accuracy: Does the router pick the correct path?
2. Answer quality: Does the answer contain expected content?
3. Latency: How long does each query take?
4. Error rate: How many queries fail?
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Run evaluation suite on question set."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    def load_questions(self, filepath: str = None) -> list[dict[str, Any]]:
        """Load test questions from JSONL file."""
        if filepath is None:
            filepath = Path(__file__).parent / "questions.jsonl"

        questions = []
        with open(filepath) as f:
            for line in f:
                questions.append(json.loads(line))

        logger.info(f"Loaded {len(questions)} test questions")
        return questions

    def query_api(self, question: str) -> dict[str, Any]:
        """Send question to API and measure latency."""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.api_url}/ask", json={"question": question}, timeout=180
            )
            response.raise_for_status()
            result = response.json()
            latency = time.time() - start_time

            return {
                "success": True,
                "answer": result.get("answer", ""),
                "route": result.get("route", "unknown"),
                "metadata": result.get("metadata", {}),
                "latency": latency,
                "error": None,
            }

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"API call failed: {e}")
            return {
                "success": False,
                "answer": "",
                "route": "unknown",
                "metadata": {},
                "latency": latency,
                "error": str(e),
            }

    def evaluate_routing(self, predicted: str, expected: str) -> bool:
        """Check if route matches expectation."""
        return predicted.lower() == expected.lower()

    def evaluate_answer(self, answer: str, expected_contains: list[str]) -> dict[str, Any]:
        """
        Check if answer contains expected content.

        Returns:
            Dict with match_count and match_rate
        """
        answer_lower = answer.lower()
        matches = sum(1 for term in expected_contains if term.lower() in answer_lower)

        return {
            "matches": matches,
            "total": len(expected_contains),
            "match_rate": matches / len(expected_contains) if expected_contains else 0,
        }

    def run_evaluation(self, questions: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Run evaluation on all questions.

        Returns:
            DataFrame with results
        """
        logger.info("Starting evaluation...")

        for i, q in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {q['question'][:50]}...")

            result = self.query_api(q["question"])

            # Evaluate routing
            routing_correct = self.evaluate_routing(
                result["route"], q.get("expected_route", "unknown")
            )

            # Evaluate answer quality
            answer_eval = self.evaluate_answer(
                result["answer"], q.get("expected_answer_contains", [])
            )

            # Store result
            self.results.append(
                {
                    "question": q["question"],
                    "expected_route": q.get("expected_route", "unknown"),
                    "actual_route": result["route"],
                    "routing_correct": routing_correct,
                    "answer": result["answer"],
                    "answer_match_rate": answer_eval["match_rate"],
                    "latency": result["latency"],
                    "success": result["success"],
                    "error": result["error"],
                }
            )

        return pd.DataFrame(self.results)

    def compute_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute aggregate metrics."""
        metrics = {
            "total_questions": len(df),
            "success_rate": df["success"].mean(),
            "routing_accuracy": df["routing_correct"].mean(),
            "answer_quality": df["answer_match_rate"].mean(),
            "avg_latency": df["latency"].mean(),
            "median_latency": df["latency"].median(),
            "error_rate": (~df["success"]).mean(),
        }

        # Breakdown by route
        route_breakdown = defaultdict(dict)
        for route in df["expected_route"].unique():
            route_df = df[df["expected_route"] == route]
            route_breakdown[route] = {
                "count": len(route_df),
                "routing_accuracy": route_df["routing_correct"].mean(),
                "answer_quality": route_df["answer_match_rate"].mean(),
                "avg_latency": route_df["latency"].mean(),
            }

        metrics["route_breakdown"] = dict(route_breakdown)

        return metrics

    def print_report(self, metrics: dict[str, Any]):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print("\nOverall Metrics:")
        print(f"  Total Questions:    {metrics['total_questions']}")
        print(f"  Success Rate:       {metrics['success_rate']:.1%}")
        print(f"  Routing Accuracy:   {metrics['routing_accuracy']:.1%}")
        print(f"  Answer Quality:     {metrics['answer_quality']:.1%}")
        print(f"  Avg Latency:        {metrics['avg_latency']:.2f}s")
        print(f"  Median Latency:     {metrics['median_latency']:.2f}s")
        print(f"  Error Rate:         {metrics['error_rate']:.1%}")

        print("\nBreakdown by Route:")
        for route, stats in metrics["route_breakdown"].items():
            print(f"\n  {route.upper()}:")
            print(f"    Questions:        {stats['count']}")
            print(f"    Routing Accuracy: {stats['routing_accuracy']:.1%}")
            print(f"    Answer Quality:   {stats['answer_quality']:.1%}")
            print(f"    Avg Latency:      {stats['avg_latency']:.2f}s")

        print("\n" + "=" * 60)

    def save_results(self, df: pd.DataFrame, output_path: str = None):
        """Save detailed results to CSV."""
        if output_path is None:
            output_path = Path(__file__).parent / "eval_results.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def main():
    """Run evaluation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation on Complaint Insights Agent")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--questions", default=None, help="Path to questions JSONL file")
    parser.add_argument("--output", default=None, help="Output CSV path")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = EvaluationRunner(api_url=args.api_url)

    # Load questions
    questions = evaluator.load_questions(args.questions)

    # Run evaluation
    results_df = evaluator.run_evaluation(questions)

    # Compute metrics
    metrics = evaluator.compute_metrics(results_df)

    # Print report
    evaluator.print_report(metrics)

    # Save results
    evaluator.save_results(results_df, args.output)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
