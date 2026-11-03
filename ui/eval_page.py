"""
"Model evaluation" page: compares the local models using the results that
`python -m eval.run_eval` writes to eval/results/.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

METRICS = {
    "answer_correct": ("Answer correct", "Answer states the facts from the reference SQL"),
    "sql_result_correct": ("SQL correct", "Model's SQL returned the reference facts"),
    "routing_correct": ("Routing", "Picked the expected route (SQL / text / both)"),
    "cites_sources": ("Cites sources", "Text-search answers that mention a record id"),
    "success": ("No errors", "Questions answered without an API error"),
}


def _pct(series: pd.Series) -> float | None:
    values = series.dropna().astype(float)
    return float(100 * values.mean()) if len(values) else None


def _to_score(value) -> float | None:
    """CSV cell -> 0..1 score, or None when the metric doesn't apply to the question."""
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        booleans = {"True": 1.0, "False": 0.0}
        return booleans[value] if value in booleans else float(value)
    return float(value)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in METRICS:
        if column in df:
            df[column] = df[column].map(_to_score).astype(float)
    df["error"] = df["error"].fillna("")
    df["answer"] = df["answer"].fillna("")
    df["generated_sql"] = df["generated_sql"].fillna("")
    return df


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model"):
        row = {"Model": model}
        for column, (label, _) in METRICS.items():
            row[label] = _pct(g[column])
        row["Median time (s)"] = g["latency_seconds"].median()
        row["Load time (s)"] = g["model_load_seconds"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Answer correct", "Routing"], ascending=False)


def how_to_run() -> None:
    st.code(
        "python -m eval.run_eval                          # every model in the picker\n"
        "python -m eval.run_eval --models llama3.2:latest # one model\n"
        "python -m eval.run_eval --limit 5                # quick smoke test",
        language="bash",
    )


def render() -> None:
    st.title("Model evaluation")
    st.caption(
        "How each local model does on the 25 test questions in `eval/questions.jsonl`. "
        "Facts are checked against reference SQL run directly on the data."
    )

    runs = sorted(RESULTS_DIR.glob("run_*.csv"), reverse=True)
    if not runs:
        st.info("No evaluation results yet. Run one from the project root:", icon=":material/info:")
        how_to_run()
        return

    labels = {
        p: pd.to_datetime(p.stem.removeprefix("run_"), format="%Y%m%d_%H%M%S").strftime(
            "%b %d, %Y %H:%M"
        )
        for p in runs
    }
    path = st.selectbox("Run", runs, format_func=labels.get, width=260)
    df = _load(path)
    models = sorted(df["model"].unique())
    st.caption(f"{df['question_id'].nunique()} questions × {len(models)} model(s)")

    # --- Leaderboard ------------------------------------------------------------------
    board = leaderboard(df)
    best = board.iloc[0]
    cols = st.columns(3)
    cols[0].metric(
        "Most accurate",
        best["Model"],
        f"{best['Answer correct']:.0f}% answers correct",
        delta_color="off",
    )
    fastest = board.sort_values("Median time (s)").iloc[0]
    cols[1].metric(
        "Fastest", fastest["Model"], f"{fastest['Median time (s)']:.1f}s median", delta_color="off"
    )
    best_route = board.sort_values("Routing", ascending=False).iloc[0]
    cols[2].metric(
        "Best router",
        best_route["Model"],
        f"{best_route['Routing']:.0f}% routed right",
        delta_color="off",
    )

    st.dataframe(
        board,
        hide_index=True,
        width="stretch",
        column_config={
            **{
                label: st.column_config.ProgressColumn(
                    label, help=help_, format="%.0f%%", min_value=0, max_value=100
                )
                for label, help_ in METRICS.values()
            },
            "Median time (s)": st.column_config.NumberColumn(format="%.1f"),
            "Load time (s)": st.column_config.NumberColumn(
                format="%.1f", help="First request: loading the model into memory"
            ),
        },
    )

    # --- Charts -----------------------------------------------------------------------
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Accuracy vs. speed**")
        fig = px.scatter(
            board,
            x="Median time (s)",
            y="Answer correct",
            text="Model",
            labels={"Answer correct": "Answers correct (%)"},
        )
        fig.update_traces(marker={"size": 14}, textposition="top center")
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=300, yaxis_range=[0, 105]
        )
        st.plotly_chart(fig, width="stretch")
    with chart_right:
        st.markdown("**Routing by question type**")
        by_route = (
            df.groupby(["model", "expected_route"])["routing_correct"].mean().mul(100).reset_index()
        )
        fig = px.bar(
            by_route,
            x="expected_route",
            y="routing_correct",
            color="model",
            barmode="group",
            labels={
                "expected_route": "Expected route",
                "routing_correct": "Routed right (%)",
                "model": "",
            },
        )
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            height=300,
            yaxis_range=[0, 105],
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig, width="stretch")

    # --- Per question -----------------------------------------------------------------
    st.subheader("Question by question")

    def mark(g: pd.DataFrame) -> str:
        row = g.iloc[0]
        if not row["success"]:
            return "⚠️ error"
        score = row["answer_correct"]
        route_ok = "" if row["routing_correct"] else " ↪"
        if score is None or pd.isna(score):
            return ("✓" if row["routing_correct"] else "✗") + " route"
        return ("✓" if score == 1 else "◐" if score > 0 else "✗") + route_ok

    grid = (
        df.groupby(["question_id", "question", "model"])
        .apply(mark, include_groups=False)
        .unstack("model")
        .reset_index()
        .drop(columns="question_id")
    )
    st.caption(
        "✓ correct · ◐ partly · ✗ wrong · ↪ wrong route · questions without reference SQL "
        "are scored on routing only"
    )
    st.dataframe(grid, hide_index=True, width="stretch", height=min(38 * (len(grid) + 1), 520))

    questions = df.drop_duplicates("question_id").set_index("question_id")["question"]
    qid = st.selectbox(
        "Compare answers for",
        questions.index,
        format_func=lambda i: f"{i + 1}. {questions[i]}",
    )
    subset = df[df["question_id"] == qid]
    first = subset.iloc[0]
    meta = [
        f"Line of business: **{first['lob']}**",
        f"Expected route: **{first['expected_route']}**",
    ]
    if first.get("expected_facts"):
        meta.append(
            f"Reference facts: `{', '.join(map(str, json.loads(first['expected_facts'])))}`"
        )
    st.markdown(" · ".join(meta))

    for col, (_, row) in zip(st.columns(len(subset)), subset.iterrows(), strict=True):
        with col, st.container(border=True):
            st.markdown(f"**{row['model']}**")
            with st.container(horizontal=True, gap="small", wrap=True):
                st.badge(
                    row["actual_route"] or "no route",
                    color="green" if row["routing_correct"] else "red",
                )
                score = row["answer_correct"]
                if score is not None and not pd.isna(score):
                    st.badge(
                        f"facts {score:.0%}",
                        color="green" if score == 1 else "orange" if score > 0 else "red",
                    )
                st.badge(f"{row['latency_seconds']:.1f}s", color="gray")
            if row["error"]:
                st.caption(f":red[{row['error'][:300]}]")
            st.markdown(row["answer"] or "_no answer_")
            if row["generated_sql"]:
                with st.expander("SQL"):
                    st.code(row["generated_sql"], language="sql")

    with st.expander("How to run an evaluation", icon=":material/terminal:"):
        how_to_run()
        st.caption(
            "Re-score a saved run after changing the scoring or questions: "
            "`python -m eval.run_eval --rescore eval/results/latest.csv`"
        )
