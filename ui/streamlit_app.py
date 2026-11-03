"""
Streamlit UI for the Complaint Insights Agent.

- Chat interface; the line of business and the local model are picked right next to
  the message box, per question
- Answers show the route, dataset, model and response time, with the SQL result
  (chart / table / query) and the matching texts in collapsible sections
"""

import os
import time
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
# Local LLMs can take a while: routing + SQL generation + synthesis are 3 LLM calls.
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "180"))
REPO_URL = "https://github.com/neuralis-intelligence/domain-vault-rag"

ROUTE_BADGES = {
    "sql": ("SQL", ":material/table_chart:", "blue"),
    "rag": ("Text search", ":material/search:", "violet"),
    "hybrid": ("SQL + text", ":material/join:", "orange"),
}

st.set_page_config(
    page_title="Complaint Insights",
    page_icon=":material/forum:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# A little spacing polish; colors come from the theme so dark mode keeps working.
st.html(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 7rem; }
    .st-key-composer-options { gap: 0.5rem; }
    .st-key-composer-options [data-testid="stSelectbox"] { min-width: 0; }
    [data-testid="stChatMessage"] { padding: 0.75rem 0.25rem; }
    .muted { opacity: 0.65; font-size: 0.9rem; }
    </style>
    """
)


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def fetch(path: str) -> dict[str, Any] | None:
    """GET an API endpoint (cached briefly so reruns don't refetch)."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def ask(question: str, lob: str, model: str) -> tuple[dict[str, Any] | None, str | None, float]:
    """POST /ask. Returns (result, error message, seconds taken)."""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "lob": lob, "model": model},
            timeout=API_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout:
        return None, f"No answer after {API_TIMEOUT_SECONDS}s. Try a smaller model.", 0.0
    except requests.exceptions.RequestException as e:
        return None, f"Couldn't reach the API at {API_URL}: {e}", 0.0

    elapsed = time.perf_counter() - start
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{detail} (HTTP {response.status_code})", elapsed
    return response.json(), None, elapsed


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def render_chart(df: pd.DataFrame, key: str) -> None:
    """Bar chart for label/value results, line chart for time series."""
    if len(df.columns) != 2 or not 1 < len(df) <= 50:
        return
    x, y = df.columns
    if not pd.api.types.is_numeric_dtype(df[y]):
        return
    labels = {x: x.replace("_", " ").title(), y: y.replace("_", " ").title()}
    is_time = pd.api.types.is_datetime64_any_dtype(df[x]) or any(
        word in x.lower() for word in ("date", "month", "year", "week", "day")
    )
    if is_time:
        fig = px.line(df, x=x, y=y, markers=True, labels=labels)
    else:
        fig = px.bar(df.sort_values(y), x=y, y=x, orientation="h", labels=labels)
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=max(260, 28 * len(df)) if not is_time else 320,
        yaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")


def render_sql(sql_result: dict[str, Any], key: str) -> None:
    rows = sql_result.get("results") or []
    if sql_result.get("error"):
        with st.expander("Query details", icon=":material/error:"):
            st.caption(sql_result["error"])
            if sql_result.get("sql"):
                st.code(sql_result["sql"], language="sql")
        return
    if not rows:
        return

    df = pd.DataFrame(rows)
    with st.expander(
        f"Data · {len(df):,} row{'s' if len(df) != 1 else ''}",
        icon=":material/table_chart:",
        expanded=len(df) > 1,
    ):
        chart_tab, table_tab, sql_tab = st.tabs(["Chart", "Table", "SQL"])
        with chart_tab:
            if len(df.columns) == 2 and len(df) > 1:
                render_chart(df, key)
            else:
                st.caption("No chart for this result shape.")
        with table_tab:
            st.dataframe(df, width="stretch", hide_index=True)
        with sql_tab:
            st.code(sql_result.get("sql", ""), language="sql")


def render_sources(documents: list[dict[str, Any]]) -> None:
    if not documents:
        return
    with st.expander(f"Sources · {len(documents)}", icon=":material/description:"):
        for i, doc in enumerate(documents):
            if i:
                st.divider()
            fields = doc.get("fields") or {}
            title = " · ".join(str(v) for k, v in fields.items() if v and k in ("company", "issue"))
            title = title or " · ".join(str(v) for v in list(fields.values())[:2] if v) or "Result"
            head, score = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"**{title}**")
            score.badge(f"{doc.get('score', 0):.2f}", color="gray")
            meta = [f"#{doc.get('id')}"] + [
                str(v) for k, v in fields.items() if v and k not in ("company", "issue")
            ]
            st.caption(" · ".join(meta))
            text = doc.get("text") or ""
            st.markdown(f"> {text[:600]}{'…' if len(text) > 600 else ''}")


def render_meta(message: dict[str, Any]) -> None:
    """Small badges under an answer: route, dataset, model, time."""
    label, icon, color = ROUTE_BADGES.get(message.get("route", ""), ("?", None, "gray"))
    with st.container(horizontal=True, gap="small"):
        st.badge(label, icon=icon, color=color)
        st.badge(message.get("lob_name", ""), icon=":material/folder:", color="gray")
        st.badge(message.get("model", ""), icon=":material/memory:", color="gray")
        if message.get("seconds"):
            st.badge(f"{message['seconds']:.1f}s", icon=":material/timer:", color="gray")


def render_assistant(message: dict[str, Any], key: str) -> None:
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")
        return
    st.markdown(message["content"])
    render_meta(message)
    if message.get("sql_result"):
        render_sql(message["sql_result"], key)
    if message.get("documents"):
        render_sources(message["documents"])


# --------------------------------------------------------------------------------------
# Pickers (shown next to the message box)
# --------------------------------------------------------------------------------------


def pick_lob(lobs_data: dict[str, Any]) -> dict[str, Any] | None:
    loaded = [lob for lob in lobs_data["lobs"] if lob["sql_available"]]
    if not loaded:
        return None
    by_id = {lob["id"]: lob for lob in loaded}
    if st.session_state.get("lob") not in by_id:
        st.session_state.lob = (
            lobs_data["default"] if lobs_data["default"] in by_id else loaded[0]["id"]
        )
    lob_id = st.selectbox(
        "Line of business",
        list(by_id),
        key="lob",
        format_func=lambda i: by_id[i]["name"],
        label_visibility="collapsed",
        width=190,
        help="Which dataset to ask about",
    )
    return by_id[lob_id]


def pick_model(models_data: dict[str, Any]) -> str | None:
    models = models_data["models"]
    if not models:
        return None
    by_name = {m["name"]: m for m in models}
    if st.session_state.get("model") not in by_name:
        default = models_data["default"]
        st.session_state.model = default if default in by_name else models[0]["name"]
    return st.selectbox(
        "Model",
        list(by_name),
        key="model",
        format_func=lambda n: f"{n} · {by_name[n]['parameter_size'] or '?'}",
        label_visibility="collapsed",
        width=220,
        help="Local model that answers. Smaller is faster; larger writes better SQL.",
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


def sidebar(health: dict[str, Any] | None) -> None:
    with st.sidebar:
        st.markdown("### :material/forum: Complaint Insights")
        st.caption("Ask questions about financial complaint data in plain English.")

        if st.button("New chat", icon=":material/add:", width="stretch", type="primary"):
            st.session_state.messages = []
            st.rerun()

        st.space("small")
        st.markdown("**Services**")
        if health is None:
            st.badge("API offline", icon=":material/cloud_off:", color="red")
        else:
            with st.container(horizontal=True, gap="small", wrap=True):
                for service in ("ollama", "qdrant", "duckdb"):
                    ok = health.get(service) == "connected"
                    st.badge(
                        service.capitalize() if service != "duckdb" else "DuckDB",
                        icon=":material/check_circle:" if ok else ":material/error:",
                        color="green" if ok else "red",
                    )

        st.space("small")
        with st.expander("How it works", icon=":material/info:"):
            st.markdown(
                "Pick a **line of business** and a **model** next to the message box. "
                "Each question is routed to:\n"
                "- **SQL** for counts, rankings and trends\n"
                "- **Text search** for what people wrote\n"
                "- **SQL + text** for both"
            )
        st.caption(f"[Source on GitHub]({REPO_URL})")


def empty_state(lob: dict[str, Any] | None) -> None:
    st.space("large")
    st.markdown("## What would you like to know?")
    if lob:
        st.markdown(
            f"<p class='muted'>Asking about <b>{lob['name']}</b> · {lob['rows']:,} records. "
            "Switch dataset or model next to the message box.</p>",
            unsafe_allow_html=True,
        )
        examples = lob.get("example_questions") or []
        if examples:
            key = f"examples_{lob['id']}"

            def use_example() -> None:
                st.session_state.pending_question = st.session_state[key]
                st.session_state[key] = None

            st.pills(
                "Try one of these",
                examples,
                key=key,
                on_change=use_example,
                label_visibility="collapsed",
            )


def main() -> None:
    st.session_state.setdefault("messages", [])

    health = fetch("/health")
    sidebar(health)

    lobs_data, models_data = fetch("/lobs"), fetch("/models")
    if health is None or lobs_data is None:
        st.error(
            f"Can't reach the API at {API_URL}. Start it with `uvicorn app.main:app --reload`.",
            icon=":material/cloud_off:",
        )
        st.stop()

    # Composer: dataset + model pickers beside the message box, pinned to the bottom.
    with st.bottom:
        with st.container(horizontal=True, gap="small", key="composer-options"):
            lob = pick_lob(lobs_data)
            model = pick_model(models_data) if models_data else None
            not_loaded = [x["name"] for x in lobs_data["lobs"] if not x["sql_available"]]
            if not_loaded:
                st.caption(f"Not loaded yet: {', '.join(not_loaded)}", width="content")
        typed = st.chat_input(
            f"Ask about {lob['name'].lower()}…" if lob else "No data loaded",
            disabled=not (lob and model),
        )

    if lob is None:
        st.warning("No line of business has data loaded. Run `python -m ingestion.load_duckdb`.")
    if model is None:
        st.warning("No local models found. Is Ollama running? Try `ollama pull llama3.2`.")

    question = typed or st.session_state.pop("pending_question", None)

    if not st.session_state.messages and not question:
        empty_state(lob)
        return

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message, key=str(i))

    if not (question and lob and model):
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model}…"):
            result, error, seconds = ask(question, lob["id"], model)

        metadata = (result or {}).get("metadata") or {}
        message = {
            "role": "assistant",
            "question": question,
            "content": (result or {}).get("answer", ""),
            "error": error,
            "route": (result or {}).get("route"),
            "lob_name": lob["name"],
            "model": (result or {}).get("model", model),
            "seconds": seconds,
            "sql_result": metadata.get("sql_result"),
            "documents": (metadata.get("rag_result") or {}).get("documents"),
        }
        render_assistant(message, key=str(len(st.session_state.messages)))
    st.session_state.messages.append(message)


main()
