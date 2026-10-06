"""
Streamlit UI for Complaint Insights Agent.

Features:
- Chat interface for natural language queries
- Visualization of SQL results (tables, charts)
- Display of RAG results with narrative snippets
- Query history
"""

import os
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
# Local LLMs can take a while: routing + SQL generation + synthesis are 3 LLM calls.
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "180"))
REPO_URL = "https://github.com/neuralis-intelligence/domain-vault-rag"

# Page config
st.set_page_config(
    page_title="CFPB Complaint Insights",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def query_api(question: str) -> dict[str, Any] | None:
    """Send question to FastAPI backend."""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def visualize_sql_results(results: list, question: str):
    """Create visualizations for SQL query results."""
    if not results:
        st.info("No results to display")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Display as table
    st.dataframe(df, width="stretch")

    # Auto-generate chart if appropriate
    if len(df.columns) == 2 and len(df) <= 20:
        # Likely a count/aggregation query - create bar chart
        col1, col2 = df.columns
        fig = px.bar(
            df,
            x=col1,
            y=col2,
            title=f"Results: {question}",
            labels={col1: col1.replace("_", " ").title(), col2: col2.replace("_", " ").title()},
        )
        st.plotly_chart(fig, width="stretch")


def display_rag_results(documents: list):
    """Display RAG search results with narratives."""
    if not documents:
        st.info("No narratives found")
        return

    st.subheader(f"Found {len(documents)} relevant complaints")

    for i, doc in enumerate(documents, 1):
        with st.expander(
            f"Complaint {i}: {doc.get('company', 'Unknown')} - {doc.get('issue', 'Unknown')}"
        ):
            st.markdown(f"**Complaint ID:** {doc.get('complaint_id', 'N/A')}")
            st.markdown(f"**Company:** {doc.get('company', 'N/A')}")
            st.markdown(f"**Product:** {doc.get('product', 'N/A')}")
            st.markdown(f"**Issue:** {doc.get('issue', 'N/A')}")
            st.markdown(f"**State:** {doc.get('state', 'N/A')}")
            st.markdown(f"**Similarity Score:** {doc.get('score', 0):.3f}")
            st.markdown("**Narrative:**")
            st.markdown(f"> {doc.get('narrative', 'No narrative available')}")


def main():
    """Main Streamlit app."""

    # Sidebar
    with st.sidebar:
        st.title("🏦 CFPB Complaint Insights")
        st.markdown("---")

        st.subheader("About")
        st.markdown("""
        Ask natural language questions about CFPB bank complaints.
        The system will automatically route your query to:
        - **SQL** for statistics and trends
        - **RAG** for narrative exploration
        - **Hybrid** for both
        """)

        st.markdown("---")
        st.subheader("Example Questions")
        examples = [
            "What is the most common issue at Bank of America?",
            "How many mortgage complaints did Wells Fargo receive?",
            "What are customers saying about Chase fraud cases?",
            "Top 5 banks by complaint volume",
            "Show me complaints about credit card fees",
        ]

        for example in examples:
            if st.button(example, key=f"example_{example[:20]}"):
                st.session_state.current_question = example

        st.markdown("---")

        # API status
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
        except requests.exceptions.RequestException:
            st.error("❌ API Offline")
        else:
            if health.get("status") == "healthy":
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API degraded")
            for service in ("ollama", "qdrant", "bigquery"):
                st.caption(f"{service}: {health.get(service, 'unknown')}")

    # Main content
    st.title("💬 Ask About Bank Complaints")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display visualizations if available
            if "sql_results" in message:
                visualize_sql_results(message["sql_results"], message.get("question", ""))

            if "rag_documents" in message:
                display_rag_results(message["rag_documents"])

    # Chat input
    question = st.chat_input("Ask a question about bank complaints...")

    # Handle example button clicks
    if st.session_state.current_question:
        question = st.session_state.current_question
        st.session_state.current_question = ""

    if question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({"role": "user", "content": question})

        # Query API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_api(question)

            if result:
                # Display answer
                answer = result.get("answer", "No answer available")
                route = result.get("route", "unknown")

                st.markdown(answer)
                st.caption(f"Route: {route.upper()}")

                # Store message with metadata
                message_data = {
                    "role": "assistant",
                    "content": answer,
                    "question": question,
                    "route": route,
                }

                # Handle SQL results
                if "sql_result" in result.get("metadata", {}):
                    sql_result = result["metadata"]["sql_result"]
                    if "results" in sql_result and sql_result["results"]:
                        message_data["sql_results"] = sql_result["results"]
                        visualize_sql_results(sql_result["results"], question)

                # Handle RAG results
                if "rag_result" in result.get("metadata", {}):
                    rag_result = result["metadata"]["rag_result"]
                    if "documents" in rag_result and rag_result["documents"]:
                        message_data["rag_documents"] = rag_result["documents"]
                        display_rag_results(rag_result["documents"])

                st.session_state.messages.append(message_data)

            else:
                st.error("Failed to get response from API")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Questions", len([m for m in st.session_state.messages if m["role"] == "user"])
        )
    with col2:
        if st.button("Clear History"):
            st.session_state.messages = []
            st.rerun()
    with col3:
        st.markdown(f"[View on GitHub]({REPO_URL})")


if __name__ == "__main__":
    main()
