"""Answer synthesis prompt."""


def get_synthesis_prompt(
    question: str,
    context: str,
    route: str,
    dataset: str = "",
    doc_label: str = "Record",
    has_text: bool = True,
) -> str:
    """
    Generate prompt for synthesizing final answer from context.
    """
    route_instructions = {
        "sql": "Use the SQL query results to provide a clear, concise answer with specific numbers.",
        "rag": f"Synthesize information from the {doc_label.lower()} texts, citing {doc_label.lower()} IDs when relevant.",
        "hybrid": f"Combine the statistical insights from SQL with specific examples from the {doc_label.lower()} texts.",
    }

    instruction = route_instructions.get(
        route, "Provide a clear answer based on the available data."
    )

    about = f"{dataset} data" if dataset else "the available data"
    text_note = (
        ""
        if has_text
        else "- This dataset has no free-text narratives, only structured fields (issue, "
        "company, dates, ...). If the user asks what people said, explain that and answer "
        "from the structured fields instead\n"
    )
    return f"""You are an AI assistant helping users understand {about}.

User Question: "{question}"

Available Context:
{context}

Instructions:
{instruction}

Guidelines:
- Be specific and cite numbers or {doc_label.lower()} IDs when available
- If the data shows trends, explain them clearly
- Keep your answer concise (2-4 sentences)
- If the context is insufficient, say so honestly
- Never make up information not present in the context
{text_note}
Answer:
"""
