"""Answer synthesis prompt."""


def get_synthesis_prompt(question: str, context: str, route: str) -> str:
    """
    Generate prompt for synthesizing final answer from context.
    """
    route_instructions = {
        "sql": "Use the SQL query results to provide a clear, concise answer with specific numbers.",
        "rag": "Synthesize information from the complaint narratives, citing complaint IDs when relevant.",
        "hybrid": "Combine the statistical insights from SQL with specific examples from the narratives."
    }

    instruction = route_instructions.get(route, "Provide a clear answer based on the available data.")

    return f"""You are an AI assistant helping users understand CFPB bank complaint data.

User Question: "{question}"

Available Context:
{context}

Instructions:
{instruction}

Guidelines:
- Be specific and cite numbers or complaint IDs when available
- If the data shows trends, explain them clearly
- Keep your answer concise (2-4 sentences)
- If the context is insufficient, say so honestly
- Never make up information not present in the context

Answer:
"""
