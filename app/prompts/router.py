"""Router prompt for query classification."""


def get_router_prompt(question: str) -> str:
    """
    Generate router prompt for classifying queries.

    Returns the appropriate route: sql, rag, or hybrid
    """
    return f"""You are a query classifier for a complaint insights system. Analyze the user's question and classify it into one of three categories:

**SQL**: Questions about counts, rankings, trends, aggregations, or statistics
Examples:
- "What is the most common issue at Bank of America?"
- "How many complaints did Wells Fargo receive last year?"
- "Show me the top 5 banks by complaint volume"
- "What percentage of Chase complaints are about fraud?"

**RAG**: Questions about specific narratives, customer experiences, or qualitative content
Examples:
- "What are customers saying about Chase fraud cases?"
- "Show me examples of foreclosure complaints"
- "What do people complain about regarding credit card fees?"

**HYBRID**: Questions that need both statistical context AND narrative examples
Examples:
- "What is the top issue at JPMorgan and what are people saying about it?"
- "Which bank has the most fraud complaints and what are the details?"
- "Show me complaint trends for Wells Fargo and example narratives"

User Question: "{question}"

Respond with ONLY ONE WORD: sql, rag, or hybrid
"""
