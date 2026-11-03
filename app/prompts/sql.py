"""SQL generation prompt."""


def get_sql_prompt(
    question: str,
    table: str,
    dataset: str,
    schema: str,
    notes: str = "",
    examples: tuple[tuple[str, str], ...] = (),
    previous_error: str | None = None,
) -> str:
    """
    Build the text-to-SQL prompt.

    `schema` is the column listing produced by SQLTool.describe_table (types plus real
    values from the data); `notes` and `examples` come from the LOB registry, where
    examples use a {table} placeholder.
    """
    examples_text = "\n\n".join(
        f'Q: "{q}"\nSQL:\n{sql.strip().format(table=table)}' for q, sql in examples
    )
    examples_block = f"\nExample Queries:\n\n{examples_text}\n" if examples else ""
    notes_block = f"\nNotes:\n{notes}\n" if notes else ""
    retry_note = (
        f"\nYour previous query failed with this error, fix it:\n{previous_error}\n"
        if previous_error
        else ""
    )

    return f"""You are an expert SQL query generator for DuckDB. \
Generate a valid SQL query to answer the user's question.

Dataset: {dataset}
Table: {table}

Columns:
{schema}
{notes_block}{examples_block}
Rules:
1. Only use a single SELECT statement, and only query the table {table}
2. End the query with exactly ONE LIMIT clause (max 1000 rows); never write two LIMITs in a row. For "top N, then the highest of those" questions, put the first LIMIT inside a WITH subquery
3. Use DuckDB syntax, e.g. EXTRACT(YEAR FROM date_received), date_trunc('month', date_received)
4. Use the column values exactly as listed above (they are case-sensitive)
5. Return ONLY the SQL query, no explanations or markdown
{retry_note}
User Question: "{question}"

SQL Query:
"""
