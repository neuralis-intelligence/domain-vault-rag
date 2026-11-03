"""SQL generation prompt."""


def get_sql_prompt(
    question: str,
    table: str,
    previous_error: str | None = None,
) -> str:
    """Build the text-to-SQL prompt with schema, real column values, and examples."""

    schema = f"""
Table: {table}

Schema:
- complaint_id (VARCHAR): Unique complaint identifier
- date_received (DATE): Date the CFPB received the complaint
- date_sent_to_company (DATE): Date forwarded to the company
- product (VARCHAR): Exactly one of 'Credit card' or 'Mortgage'
- sub_product (VARCHAR): Product subtype
- issue (VARCHAR): Main issue category
- sub_issue (VARCHAR): Issue detail
- consumer_complaint_narrative (VARCHAR): Customer's written complaint (often NULL)
- company (VARCHAR): Company legal name, UPPER or mixed case (see below)
- state (VARCHAR): U.S. state abbreviation
- zip_code (VARCHAR): 5-digit ZIP code
- submitted_via (VARCHAR): Submission channel (Web, Referral, Phone, ...)
- company_response (VARCHAR): Company's response to the consumer
- timely_response (VARCHAR): 'Yes' or 'No'
- company_public_response (VARCHAR): Company's optional public statement
- tags (VARCHAR): e.g. 'Older American', 'Servicemember'

Company names are legal names, e.g.:
- 'JPMORGAN CHASE & CO.' (Chase, JPMorgan)
- 'BANK OF AMERICA, NATIONAL ASSOCIATION'
- 'WELLS FARGO & COMPANY'
- 'CAPITAL ONE FINANCIAL CORPORATION'
- 'CITIBANK, N.A.'
- 'AMERICAN EXPRESS COMPANY'
- 'SYNCHRONY FINANCIAL'
When the user gives a short or informal name, match with
UPPER(company) LIKE '%KEYWORD%' instead of exact equality.

Common credit card issues:
- 'Problem with a purchase shown on your statement'
- 'Getting a credit card'
- 'Fees or interest'
- 'Closing your account'

Common mortgage issues:
- 'Trouble during payment process'
- 'Struggling to pay mortgage'
- 'Applying for a mortgage or refinancing an existing mortgage'
- 'Closing on a mortgage'
"""

    examples = f"""
Example Queries:

Q: "What is the most common issue at Bank of America?"
SQL:
SELECT issue, COUNT(*) AS complaint_count
FROM {table}
WHERE UPPER(company) LIKE '%BANK OF AMERICA%'
GROUP BY issue
ORDER BY complaint_count DESC
LIMIT 1

Q: "How many credit card complaints did Chase receive in 2024?"
SQL:
SELECT COUNT(*) AS total_complaints
FROM {table}
WHERE company = 'JPMORGAN CHASE & CO.'
  AND product = 'Credit card'
  AND EXTRACT(YEAR FROM date_received) = 2024
LIMIT 1

Q: "Top 5 companies by complaint volume"
SQL:
SELECT company, COUNT(*) AS complaint_count
FROM {table}
GROUP BY company
ORDER BY complaint_count DESC
LIMIT 5
"""

    retry_note = (
        f"\nYour previous query failed with this error, fix it:\n{previous_error}\n"
        if previous_error
        else ""
    )

    return f"""You are an expert SQL query generator for DuckDB. \
Generate a valid SQL query to answer the user's question.

{schema}

{examples}

Rules:
1. Only use a single SELECT statement
2. Always include a LIMIT clause (max 1000 rows)
3. Use DuckDB syntax, e.g. EXTRACT(YEAR FROM date_received), date_trunc('month', date_received)
4. Query the table as {table} (no backticks, no dataset prefix)
5. Return ONLY the SQL query, no explanations or markdown
{retry_note}
User Question: "{question}"

SQL Query:
"""
