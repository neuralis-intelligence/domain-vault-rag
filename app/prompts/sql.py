"""SQL generation prompt."""


def get_sql_prompt(question: str, dataset_id: str, table_id: str) -> str:
    """
    Generate text-to-SQL prompt with schema and examples.
    """
    schema = f"""
Table: `{dataset_id}.{table_id}`

Schema:
- complaint_id (STRING): Unique complaint identifier
- date_received (DATE): Date complaint was received
- product (STRING): Product type (e.g., "Credit card", "Mortgage")
- sub_product (STRING): Specific product subtype
- issue (STRING): Main issue category
- sub_issue (STRING): Specific issue detail
- consumer_complaint_narrative (STRING): Customer's written complaint
- company (STRING): Company name (e.g., "JPMORGAN CHASE & CO.", "BANK OF AMERICA, N.A.")
- state (STRING): U.S. state abbreviation
- zip_code (STRING): ZIP code
- submitted_via (STRING): Submission channel
- date_sent_to_company (DATE): Date forwarded to company
- company_response (STRING): Company's response to complaint
- timely_response (STRING): "Yes" or "No"
- consumer_disputed (STRING): "Yes", "No", or NULL

Common Issues:
- "Problem with a purchase shown on your statement"
- "Managing an account"
- "Trouble using your card"
- "Problem with a credit reporting company's investigation"
- "Closing your account"
- "Incorrect information on your report"

Common Products:
- "Credit card or prepaid card"
- "Mortgage"
- "Checking or savings account"
- "Credit reporting, credit repair services, or other personal consumer reports"
"""

    examples = """
Example Queries:

Q: "What is the most common issue at Bank of America?"
SQL:
SELECT issue, COUNT(*) as count
FROM `{dataset}.{table}`
WHERE company = 'BANK OF AMERICA, N.A.'
GROUP BY issue
ORDER BY count DESC
LIMIT 1

Q: "How many credit card complaints did Chase receive in 2023?"
SQL:
SELECT COUNT(*) as total_complaints
FROM `{dataset}.{table}`
WHERE company = 'JPMORGAN CHASE & CO.'
  AND product = 'Credit card or prepaid card'
  AND EXTRACT(YEAR FROM date_received) = 2023

Q: "Top 5 companies by complaint volume"
SQL:
SELECT company, COUNT(*) as complaint_count
FROM `{dataset}.{table}`
GROUP BY company
ORDER BY complaint_count DESC
LIMIT 5
"""

    return f"""You are an expert SQL query generator for BigQuery. Generate a valid SQL query to answer the user's question.

{schema}

{examples.format(dataset=dataset_id, table=table_id)}

Rules:
1. Only use SELECT statements
2. Always include a LIMIT clause (max 1000 rows)
3. Use proper BigQuery syntax for dates and functions
4. Use backticks for table references: `{dataset_id}.{table_id}`
5. Return ONLY the SQL query, no explanations

User Question: "{question}"

SQL Query:
"""
