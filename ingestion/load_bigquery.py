"""
BigQuery loader for cleaned CFPB complaint data.

- Reads data/processed/complaints.parquet (output of clean.py)
- Creates the dataset if needed
- Loads the complaints table, partitioned by date_received and clustered by company, product
- Creates and fills the company_aliases lookup table

Usage (from the project root, with BIGQUERY_PROJECT_ID set and `gcloud auth
application-default login` done, or GOOGLE_APPLICATION_CREDENTIALS pointing at a key):
    python -m ingestion.load_bigquery
"""

import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from app.config import CLEANED_COMPLAINTS_PATH, settings

logger = logging.getLogger(__name__)

COMPLAINTS_SCHEMA = [
    bigquery.SchemaField("complaint_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("date_received", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("date_sent_to_company", "DATE"),
    bigquery.SchemaField("product", "STRING"),
    bigquery.SchemaField("sub_product", "STRING"),
    bigquery.SchemaField("issue", "STRING"),
    bigquery.SchemaField("sub_issue", "STRING"),
    bigquery.SchemaField("consumer_complaint_narrative", "STRING"),
    bigquery.SchemaField("company", "STRING"),
    bigquery.SchemaField("state", "STRING"),
    bigquery.SchemaField("zip_code", "STRING"),
    bigquery.SchemaField("submitted_via", "STRING"),
    bigquery.SchemaField("company_response", "STRING"),
    bigquery.SchemaField("timely_response", "STRING"),
    bigquery.SchemaField("company_public_response", "STRING"),
    bigquery.SchemaField("tags", "STRING"),
]

ALIASES_TABLE = "company_aliases"

# Informal name -> legal name as it appears in the CFPB data.
COMPANY_ALIASES = {
    "Chase": "JPMORGAN CHASE & CO.",
    "JPMorgan": "JPMORGAN CHASE & CO.",
    "JP Morgan": "JPMORGAN CHASE & CO.",
    "Bank of America": "BANK OF AMERICA, NATIONAL ASSOCIATION",
    "BofA": "BANK OF AMERICA, NATIONAL ASSOCIATION",
    "Wells Fargo": "WELLS FARGO & COMPANY",
    "Capital One": "CAPITAL ONE FINANCIAL CORPORATION",
    "Citi": "CITIBANK, N.A.",
    "Citibank": "CITIBANK, N.A.",
    "Amex": "AMERICAN EXPRESS COMPANY",
    "American Express": "AMERICAN EXPRESS COMPANY",
    "Synchrony": "SYNCHRONY FINANCIAL",
    "Discover": "DISCOVER BANK",
    "Goldman Sachs": "GOLDMAN SACHS BANK USA",
    "Apple Card": "GOLDMAN SACHS BANK USA",
    "Equifax": "EQUIFAX, INC.",
    "TransUnion": "TRANSUNION INTERMEDIATE HOLDINGS, INC.",
    "Experian": "Experian Information Solutions Inc.",
    "Mr. Cooper": "Mr. Cooper Group Inc.",
    "Rocket Mortgage": "Rocket Mortgage, LLC",
    "Shellpoint": "Shellpoint Partners, LLC",
}


def create_dataset(client: bigquery.Client, dataset_id: str, location: str = "US") -> None:
    """Create the BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{client.project}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        logger.info("Dataset %s already exists", dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        logger.info("Created dataset %s", dataset_ref)


def load_complaints(
    client: bigquery.Client, dataset_id: str, table_id: str, df: pd.DataFrame
) -> None:
    """Replace the complaints table with `df` (partitioned + clustered)."""
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    job_config = bigquery.LoadJobConfig(
        schema=COMPLAINTS_SCHEMA,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH, field="date_received"
        ),
        clustering_fields=["company", "product"],
    )
    client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
    logger.info("Loaded %d rows into %s", len(df), table_ref)


def load_company_aliases(client: bigquery.Client, dataset_id: str) -> None:
    """Replace the company_aliases lookup table."""
    table_ref = f"{client.project}.{dataset_id}.{ALIASES_TABLE}"
    aliases = pd.DataFrame(
        [{"alias": alias, "canonical_name": name} for alias, name in COMPANY_ALIASES.items()]
    )
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("alias", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("canonical_name", "STRING", mode="REQUIRED"),
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_dataframe(aliases, table_ref, job_config=job_config).result()
    logger.info("Loaded %d aliases into %s", len(aliases), table_ref)


def main(input_path: Path = CLEANED_COMPLAINTS_PATH) -> None:
    """Load cleaned complaints and the alias table into BigQuery."""
    if not settings.bigquery_project_id:
        raise SystemExit("BIGQUERY_PROJECT_ID is not set (see .env.example)")
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found. Run `python -m ingestion.clean` first.")

    df = pd.read_parquet(input_path)
    client = bigquery.Client(project=settings.bigquery_project_id)

    create_dataset(client, settings.bigquery_dataset)
    load_complaints(client, settings.bigquery_dataset, settings.bigquery_table, df)
    load_company_aliases(client, settings.bigquery_dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
