"""
BigQuery data loader for CFPB complaint data.

This script:
- Loads cleaned complaint data from clean.py output
- Creates BigQuery dataset and table if not exists
- Configures partitioning (by date_received) and clustering (by company, product)
- Uploads data to BigQuery
- Creates company_aliases lookup table
"""

import os
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset(client: bigquery.Client, dataset_id: str, location: str = "US"):
    """Create BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{client.project}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_ref} already exists.")
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_ref}")


def create_complaints_table(client: bigquery.Client, dataset_id: str, table_id: str):
    """
    Create the complaints table with proper schema, partitioning, and clustering.
    """
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    schema = [
        bigquery.SchemaField("complaint_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date_received", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("product", "STRING"),
        bigquery.SchemaField("sub_product", "STRING"),
        bigquery.SchemaField("issue", "STRING"),
        bigquery.SchemaField("sub_issue", "STRING"),
        bigquery.SchemaField("consumer_complaint_narrative", "STRING"),
        bigquery.SchemaField("company", "STRING"),
        bigquery.SchemaField("state", "STRING"),
        bigquery.SchemaField("zip_code", "STRING"),
        bigquery.SchemaField("submitted_via", "STRING"),
        bigquery.SchemaField("date_sent_to_company", "DATE"),
        bigquery.SchemaField("company_response", "STRING"),
        bigquery.SchemaField("timely_response", "STRING"),
        bigquery.SchemaField("consumer_disputed", "STRING"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    # Partition by date_received
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="date_received",
    )

    # Cluster by company and product for efficient queries
    table.clustering_fields = ["company", "product"]

    try:
        table = client.create_table(table)
        logger.info(f"Created table {table_ref} with partitioning and clustering")
    except Exception as e:
        logger.warning(f"Table creation skipped: {e}")


def load_data_to_bigquery(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    dataframe: pd.DataFrame
):
    """Upload DataFrame to BigQuery table."""
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    job = client.load_table_from_dataframe(dataframe, table_ref, job_config=job_config)
    job.result()  # Wait for job to complete

    logger.info(f"Loaded {len(dataframe)} rows to {table_ref}")


def create_company_aliases_table(client: bigquery.Client, dataset_id: str):
    """
    Create company_aliases table for normalizing company name variations.
    Example: "JP Morgan", "Chase" → "JPMORGAN CHASE & CO."
    """
    table_id = "company_aliases"
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    schema = [
        bigquery.SchemaField("alias", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("canonical_name", "STRING", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    try:
        table = client.create_table(table)
        logger.info(f"Created table {table_ref}")

        # TODO: Populate with actual alias mappings
        # Example data:
        # aliases_df = pd.DataFrame([
        #     {"alias": "Chase", "canonical_name": "JPMORGAN CHASE & CO."},
        #     {"alias": "JP Morgan", "canonical_name": "JPMORGAN CHASE & CO."},
        #     # ... more aliases
        # ])
        # load_data_to_bigquery(client, dataset_id, table_id, aliases_df)

    except Exception as e:
        logger.warning(f"Company aliases table creation skipped: {e}")


def main():
    """Main BigQuery loading pipeline."""
    logger.info("Starting BigQuery loading pipeline...")

    # Get configuration from environment
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET", "cfpb_complaints")
    table_id = os.getenv("BIGQUERY_TABLE", "complaints")

    if not project_id:
        logger.error("BIGQUERY_PROJECT_ID environment variable not set")
        return

    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)

    # Create dataset
    create_dataset(client, dataset_id)

    # Create complaints table
    create_complaints_table(client, dataset_id, table_id)

    # Create company aliases table
    create_company_aliases_table(client, dataset_id)

    # TODO: Load cleaned data
    # cleaned_data_path = Path(__file__).parent.parent / "data" / "cleaned_complaints.csv"
    # if cleaned_data_path.exists():
    #     df = pd.read_csv(cleaned_data_path)
    #     load_data_to_bigquery(client, dataset_id, table_id, df)
    # else:
    #     logger.warning("No cleaned data found. Run clean.py first.")

    logger.info("BigQuery loading pipeline completed.")


if __name__ == "__main__":
    main()
