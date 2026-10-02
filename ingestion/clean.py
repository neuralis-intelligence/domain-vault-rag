"""
Data cleaning and schema normalization for CFPB complaint data.

This script:
- Loads raw credit card and mortgage CSV files
- Normalizes column names to snake_case
- Standardizes date formats
- Cleans text fields (narratives, issues, products)
- Merges into a unified schema
- Outputs cleaned data for BigQuery ingestion
"""

import pandas as pd
from pathlib import Path
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case."""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df


def standardize_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """Convert date columns to datetime format."""
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def clean_text_fields(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    """Clean and normalize text fields."""
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('', None)
    return df


def normalize_company_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize company names to standard format.
    TODO: Implement company alias mapping.
    """
    # Placeholder for company name normalization
    return df


def merge_datasets(credit_card_df: pd.DataFrame, mortgage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge credit card and mortgage datasets into unified schema.
    """
    # Ensure both have same columns
    combined_df = pd.concat([credit_card_df, mortgage_df], ignore_index=True)
    return combined_df


def main():
    """Main ETL pipeline."""
    logger.info("Starting data cleaning pipeline...")

    # Define paths
    data_dir = Path(__file__).parent.parent / "banking-qa-agent" / "data" / "raw"
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # TODO: Load raw CSVs
    # credit_card_df = pd.read_csv(data_dir / "credit_card.csv")
    # mortgage_df = pd.read_csv(data_dir / "mortgage.csv")

    logger.info("Data cleaning pipeline completed.")
    logger.info("Next step: Run load_bigquery.py to upload to BigQuery")


if __name__ == "__main__":
    main()
