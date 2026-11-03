"""
Data cleaning and schema normalization for CFPB complaint data.

Mirrors the cleaning steps prototyped in notebooks/*_exploration.ipynb:
- Load every raw CFPB product CSV in data/raw/ (see CFPB_PRODUCTS in pull_cfpb.py)
- Rename columns to snake_case
- Parse dates, clean text fields and ZIP codes
- Drop duplicate complaint IDs
- Write the unified table to data/processed/complaints.parquet

Usage (from the project root):
    python -m ingestion.clean
"""

import logging
from pathlib import Path

import pandas as pd

from app.config import CLEANED_COMPLAINTS_PATH, RAW_DIR
from ingestion.pull_cfpb import CFPB_PRODUCTS

logger = logging.getLogger(__name__)


COLUMN_MAPPING = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "consumer_complaint_narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
    "Complaint ID": "complaint_id",
}

DATE_COLUMNS = ("date_received", "date_sent_to_company")

TEXT_COLUMNS = (
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company_public_response",
    "company",
    "state",
    "submitted_via",
    "company_response",
    "timely_response",
    "tags",
)

# Final column order of the cleaned complaints table.
FINAL_COLUMNS = [
    "complaint_id",
    "date_received",
    "date_sent_to_company",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company",
    "state",
    "zip_code",
    "submitted_via",
    "company_response",
    "timely_response",
    "company_public_response",
    "tags",
]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CFPB headers to snake_case."""
    return df.rename(columns=COLUMN_MAPPING)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ISO timestamps to calendar dates (SQL DATE)."""
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.date
    return df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and turn empty strings into nulls."""
    for col in TEXT_COLUMNS:
        if col in df.columns:
            cleaned = df[col].astype("string").str.strip()
            df[col] = cleaned.mask(cleaned == "")
    return df


def clean_zip_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first 5 digits; masked (e.g. '123XX') or short ZIPs become null."""
    zips = df["zip_code"].astype("string").str.strip()
    digits = zips.str.replace(r"\D", "", regex=True)
    valid = ~zips.str.contains("XX", case=False, na=True) & (digits.str.len() >= 5)
    df["zip_code"] = digits.str[:5].where(valid)
    return df


def clean_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to one raw CFPB DataFrame."""
    df = rename_columns(df)
    df = parse_dates(df)
    df = clean_text_fields(df)
    df = clean_zip_codes(df)
    df["complaint_id"] = df["complaint_id"].astype("string")

    df = df.dropna(subset=["complaint_id", "date_received"])
    df = df.drop_duplicates(subset=["complaint_id"], keep="first")
    return df[FINAL_COLUMNS].reset_index(drop=True)


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate every raw product CSV that has been downloaded."""
    frames = []
    for key in CFPB_PRODUCTS:
        path = raw_dir / f"{key}.csv"
        if not path.exists():
            logger.warning(
                "%s not found, skipping (python -m ingestion.pull_cfpb --products %s)", path, key
            )
            continue
        logger.info("Loading %s", path)
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""]))
    if not frames:
        raise FileNotFoundError(f"No raw CSVs in {raw_dir}. Run `python -m ingestion.pull_cfpb`.")
    return pd.concat(frames, ignore_index=True)


def main(output_path: Path = CLEANED_COMPLAINTS_PATH) -> None:
    """Run the cleaning pipeline end to end."""
    raw = load_raw()
    logger.info("Loaded %d raw rows", len(raw))

    cleaned = clean_complaints(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)

    logger.info(
        "Wrote %d rows (%d with narratives) to %s",
        len(cleaned),
        cleaned["consumer_complaint_narrative"].notna().sum(),
        output_path,
    )
    logger.info("Next: python -m ingestion.load_duckdb and python -m ingestion.embed_narratives")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
