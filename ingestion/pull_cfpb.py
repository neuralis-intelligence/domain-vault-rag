"""
Download CFPB complaints into data/raw/, one CSV per product.

Pulls one month at a time from the CFPB public API (with 429 backoff) and writes
data/raw/<key>.csv for each product in CFPB_PRODUCTS (e.g. data/raw/credit_card.csv).

Usage (from the project root):
    python -m ingestion.pull_cfpb                                   # every product
    python -m ingestion.pull_cfpb --products student_loan vehicle_loan
"""

import argparse
import time
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import requests

from app.config import RAW_DIR

# File key (data/raw/<key>.csv) -> product name exactly as the CFPB API spells it.
# To add a product, add it here and to config/lobs.toml.
CFPB_PRODUCTS = {
    "credit_card": "Credit card",
    "mortgage": "Mortgage",
    "student_loan": "Student loan",
    "vehicle_loan": "Vehicle loan or lease",
    "checking_savings": "Checking or savings account",
}

# ---------------------------------------------------------
# CFPB API configuration
# ---------------------------------------------------------

BASE_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

START_DATE = date(2023, 8, 1)

# Use tomorrow because the API's max date is used as the upper boundary
END_DATE = date.today() + timedelta(days=1)


# ---------------------------------------------------------
# Get the first day of the next month
# ---------------------------------------------------------


def get_next_month(current_date):

    if current_date.month == 12:
        return date(current_date.year + 1, 1, 1)

    return date(
        current_date.year,
        current_date.month + 1,
        1,
    )


# ---------------------------------------------------------
# Generate monthly date ranges
# ---------------------------------------------------------


def month_ranges(start_date, end_date):

    current = start_date

    while current < end_date:
        next_month = get_next_month(current)

        chunk_end = min(next_month, end_date)

        yield current, chunk_end

        current = chunk_end


# ---------------------------------------------------------
# Download one month of complaints
# ---------------------------------------------------------


def download_chunk(
    session,
    product,
    start_date,
    end_date,
    max_retries=6,
):

    params = {
        "product": product,
        "date_received_min": start_date.isoformat(),
        "date_received_max": end_date.isoformat(),
        "format": "csv",
    }

    print(f"Downloading {product}: {start_date} -> {end_date}")

    for attempt in range(max_retries):
        response = session.get(
            BASE_URL,
            params=params,
            timeout=120,
        )

        # CFPB is rate-limiting us
        if response.status_code == 429:
            # If CFPB tells us how long to wait, use that.
            retry_after = response.headers.get("Retry-After")

            # Otherwise exponential backoff: 30s, 60s, 120s, 240s, capped at 300s.
            wait_time = int(retry_after) if retry_after else min(30 * 2**attempt, 300)

            print(f"Rate limited (429). Waiting {wait_time} seconds before retrying...")

            time.sleep(wait_time)
            continue

        # Raise errors for other unsuccessful responses
        response.raise_for_status()

        df = pd.read_csv(
            BytesIO(response.content),
            low_memory=False,
        )

        print(f"Downloaded {len(df):,} rows")

        return df

    raise RuntimeError(
        f"Failed to download {product} for {start_date} -> {end_date} after {max_retries} retries."
    )


# ---------------------------------------------------------
# Download all months for one product
# ---------------------------------------------------------


def pull_product(product):

    frames = []

    with requests.Session() as session:
        for start_date, end_date in month_ranges(
            START_DATE,
            END_DATE,
        ):
            df = download_chunk(
                session=session,
                product=product,
                start_date=start_date,
                end_date=end_date,
            )

            if not df.empty:
                frames.append(df)

            # Small pause between API requests
            time.sleep(3)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    # Remove accidental duplicate complaints
    result = result.drop_duplicates(subset=["Complaint ID"])

    return result


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------


NARRATIVE_COLUMN = "Consumer complaint narrative"


def loses_narratives(path, new_df) -> bool:
    """True if `path` holds narratives that writing `new_df` over it would throw away."""
    if not path.exists() or NARRATIVE_COLUMN in new_df.columns:
        return False
    header = pd.read_csv(path, nrows=0).columns
    return NARRATIVE_COLUMN in header


def main():
    parser = argparse.ArgumentParser(description="Download CFPB complaints into data/raw/")
    parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(CFPB_PRODUCTS),
        default=list(CFPB_PRODUCTS),
        help="Product keys to download (default: all)",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for key in args.products:
        product = CFPB_PRODUCTS[key]
        print(f"\nPulling {product} complaints...\n")
        df = pull_product(product)

        path = RAW_DIR / f"{key}.csv"
        if loses_narratives(path, df):
            path = RAW_DIR / f"{key}.no_narratives.csv"
            print(
                f"WARNING: data/raw/{key}.csv has complaint narratives and this download has "
                f"none (the CFPB no longer publishes them). Kept the old file; saved the new "
                f"download to {path} instead."
            )
        df.to_csv(path, index=False)
        print(f"{product}: {len(df):,} rows saved to {path}")

    print("\nDownload complete. Next: python -m ingestion.clean")


if __name__ == "__main__":
    main()
