from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
import time

import pandas as pd
import requests


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

# This points to:
# banking-qa-agent/data/
DATA_DIR = Path(__file__).resolve().parent

# This points to:
# banking-qa-agent/data/raw/
RAW_DIR = DATA_DIR / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CFPB API configuration
# ---------------------------------------------------------

BASE_URL = (
    "https://www.consumerfinance.gov/"
    "data-research/consumer-complaints/search/api/v1/"
)

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

    print(
        f"Downloading {product}: "
        f"{start_date} -> {end_date}"
    )

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

            if retry_after:
                wait_time = int(retry_after)
            else:
                # Exponential backoff:
                # 30 sec, 60 sec, 120 sec, 240 sec...
                wait_time = min(
                    30 * (2 ** attempt),
                    300,
                )

            print(
                f"Rate limited (429). "
                f"Waiting {wait_time} seconds before retrying..."
            )

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
        f"Failed to download {product} "
        f"for {start_date} -> {end_date} "
        f"after {max_retries} retries."
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
    result = result.drop_duplicates(
        subset=["Complaint ID"]
    )

    return result


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():

    print("\nPulling Credit Card complaints...\n")

    credit_card = pull_product(
        "Credit card"
    )

    print("\nPulling Mortgage complaints...\n")

    mortgage = pull_product(
        "Mortgage"
    )


    # Save Credit Card data
    credit_card.to_csv(
        RAW_DIR / "credit_card.csv",
        index=False,
    )


    # Save Mortgage data
    mortgage.to_csv(
        RAW_DIR / "mortgage.csv",
        index=False,
    )


    print("\nDownload complete.")

    print(
        f"Credit Card rows: "
        f"{len(credit_card):,}"
    )

    print(
        f"Mortgage rows: "
        f"{len(mortgage):,}"
    )

    print(
        f"\nCredit Card saved to:\n"
        f"{RAW_DIR / 'credit_card.csv'}"
    )

    print(
        f"\nMortgage saved to:\n"
        f"{RAW_DIR / 'mortgage.csv'}"
    )


if __name__ == "__main__":
    main()