"""
DuckDB loader for cleaned CFPB complaint data.

- Reads data/processed/complaints.parquet (output of clean.py)
- Builds the `complaints` table with an explicit schema
- Creates and fills the `company_aliases` lookup table

The database is built in a temporary file and then swapped into place, so a running
API keeps answering from the old copy. Restart the API to pick up the new data.

Usage (from the project root):
    python -m ingestion.load_duckdb
"""

import logging
import os
from pathlib import Path

import duckdb

from app.config import CLEANED_COMPLAINTS_PATH, settings

logger = logging.getLogger(__name__)

# Column -> DuckDB type. Order and names must match FINAL_COLUMNS in clean.py.
COMPLAINTS_SCHEMA = {
    "complaint_id": "VARCHAR NOT NULL",
    "date_received": "DATE NOT NULL",
    "date_sent_to_company": "DATE",
    "product": "VARCHAR",
    "sub_product": "VARCHAR",
    "issue": "VARCHAR",
    "sub_issue": "VARCHAR",
    "consumer_complaint_narrative": "VARCHAR",
    "company": "VARCHAR",
    "state": "VARCHAR",
    "zip_code": "VARCHAR",
    "submitted_via": "VARCHAR",
    "company_response": "VARCHAR",
    "timely_response": "VARCHAR",
    "company_public_response": "VARCHAR",
    "tags": "VARCHAR",
}

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


def load_complaints(con: duckdb.DuckDBPyConnection, table: str, parquet_path: Path) -> int:
    """Create `table` from the cleaned Parquet file, casting each column to its schema type."""
    columns = ",\n    ".join(f"{name} {dtype}" for name, dtype in COMPLAINTS_SCHEMA.items())
    con.execute(f"CREATE OR REPLACE TABLE {table} (\n    {columns}\n)")

    casts = ", ".join(
        f"CAST({name} AS {dtype.split()[0]})" for name, dtype in COMPLAINTS_SCHEMA.items()
    )
    con.execute(f"INSERT INTO {table} SELECT {casts} FROM read_parquet(?)", [str(parquet_path)])

    (count,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    logger.info("Loaded %d rows into %s", count, table)
    return count


def load_company_aliases(con: duckdb.DuckDBPyConnection) -> None:
    """Replace the company_aliases lookup table."""
    con.execute(
        f"CREATE OR REPLACE TABLE {ALIASES_TABLE} "
        "(alias VARCHAR NOT NULL, canonical_name VARCHAR NOT NULL)"
    )
    con.executemany(f"INSERT INTO {ALIASES_TABLE} VALUES (?, ?)", list(COMPANY_ALIASES.items()))
    logger.info("Loaded %d aliases into %s", len(COMPANY_ALIASES), ALIASES_TABLE)


def build_database(parquet_path: Path, db_path: Path, table: str) -> int:
    """Build the database at `db_path` from `parquet_path`; returns the complaint row count."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    with duckdb.connect(str(tmp_path)) as con:
        count = load_complaints(con, table, parquet_path)
        load_company_aliases(con)

    os.replace(tmp_path, db_path)
    logger.info("Wrote %s", db_path)
    return count


def main(input_path: Path = CLEANED_COMPLAINTS_PATH) -> None:
    """Load cleaned complaints and the alias table into DuckDB."""
    if not input_path.exists():
        raise SystemExit(f"{input_path} not found. Run `python -m ingestion.clean` first.")
    build_database(input_path, settings.duckdb_path, settings.sql_table)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
