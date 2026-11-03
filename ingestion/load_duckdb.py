"""
Build the DuckDB database: one table per line of business in config/lobs.toml.

- Each LOB's table is created from its `source` Parquet file (optionally filtered by `where`)
- LOBs whose source file doesn't exist yet are skipped with a warning
- The `company_aliases` lookup table is created for the CFPB data

The database is built in a temporary file and then swapped into place, so a running
API keeps answering from the old copy. Restart the API to pick up the new data.

Usage (from the project root):
    python -m ingestion.load_duckdb
"""

import glob
import logging
import os
from pathlib import Path

import duckdb

from app.config import settings
from app.lob import LineOfBusiness, Registry, get_registry

logger = logging.getLogger(__name__)

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


def source_exists(lob: LineOfBusiness) -> bool:
    """True if the LOB's source path (or glob) matches at least one file."""
    return bool(glob.glob(str(lob.source)))


def load_lob_table(con: duckdb.DuckDBPyConnection, lob: LineOfBusiness) -> int:
    """Create the LOB's table from its source Parquet; returns the row count."""
    where = f" WHERE {lob.where}" if lob.where else ""
    con.execute(
        f"CREATE OR REPLACE TABLE {lob.table} AS SELECT * FROM read_parquet(?){where}",
        [str(lob.source)],
    )
    (count,) = con.execute(f"SELECT COUNT(*) FROM {lob.table}").fetchone()
    if count:
        logger.info("%-18s %9d rows -> %s", lob.id, count, lob.table)
    else:
        logger.warning(
            "%s: 0 rows in %s (check `where` / that the data was pulled)", lob.id, lob.table
        )
    return count


def load_company_aliases(con: duckdb.DuckDBPyConnection) -> None:
    """Replace the company_aliases lookup table."""
    con.execute(
        f"CREATE OR REPLACE TABLE {ALIASES_TABLE} "
        "(alias VARCHAR NOT NULL, canonical_name VARCHAR NOT NULL)"
    )
    con.executemany(f"INSERT INTO {ALIASES_TABLE} VALUES (?, ?)", list(COMPANY_ALIASES.items()))
    logger.info("Loaded %d aliases into %s", len(COMPANY_ALIASES), ALIASES_TABLE)


def build_database(registry: Registry, db_path: Path) -> dict[str, int]:
    """Build the database at `db_path`; returns {lob_id: row_count} for the loaded LOBs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    counts: dict[str, int] = {}
    with duckdb.connect(str(tmp_path)) as con:
        for lob in registry.lobs.values():
            if not source_exists(lob):
                logger.warning("%s: skipped, %s not found", lob.id, lob.source)
                continue
            counts[lob.id] = load_lob_table(con, lob)
        load_company_aliases(con)

    if not counts:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit("No LOB source files found. Run `python -m ingestion.clean` first.")

    os.replace(tmp_path, db_path)
    logger.info("Wrote %s", db_path)
    return counts


def main() -> None:
    build_database(get_registry(), settings.duckdb_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
