"""
preprocess.py
-------------
Cleans and normalises the raw review dataset produced by scrape_reviews.py.

Steps:
  1. Load raw CSV from data/raw/reviews_raw.csv
  2. Remove exact duplicate rows
  3. Drop rows missing review text or rating
  4. Normalise the date column to YYYY-MM-DD (string)
  5. Enforce correct column dtypes
  6. Save the cleaned dataset to data/raw/reviews_clean.csv

Output columns: review, rating, date, bank, source

Usage:
    python scripts/preprocess.py
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_PATH = RAW_DIR / "reviews_raw.csv"
CLEAN_PATH = RAW_DIR / "reviews_clean.csv"

# Expected final columns in order
FINAL_COLUMNS = ["review", "rating", "date", "bank", "source"]


# ---------------------------------------------------------------------------
# Pipeline steps (each returns a DataFrame + optional stats dict)
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Run 'python scripts/scrape_reviews.py' first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows (same review text, rating, date, and bank).
    Returns (cleaned_df, n_removed).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["review", "rating", "date", "bank"])
    removed = before - len(df)
    logger.info("Duplicates removed: %d  (remaining: %d)", removed, len(df))
    return df, removed


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows where review text or rating is missing.
    Returns (cleaned_df, counts_dict).
    """
    stats: dict = {}

    missing_review = df["review"].isna().sum()
    missing_rating = df["rating"].isna().sum()
    stats["missing_review"] = int(missing_review)
    stats["missing_rating"] = int(missing_rating)

    df = df.dropna(subset=["review", "rating"])
    # Also drop rows with empty review strings
    empty_review = (df["review"].astype(str).str.strip() == "").sum()
    df = df[df["review"].astype(str).str.strip() != ""]
    stats["empty_review_strings"] = int(empty_review)

    logger.info(
        "Missing values – review text: %d, rating: %d, empty strings: %d",
        missing_review,
        missing_rating,
        empty_review,
    )
    logger.info("Rows remaining after missing-value handling: %d", len(df))
    return df, stats


def normalise_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the 'date' column to YYYY-MM-DD string format.

    google-play-scraper returns datetime objects or ISO strings; this step
    handles both and any other common formats gracefully.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    n_invalid = df["date"].isna().sum()
    if n_invalid > 0:
        logger.warning("%d rows have unparseable dates and will be dropped.", n_invalid)
        df = df.dropna(subset=["date"])

    logger.info("Dates normalised to YYYY-MM-DD format.")
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their expected types."""
    df = df.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["review"] = df["review"].astype(str).str.strip()
    df["bank"] = df["bank"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    # Drop any rows where rating coercion failed
    invalid_ratings = df["rating"].isna().sum()
    if invalid_ratings > 0:
        logger.warning("Dropping %d rows with non-numeric ratings.", invalid_ratings)
        df = df.dropna(subset=["rating"])
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the five required columns in the canonical order."""
    missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
    return df[FINAL_COLUMNS].reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> None:
    """Save the cleaned DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cleaned data saved to %s  (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline.

    Returns the cleaned DataFrame (useful for unit tests and notebooks).
    """
    start = datetime.now()
    logger.info("=== Preprocessing started at %s ===", start.strftime("%Y-%m-%d %H:%M:%S"))

    df = load_raw(raw_path)
    df, n_dups = remove_duplicates(df)
    df, missing_stats = handle_missing(df)
    df = normalise_dates(df)
    df = enforce_dtypes(df)
    df = select_columns(df)

    save_clean(df, clean_path)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    elapsed = (datetime.now() - start).seconds
    logger.info("=== Preprocessing finished in %ds ===", elapsed)
    logger.info("\n--- Dataset summary ---")
    logger.info("Total clean reviews : %d", len(df))
    logger.info("Missing-data rate   : %.2f%%", (1 - len(df) / max(1, len(df) + n_dups + sum(missing_stats.values()))) * 100)

    if "bank" in df.columns:
        bank_summary = df.groupby("bank").agg(
            count=("review", "count"),
            avg_rating=("rating", "mean"),
        )
        logger.info("\nPer-bank summary:\n%s", bank_summary.to_string())

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
