"""
analyse_sentiment_themes.py
---------------------------
End-to-end Task 2 pipeline:
  1. Load the clean review dataset (output of preprocess.py)
  2. Run sentiment analysis (DistilBERT → VADER fallback)
  3. Run thematic analysis (keyword-based theme assignment)
  4. Save enriched dataset to data/raw/reviews_analysed.csv

Output CSV columns:
    review_id, review_text, rating, date, bank, source,
    sentiment_label, sentiment_score, identified_theme

Usage:
    python scripts/analyse_sentiment_themes.py
    python scripts/analyse_sentiment_themes.py --vader   # force VADER (no GPU needed)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make src/ importable when running from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment import analyse_sentiment, sentiment_by_bank, sentiment_by_rating
from themes import analyse_themes, extract_keywords_by_bank, keyword_theme_examples, theme_summary

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
RAW_DIR = ROOT / "data" / "raw"
CLEAN_PATH = RAW_DIR / "reviews_clean.csv"
OUTPUT_PATH = RAW_DIR / "reviews_analysed.csv"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Run 'python scripts/preprocess.py' first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d clean reviews from %s", len(df), path)
    return df


def add_review_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable integer review_id column (1-based)."""
    df = df.copy()
    df.insert(0, "review_id", range(1, len(df) + 1))
    return df


def run_pipeline(use_transformer: bool = True) -> pd.DataFrame:
    start = datetime.now()
    logger.info("=== Task 2 pipeline started at %s ===", start.strftime("%Y-%m-%d %H:%M:%S"))

    # 1. Load
    df = load_clean()
    df = add_review_id(df)

    # 2. Sentiment
    df = analyse_sentiment(df, text_col="review", use_transformer=use_transformer)

    # 3. Themes
    df = analyse_themes(df, text_col="review")

    # 4. Rename for output spec
    df = df.rename(columns={"review": "review_text"})

    # Select and order output columns
    out_cols = [
        "review_id", "review_text", "rating", "date", "bank", "source",
        "sentiment_label", "sentiment_score", "identified_theme",
    ]
    df = df[out_cols]

    # 5. Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Enriched dataset saved to %s  (%d rows)", OUTPUT_PATH, len(df))

    # 6. Summary reports
    elapsed = (datetime.now() - start).seconds
    logger.info("=== Pipeline finished in %ds ===", elapsed)

    logger.info("\n--- Sentiment by bank ---\n%s", sentiment_by_bank(df).to_string(index=False))
    logger.info("\n--- Sentiment by rating ---\n%s", sentiment_by_rating(df).to_string(index=False))
    logger.info("\n--- Theme distribution ---\n%s", theme_summary(df).to_string(index=False))

    # Top keywords per bank (logged for quick inspection)
    df_renamed = df.rename(columns={"review_text": "review"})
    kw_by_bank = extract_keywords_by_bank(df_renamed, top_n=15)
    for bank, kws in kw_by_bank.items():
        top_terms = ", ".join(k for k, _ in kws[:10])
        logger.info("Top keywords – %s: %s", bank, top_terms)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 2 sentiment & theme pipeline.")
    parser.add_argument(
        "--vader",
        action="store_true",
        help="Force VADER sentiment (skip transformer; useful without GPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(use_transformer=not args.vader)
