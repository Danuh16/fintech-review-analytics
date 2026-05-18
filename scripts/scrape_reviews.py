"""
scrape_reviews.py
-----------------
Scrapes Google Play Store reviews for three Ethiopian bank mobile apps:
  - Commercial Bank of Ethiopia (CBE)
  - Bank of Abyssinia (BOA)
  - Dashen Bank

Output: data/raw/reviews_raw.csv  (excluded from version control via .gitignore)

Usage:
    python scripts/scrape_reviews.py
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

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
# App configuration
# ---------------------------------------------------------------------------
APPS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp",
}

# Minimum reviews to collect per bank
MIN_REVIEWS_PER_BANK = 400

# How many reviews to fetch per API call (library max is 200)
BATCH_SIZE = 200

# Polite delay between batches (seconds) to avoid rate-limiting
BATCH_DELAY = 2

# Output paths
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "reviews_raw.csv"


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def fetch_reviews_for_app(app_name: str, app_id: str, target: int) -> list[dict]:
    """
    Fetch at least `target` reviews for a single app using pagination.

    Returns a list of normalised review dicts with keys:
        review, rating, date, bank, source
    """
    collected: list[dict] = []
    continuation_token = None
    attempt = 0

    logger.info("Scraping %-35s (app_id=%s, target=%d)", app_name, app_id, target)

    while len(collected) < target:
        attempt += 1
        try:
            result, continuation_token = reviews(
                app_id,
                lang="en",
                country="et",  # Ethiopia
                sort=Sort.NEWEST,
                count=BATCH_SIZE,
                continuation_token=continuation_token,
            )
        except Exception as exc:
            logger.warning("Batch %d failed for %s: %s", attempt, app_name, exc)
            # Try once more with a longer delay before giving up
            if attempt <= 3:
                time.sleep(BATCH_DELAY * 3)
                continue
            logger.error("Giving up on %s after 3 failed attempts.", app_name)
            break

        if not result:
            logger.info("No more reviews available for %s after %d collected.", app_name, len(collected))
            break

        for r in result:
            collected.append(
                {
                    "review": r.get("content", ""),
                    "rating": r.get("score"),
                    "date": r.get("at"),
                    "bank": app_name,
                    "source": "Google Play",
                }
            )

        logger.info(
            "  batch %02d → %d new reviews (total so far: %d)",
            attempt,
            len(result),
            len(collected),
        )

        if continuation_token is None:
            logger.info("Reached end of available reviews for %s.", app_name)
            break

        time.sleep(BATCH_DELAY)

    if len(collected) < target:
        logger.warning(
            "Only %d reviews collected for %s (target was %d). "
            "This may be due to Play Store availability or rate limits.",
            len(collected),
            app_name,
            target,
        )
    else:
        logger.info("Collected %d reviews for %s.", len(collected), app_name)

    return collected


def scrape_all_apps() -> pd.DataFrame:
    """Scrape all three bank apps and return a combined raw DataFrame."""
    all_reviews: list[dict] = []

    for app_name, app_id in APPS.items():
        bank_reviews = fetch_reviews_for_app(app_name, app_id, MIN_REVIEWS_PER_BANK)
        all_reviews.extend(bank_reviews)
        # Extra pause between different apps
        time.sleep(BATCH_DELAY * 2)

    df = pd.DataFrame(all_reviews)
    logger.info("Total raw reviews collected: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_raw(df: pd.DataFrame) -> None:
    """Persist the raw DataFrame to CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Raw data saved to %s", OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    start = datetime.now()
    logger.info("=== Scraping started at %s ===", start.strftime("%Y-%m-%d %H:%M:%S"))

    df = scrape_all_apps()
    save_raw(df)

    elapsed = (datetime.now() - start).seconds
    logger.info("=== Scraping finished in %ds. Total records: %d ===", elapsed, len(df))

    # Quick summary by bank
    if not df.empty and "bank" in df.columns:
        summary = df.groupby("bank").size().rename("count")
        logger.info("\nReview counts by bank:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
