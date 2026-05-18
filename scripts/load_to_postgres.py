"""
load_to_postgres.py
-------------------
Task 3 loader: persist processed fintech app reviews into PostgreSQL.

Loads data from:
  1) data/raw/reviews_analysed.csv (preferred, contains sentiment/theme)
  2) data/raw/reviews_clean.csv    (fallback, sentiment/theme set to NULL)

Environment variables (optional):
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Defaults:
  host=localhost, port=5432, database=bank_reviews, user=postgres

Usage:
  python scripts/load_to_postgres.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
ANALYSED_PATH = RAW_DIR / "reviews_analysed.csv"
CLEAN_PATH = RAW_DIR / "reviews_clean.csv"
SCHEMA_PATH = ROOT / "sql" / "schema.sql"

APP_NAME_MAP = {
    "Commercial Bank of Ethiopia": "CBE Mobile",
    "Bank of Abyssinia": "BOA Mobile Banking",
    "Dashen Bank": "Dashen Super App",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_connection_url() -> str:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "bank_reviews")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine() -> Engine:
    url = build_connection_url()
    return create_engine(url, pool_pre_ping=True)


def choose_input_file() -> Path:
    if ANALYSED_PATH.exists():
        return ANALYSED_PATH
    if CLEAN_PATH.exists():
        return CLEAN_PATH
    raise FileNotFoundError(
        f"No input CSV found. Expected one of:\n- {ANALYSED_PATH}\n- {CLEAN_PATH}"
    )


def load_input_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "review_text" not in df.columns and "review" in df.columns:
        df = df.rename(columns={"review": "review_text"})

    if "review_id" not in df.columns:
        df.insert(0, "review_id", range(1, len(df) + 1))

    for col in ["sentiment_label", "sentiment_score", "identified_theme"]:
        if col not in df.columns:
            df[col] = None

    required = [
        "review_id",
        "review_text",
        "rating",
        "date",
        "bank",
        "source",
        "sentiment_label",
        "sentiment_score",
        "identified_theme",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    df = df[required].copy()
    df["review_id"] = pd.to_numeric(df["review_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Minimal safety cleanup before insert
    df = df.dropna(subset=["review_id", "review_text", "rating", "bank", "source"])
    df = df[df["review_text"].astype(str).str.strip() != ""]
    df["review_id"] = df["review_id"].astype("int64")
    df["rating"] = df["rating"].astype("int16")
    df["review_text"] = df["review_text"].astype(str)
    df["bank"] = df["bank"].astype(str)
    df["source"] = df["source"].astype(str)
    return df


def apply_schema(engine: Engine) -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    # Use raw DBAPI execution to support full SQL files with multiple statements.
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(schema_sql)
        raw_conn.commit()
    finally:
        raw_conn.close()
    logger.info("Schema applied from %s", SCHEMA_PATH)


def upsert_banks(engine: Engine, bank_names: list[str]) -> dict[str, int]:
    with engine.begin() as conn:
        for bank_name in sorted(set(bank_names)):
            app_name = APP_NAME_MAP.get(bank_name, f"{bank_name} App")
            conn.execute(
                text(
                    """
                    INSERT INTO banks (bank_name, app_name)
                    VALUES (:bank_name, :app_name)
                    ON CONFLICT (bank_name)
                    DO UPDATE SET app_name = EXCLUDED.app_name
                    """
                ),
                {"bank_name": bank_name, "app_name": app_name},
            )

        rows = conn.execute(text("SELECT bank_id, bank_name FROM banks")).mappings().all()
    bank_map = {row["bank_name"]: row["bank_id"] for row in rows}
    logger.info("Bank dimension ready with %d rows.", len(bank_map))
    return bank_map


def insert_reviews(engine: Engine, df: pd.DataFrame, bank_map: dict[str, int]) -> int:
    rows = []
    for rec in df.to_dict(orient="records"):
        bank_id = bank_map.get(rec["bank"])
        if bank_id is None:
            continue
        rows.append(
            {
                "review_id": int(rec["review_id"]),
                "bank_id": int(bank_id),
                "review_text": rec["review_text"],
                "rating": int(rec["rating"]),
                "review_date": rec["date"],
                "sentiment_label": rec["sentiment_label"],
                "sentiment_score": None
                if pd.isna(rec["sentiment_score"])
                else float(rec["sentiment_score"]),
                "identified_theme": rec["identified_theme"],
                "source": rec["source"],
            }
        )

    sql = text(
        """
        INSERT INTO reviews (
            review_id, bank_id, review_text, rating, review_date,
            sentiment_label, sentiment_score, identified_theme, source
        ) VALUES (
            :review_id, :bank_id, :review_text, :rating, :review_date,
            :sentiment_label, :sentiment_score, :identified_theme, :source
        )
        ON CONFLICT (review_id)
        DO UPDATE SET
            bank_id = EXCLUDED.bank_id,
            review_text = EXCLUDED.review_text,
            rating = EXCLUDED.rating,
            review_date = EXCLUDED.review_date,
            sentiment_label = EXCLUDED.sentiment_label,
            sentiment_score = EXCLUDED.sentiment_score,
            identified_theme = EXCLUDED.identified_theme,
            source = EXCLUDED.source
        """
    )

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def run_verification_queries(engine: Engine) -> None:
    q_counts = text(
        """
        SELECT b.bank_name, COUNT(*) AS review_count
        FROM reviews r
        JOIN banks b ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY review_count DESC
        """
    )

    q_avg_rating = text(
        """
        SELECT b.bank_name, ROUND(AVG(r.rating)::numeric, 2) AS avg_rating
        FROM reviews r
        JOIN banks b ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY avg_rating DESC
        """
    )

    q_nulls = text(
        """
        SELECT
          SUM(CASE WHEN review_text IS NULL OR review_text = '' THEN 1 ELSE 0 END) AS null_review_text,
          SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS null_rating,
          SUM(CASE WHEN review_date IS NULL THEN 1 ELSE 0 END) AS null_review_date,
          SUM(CASE WHEN source IS NULL OR source = '' THEN 1 ELSE 0 END) AS null_source
        FROM reviews
        """
    )

    with engine.begin() as conn:
        counts = conn.execute(q_counts).mappings().all()
        avg_rating = conn.execute(q_avg_rating).mappings().all()
        nulls = conn.execute(q_nulls).mappings().one()

    logger.info("\\nVerification: review count per bank")
    for row in counts:
        logger.info("  - %s: %s", row["bank_name"], row["review_count"])

    logger.info("\\nVerification: average rating per bank")
    for row in avg_rating:
        logger.info("  - %s: %s", row["bank_name"], row["avg_rating"])

    logger.info("\\nVerification: null checks")
    logger.info("  - null_review_text: %s", nulls["null_review_text"])
    logger.info("  - null_rating: %s", nulls["null_rating"])
    logger.info("  - null_review_date: %s", nulls["null_review_date"])
    logger.info("  - null_source: %s", nulls["null_source"])


def main() -> None:
    input_path = choose_input_file()
    logger.info("Using input file: %s", input_path)

    df = load_input_dataframe(input_path)
    logger.info("Prepared %d records for DB insertion.", len(df))

    engine = get_engine()
    apply_schema(engine)

    bank_map = upsert_banks(engine, df["bank"].tolist())
    inserted = insert_reviews(engine, df, bank_map)
    logger.info("Inserted/updated %d reviews into PostgreSQL.", inserted)

    run_verification_queries(engine)


if __name__ == "__main__":
    main()
