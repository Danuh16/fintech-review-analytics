"""
sentiment.py
------------
Sentiment analysis for bank app reviews.

Strategy (two-tier):
  1. Primary  – DistilBERT fine-tuned on SST-2
       huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
       Chosen because it is compact (~67 M params), runs on CPU in reasonable
       time, and produces well-calibrated confidence scores.  SST-2 training
       maps well to short, opinionated user-review text.

  2. Fallback – VADER
       Used when the transformer model is unavailable (no internet, memory
       constraints).  VADER is rule-based and handles informal text well but
       cannot be fine-tuned.  It also natively produces a neutral class.

Label mapping:
  DistilBERT native:  POSITIVE / NEGATIVE
  → we add a NEUTRAL band when confidence < NEUTRAL_THRESHOLD (default 0.65)
    to bring it in line with VADER's three-class output.

  VADER compound score:  ≥ 0.05 → positive, ≤ -0.05 → negative, else neutral.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

# Confidence threshold below which a DistilBERT prediction is labelled neutral
NEUTRAL_THRESHOLD = 0.65

# Batch size for transformer inference (reduce if OOM on CPU)
TRANSFORMER_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# DistilBERT classifier
# ---------------------------------------------------------------------------

_PIPELINE = None  # cached pipeline instance


def _get_transformer_pipeline():
    """Lazy-load the HuggingFace pipeline (avoids import cost at module level)."""
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline  # noqa: PLC0415

        logger.info("Loading DistilBERT sentiment pipeline …")
        _PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        logger.info("Pipeline loaded.")
    return _PIPELINE


def _transformer_label(raw_label: str, score: float) -> SentimentLabel:
    """Convert HuggingFace raw label + score to our three-class label."""
    if score < NEUTRAL_THRESHOLD:
        return "neutral"
    return "positive" if raw_label.upper() == "POSITIVE" else "negative"


def classify_with_transformer(texts: list[str]) -> list[dict]:
    """
    Run DistilBERT on a list of texts.

    Returns a list of dicts: [{"label": str, "score": float}, ...]
    """
    pipe = _get_transformer_pipeline()
    results = []
    for i in range(0, len(texts), TRANSFORMER_BATCH_SIZE):
        batch = texts[i : i + TRANSFORMER_BATCH_SIZE]
        raw = pipe(batch)
        for r in raw:
            label = _transformer_label(r["label"], r["score"])
            results.append({"sentiment_label": label, "sentiment_score": round(r["score"], 4)})
    return results


# ---------------------------------------------------------------------------
# VADER fallback
# ---------------------------------------------------------------------------

def classify_with_vader(texts: list[str]) -> list[dict]:
    """
    Run VADER on a list of texts.

    Returns a list of dicts: [{"label": str, "score": float}, ...]
    where score is the compound score in [-1, 1].
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: PLC0415

    analyser = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        compound = analyser.polarity_scores(str(text))["compound"]
        if compound >= 0.05:
            label: SentimentLabel = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        # Normalise compound [-1,1] → [0,1] for a comparable "score" column
        results.append({"sentiment_label": label, "sentiment_score": round((compound + 1) / 2, 4)})
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_sentiment(
    df: pd.DataFrame,
    text_col: str = "review",
    use_transformer: bool = True,
) -> pd.DataFrame:
    """
    Add `sentiment_label` and `sentiment_score` columns to *df*.

    Parameters
    ----------
    df              : Input DataFrame (must contain *text_col*).
    text_col        : Column name that holds the review text.
    use_transformer : If True, attempt DistilBERT; fall back to VADER on error.

    Returns
    -------
    DataFrame with two new columns appended.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    logger.info("Running sentiment analysis on %d reviews …", len(texts))

    if use_transformer:
        try:
            results = classify_with_transformer(texts)
            method = "DistilBERT"
        except Exception as exc:
            logger.warning("Transformer inference failed (%s). Falling back to VADER.", exc)
            results = classify_with_vader(texts)
            method = "VADER (fallback)"
    else:
        results = classify_with_vader(texts)
        method = "VADER"

    logger.info("Sentiment analysis complete using %s.", method)

    df = df.copy()
    df["sentiment_label"] = [r["sentiment_label"] for r in results]
    df["sentiment_score"] = [r["sentiment_score"] for r in results]
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def sentiment_by_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean sentiment score and label distribution grouped by bank."""
    label_counts = (
        df.groupby(["bank", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .add_suffix("_count")
    )
    mean_score = df.groupby("bank")["sentiment_score"].mean().rename("mean_sentiment_score")
    return label_counts.join(mean_score).reset_index()


def sentiment_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean sentiment score and label distribution grouped by star rating."""
    label_counts = (
        df.groupby(["rating", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .add_suffix("_count")
    )
    mean_score = df.groupby("rating")["sentiment_score"].mean().rename("mean_sentiment_score")
    return label_counts.join(mean_score).reset_index()
