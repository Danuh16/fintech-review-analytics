"""
test_sentiment.py
-----------------
Unit tests for src/sentiment.py

Tests use VADER only (no network / model download needed in CI).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentiment import (
    _transformer_label,
    analyse_sentiment,
    classify_with_vader,
    sentiment_by_bank,
    sentiment_by_rating,
)


# ---------------------------------------------------------------------------
# classify_with_vader
# ---------------------------------------------------------------------------

class TestVaderClassification:
    def test_clearly_positive(self):
        results = classify_with_vader(["This app is absolutely fantastic and wonderful!"])
        assert results[0]["sentiment_label"] == "positive"
        assert 0.5 < results[0]["sentiment_score"] <= 1.0

    def test_clearly_negative(self):
        results = classify_with_vader(["This app is terrible and broken, I hate it!"])
        assert results[0]["sentiment_label"] == "negative"
        assert results[0]["sentiment_score"] < 0.5

    def test_neutral(self):
        results = classify_with_vader(["The app exists."])
        assert results[0]["sentiment_label"] == "neutral"

    def test_returns_all_rows(self):
        texts = ["good", "bad", "ok", "great", "awful"]
        results = classify_with_vader(texts)
        assert len(results) == len(texts)

    def test_score_in_unit_interval(self):
        results = classify_with_vader(["Works well.", "Crashes constantly."])
        for r in results:
            assert 0.0 <= r["sentiment_score"] <= 1.0

    def test_empty_string_handled(self):
        results = classify_with_vader([""])
        assert results[0]["sentiment_label"] in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# _transformer_label
# ---------------------------------------------------------------------------

class TestTransformerLabel:
    def test_high_confidence_positive(self):
        assert _transformer_label("POSITIVE", 0.95) == "positive"

    def test_high_confidence_negative(self):
        assert _transformer_label("NEGATIVE", 0.95) == "negative"

    def test_low_confidence_becomes_neutral(self):
        assert _transformer_label("POSITIVE", 0.50) == "neutral"
        assert _transformer_label("NEGATIVE", 0.60) == "neutral"

    def test_exactly_at_threshold_is_not_neutral(self):
        # score == NEUTRAL_THRESHOLD (0.65) is NOT below threshold
        assert _transformer_label("POSITIVE", 0.65) == "positive"


# ---------------------------------------------------------------------------
# analyse_sentiment (VADER path)
# ---------------------------------------------------------------------------

class TestAnalyseSentiment:
    def _sample_df(self):
        return pd.DataFrame(
            {
                "review": [
                    "Great app, very fast!",
                    "Terrible experience, keeps crashing.",
                    "It works.",
                    "Love the design.",
                    "Login fails every time.",
                ],
                "rating": [5, 1, 3, 4, 1],
                "bank": ["CBE", "BOA", "Dashen", "CBE", "BOA"],
            }
        )

    def test_adds_sentiment_columns(self):
        df = analyse_sentiment(self._sample_df(), use_transformer=False)
        assert "sentiment_label" in df.columns
        assert "sentiment_score" in df.columns

    def test_no_missing_labels(self):
        df = analyse_sentiment(self._sample_df(), use_transformer=False)
        assert df["sentiment_label"].notna().all()

    def test_coverage_100_percent(self):
        df = analyse_sentiment(self._sample_df(), use_transformer=False)
        coverage = df["sentiment_label"].notna().mean()
        assert coverage >= 0.90

    def test_does_not_mutate_input(self):
        original = self._sample_df()
        analyse_sentiment(original, use_transformer=False)
        assert "sentiment_label" not in original.columns

    def test_valid_labels_only(self):
        df = analyse_sentiment(self._sample_df(), use_transformer=False)
        assert set(df["sentiment_label"]).issubset({"positive", "negative", "neutral"})


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

class TestAggregations:
    def _enriched_df(self):
        df = pd.DataFrame(
            {
                "bank": ["CBE", "CBE", "BOA", "BOA", "Dashen"],
                "rating": [5, 1, 4, 2, 3],
                "sentiment_label": ["positive", "negative", "positive", "negative", "neutral"],
                "sentiment_score": [0.9, 0.1, 0.85, 0.15, 0.5],
            }
        )
        return df

    def test_sentiment_by_bank_has_bank_column(self):
        result = sentiment_by_bank(self._enriched_df())
        assert "bank" in result.columns
        assert "mean_sentiment_score" in result.columns

    def test_sentiment_by_bank_row_per_bank(self):
        result = sentiment_by_bank(self._enriched_df())
        assert len(result) == 3  # CBE, BOA, Dashen

    def test_sentiment_by_rating_has_rating_column(self):
        result = sentiment_by_rating(self._enriched_df())
        assert "rating" in result.columns
