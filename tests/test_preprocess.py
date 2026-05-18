"""
test_preprocess.py
------------------
Unit tests for the preprocessing pipeline defined in scripts/preprocess.py.

Run with:
    pytest tests/test_preprocess.py -v
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

# Make scripts/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from preprocess import (
    enforce_dtypes,
    handle_missing,
    normalise_dates,
    remove_duplicates,
    select_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Helper: build a minimal valid DataFrame, override fields via kwargs."""
    base = {
        "review": ["Great app!", "Crashes a lot", "Decent UI", "Slow login"],
        "rating": [5, 1, 3, 2],
        "date": [
            "2024-01-10",
            "2024-02-15",
            "2024-03-20",
            "2024-04-05",
        ],
        "bank": ["CBE", "BOA", "Dashen", "CBE"],
        "source": ["Google Play"] * 4,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_exact_duplicates_removed(self):
        df = _make_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add one duplicate
        cleaned, n_removed = remove_duplicates(df)
        assert n_removed == 1
        assert len(cleaned) == len(df) - 1

    def test_no_duplicates_unchanged(self):
        df = _make_df()
        cleaned, n_removed = remove_duplicates(df)
        assert n_removed == 0
        assert len(cleaned) == len(df)

    def test_all_duplicates_removed(self):
        df = _make_df()
        all_dup = pd.concat([df.iloc[[0]]] * 5, ignore_index=True)
        cleaned, n_removed = remove_duplicates(all_dup)
        assert len(cleaned) == 1
        assert n_removed == 4


# ---------------------------------------------------------------------------
# handle_missing
# ---------------------------------------------------------------------------

class TestHandleMissing:
    def test_drops_missing_review(self):
        df = _make_df()
        df.loc[0, "review"] = None
        cleaned, stats = handle_missing(df)
        assert stats["missing_review"] == 1
        assert len(cleaned) == len(df) - 1

    def test_drops_missing_rating(self):
        df = _make_df()
        df.loc[1, "rating"] = None
        cleaned, stats = handle_missing(df)
        assert stats["missing_rating"] == 1
        assert len(cleaned) == len(df) - 1

    def test_drops_empty_string_review(self):
        df = _make_df()
        df.loc[2, "review"] = "   "
        cleaned, stats = handle_missing(df)
        assert stats["empty_review_strings"] == 1
        assert len(cleaned) == len(df) - 1

    def test_no_missing_unchanged(self):
        df = _make_df()
        cleaned, stats = handle_missing(df)
        assert len(cleaned) == len(df)
        assert stats["missing_review"] == 0
        assert stats["missing_rating"] == 0


# ---------------------------------------------------------------------------
# normalise_dates
# ---------------------------------------------------------------------------

class TestNormaliseDates:
    def test_iso_string_format(self):
        df = _make_df(date=["2024-01-10", "2024-02-15", "2024-03-20", "2024-04-05"])
        cleaned = normalise_dates(df)
        assert all(cleaned["date"].str.match(r"\d{4}-\d{2}-\d{2}"))

    def test_datetime_object_converted(self):
        from datetime import datetime, timezone

        dt_list = [
            datetime(2024, 1, 10, tzinfo=timezone.utc),
            datetime(2024, 2, 15, tzinfo=timezone.utc),
            datetime(2024, 3, 20, tzinfo=timezone.utc),
            datetime(2024, 4, 5, tzinfo=timezone.utc),
        ]
        df = _make_df(date=dt_list)
        cleaned = normalise_dates(df)
        assert cleaned["date"].iloc[0] == "2024-01-10"
        assert cleaned["date"].iloc[3] == "2024-04-05"

    def test_invalid_dates_dropped(self):
        df = _make_df(date=["2024-01-10", "not-a-date", "2024-03-20", "2024-04-05"])
        cleaned = normalise_dates(df)
        assert len(cleaned) == 3  # one invalid row dropped


# ---------------------------------------------------------------------------
# enforce_dtypes
# ---------------------------------------------------------------------------

class TestEnforceDtypes:
    def test_rating_cast_to_integer(self):
        df = _make_df(rating=["5", "1", "3", "2"])
        cleaned = enforce_dtypes(df)
        assert pd.api.types.is_integer_dtype(cleaned["rating"])

    def test_review_stripped(self):
        df = _make_df(review=["  Great app!  ", "Crashes  ", " Decent UI", "Slow login "])
        cleaned = enforce_dtypes(df)
        assert cleaned["review"].iloc[0] == "Great app!"
        assert cleaned["review"].iloc[1] == "Crashes"

    def test_invalid_rating_dropped(self):
        df = _make_df(rating=[5, 1, "abc", 2])
        cleaned = enforce_dtypes(df)
        assert len(cleaned) == 3


# ---------------------------------------------------------------------------
# select_columns
# ---------------------------------------------------------------------------

class TestSelectColumns:
    def test_returns_five_columns_in_order(self):
        df = _make_df()
        result = select_columns(df)
        assert list(result.columns) == ["review", "rating", "date", "bank", "source"]

    def test_extra_columns_dropped(self):
        df = _make_df()
        df["extra"] = "noise"
        result = select_columns(df)
        assert "extra" not in result.columns

    def test_missing_required_column_raises(self):
        df = _make_df().drop(columns=["bank"])
        with pytest.raises(ValueError, match="missing required columns"):
            select_columns(df)
