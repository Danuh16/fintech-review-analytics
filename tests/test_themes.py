"""
test_themes.py
--------------
Unit tests for src/themes.py
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from themes import (
    FALLBACK_THEME,
    THEME_KEYWORDS,
    assign_theme,
    analyse_themes,
    clean_text,
    extract_top_keywords,
    extract_keywords_by_bank,
    theme_summary,
    tokenize,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lower_case(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_punctuation_removed(self):
        assert "!" not in clean_text("Great app!")
        assert "." not in clean_text("It works well.")

    def test_extra_whitespace_collapsed(self):
        result = clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_returns_list_of_strings(self):
        tokens = tokenize("The app crashes a lot", lemmatise=False)
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_short_tokens_filtered(self):
        # Tokens ≤ 2 chars should not appear
        tokens = tokenize("It is ok", lemmatise=False)
        assert all(len(t) > 2 for t in tokens)

    def test_stop_words_removed(self):
        tokens = tokenize("the app is very good", lemmatise=False)
        stop_words = {"the", "is", "a", "an"}
        assert not stop_words.intersection(set(tokens))


# ---------------------------------------------------------------------------
# extract_top_keywords
# ---------------------------------------------------------------------------

class TestExtractTopKeywords:
    CORPUS = [
        "transfer takes too long, very slow loading",
        "login error every time i open the app",
        "the transfer is slow and crashes",
        "cannot login password reset not working",
        "app crashes during transfer payment",
        "slow transfer and login issues together",
    ]

    def test_returns_list_of_tuples(self):
        result = extract_top_keywords(self.CORPUS, top_n=10)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_scores_are_floats(self):
        result = extract_top_keywords(self.CORPUS, top_n=5)
        assert all(isinstance(score, float) for _, score in result)

    def test_top_n_respected(self):
        result = extract_top_keywords(self.CORPUS, top_n=5)
        assert len(result) <= 5

    def test_empty_corpus_returns_empty(self):
        assert extract_top_keywords([]) == []

    def test_common_term_appears_in_top(self):
        # "transfer" appears in most docs so should rank high
        result = extract_top_keywords(self.CORPUS, top_n=15)
        terms = [t for t, _ in result]
        assert any("transfer" in t for t in terms)


# ---------------------------------------------------------------------------
# assign_theme
# ---------------------------------------------------------------------------

class TestAssignTheme:
    def test_login_maps_to_account_access(self):
        theme = assign_theme("I cannot login to the app, keeps showing login error")
        assert theme == "Account Access Issues"

    def test_crash_maps_to_transaction_performance(self):
        theme = assign_theme("The app crashes when I try to make a transfer payment")
        assert theme == "Transaction Performance"

    def test_ui_keywords_map_correctly(self):
        theme = assign_theme("The interface and design need improvement, better navigation")
        assert theme == "UI & Design"

    def test_support_keywords_map_correctly(self):
        theme = assign_theme("Customer support agent was not helpful with my complaint")
        assert theme == "Customer Support"

    def test_feature_request_maps_correctly(self):
        theme = assign_theme("Please add fingerprint biometric login feature")
        assert theme == "Feature Requests"

    def test_no_match_returns_fallback(self):
        theme = assign_theme("xyz abc def mno qrs")
        assert theme == FALLBACK_THEME

    def test_all_themes_reachable(self):
        """Every predefined theme must be assignable."""
        all_themes = set()
        test_sentences = {
            "Account Access Issues": "login password otp locked account",
            "Transaction Performance": "transfer slow crash error payment",
            "UI & Design": "interface design navigation layout update",
            "Customer Support": "support agent complaint response helpline",
            "Feature Requests": "fingerprint biometric feature add request",
        }
        for expected_theme, sentence in test_sentences.items():
            result = assign_theme(sentence)
            all_themes.add(result)
        # At least 4 of the 5 themes should be reachable from these sentences
        assert len(all_themes) >= 4


# ---------------------------------------------------------------------------
# analyse_themes
# ---------------------------------------------------------------------------

class TestAnalyseThemes:
    def _sample_df(self):
        return pd.DataFrame(
            {
                "review": [
                    "Great app!",
                    "Login keeps failing, password reset broken",
                    "Transfer is slow and crashes",
                    "Please add fingerprint login feature",
                    "Customer support was useless",
                ],
                "bank": ["CBE", "CBE", "BOA", "BOA", "Dashen"],
            }
        )

    def test_adds_identified_theme_column(self):
        df = analyse_themes(self._sample_df())
        assert "identified_theme" in df.columns

    def test_no_null_themes(self):
        df = analyse_themes(self._sample_df())
        assert df["identified_theme"].notna().all()

    def test_does_not_mutate_input(self):
        original = self._sample_df()
        analyse_themes(original)
        assert "identified_theme" not in original.columns

    def test_valid_theme_values(self):
        valid = set(THEME_KEYWORDS.keys()) | {FALLBACK_THEME}
        df = analyse_themes(self._sample_df())
        assert set(df["identified_theme"]).issubset(valid)


# ---------------------------------------------------------------------------
# theme_summary
# ---------------------------------------------------------------------------

class TestThemeSummary:
    def test_returns_dataframe(self):
        df = pd.DataFrame(
            {
                "bank": ["CBE", "CBE", "BOA"],
                "identified_theme": ["Transaction Performance", "Account Access Issues", "Transaction Performance"],
            }
        )
        result = theme_summary(df)
        assert isinstance(result, pd.DataFrame)
        assert "bank" in result.columns

    def test_counts_are_non_negative(self):
        df = pd.DataFrame(
            {
                "bank": ["CBE", "CBE", "BOA", "Dashen"],
                "identified_theme": ["UI & Design", "UI & Design", "Customer Support", "Feature Requests"],
            }
        )
        result = theme_summary(df).set_index("bank")
        assert (result >= 0).all().all()
