"""
themes.py
---------
Thematic analysis for bank app reviews.

Approach
--------
1. Text cleaning  – lower-case, strip punctuation, remove stop-words,
                    optionally lemmatise with spaCy.
2. TF-IDF         – extract the most discriminating unigrams and bigrams
                    per bank.
3. Theme mapping  – map extracted keywords to one of five predefined themes
                    using a keyword → theme lookup table.  Reviews that match
                    no theme keyword are labelled "General Feedback".

Predefined themes and their seed keywords
------------------------------------------
| Theme                     | Representative keywords                        |
|---------------------------|------------------------------------------------|
| Account Access Issues     | login, password, otp, sign in, locked, account |
| Transaction Performance   | transfer, slow, crash, loading, transaction    |
| UI & Design               | interface, design, update, navigation, layout  |
| Customer Support          | support, service, response, agent, complaint   |
| Feature Requests          | fingerprint, biometric, feature, add, request  |

A review is assigned to the theme whose seed keywords appear most often
in the review text (majority vote over matched tokens).  Ties go to the
first matching theme in priority order.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme taxonomy
# ---------------------------------------------------------------------------

THEME_KEYWORDS: dict[str, list[str]] = {
    "Account Access Issues": [
        "login", "log in", "password", "otp", "sign in", "signin",
        "locked", "lock", "account", "register", "registration",
        "verification", "verify", "pin", "fingerprint login",
        "authentication", "authenticate", "two factor", "2fa",
    ],
    "Transaction Performance": [
        "transfer", "transaction", "payment", "slow", "crash", "crashing",
        "loading", "load", "freeze", "hang", "error", "bug", "fail",
        "failed", "network", "timeout", "delay", "stuck", "not working",
        "balance", "send money", "receive",
    ],
    "UI & Design": [
        "interface", "design", "layout", "navigation", "ui", "ux",
        "update", "upgrade", "look", "appearance", "screen", "button",
        "menu", "theme", "dark mode", "font", "color", "colour",
        "user friendly", "easy to use", "simple",
    ],
    "Customer Support": [
        "support", "service", "agent", "customer care", "response",
        "complaint", "help", "helpline", "call center", "call centre",
        "feedback", "issue resolved", "staff", "representative",
        "problem solved", "not helpful",
    ],
    "Feature Requests": [
        "fingerprint", "biometric", "face id", "feature", "add",
        "request", "wish", "want", "need", "budget", "budgeting",
        "statement", "notification", "alert", "schedule", "recurring",
        "multi account", "international", "forex",
    ],
}

FALLBACK_THEME = "General Feedback"

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

# Attempt spaCy lemmatisation; fall back to simple whitespace tokenisation
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy  # noqa: PLC0415

            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("spaCy en_core_web_sm loaded for lemmatisation.")
        except Exception as exc:
            logger.warning("spaCy unavailable (%s). Using simple tokenisation.", exc)
            _NLP = False  # sentinel: do not retry
    return _NLP


def clean_text(text: str) -> str:
    """Lower-case, remove punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, lemmatise: bool = True) -> list[str]:
    """
    Tokenise and optionally lemmatise *text*.

    Returns a list of tokens with stop-words removed.
    """
    nlp = _get_nlp() if lemmatise else False
    cleaned = clean_text(text)

    if nlp:
        doc = nlp(cleaned)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
    else:
        # Minimal stop-word list when spaCy is absent
        _STOP = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "is", "it", "this", "that", "i", "my",
            "me", "we", "you", "he", "she", "they", "be", "was", "are",
            "has", "have", "had", "do", "did", "not", "no", "so", "if",
            "as", "by", "up", "its", "can", "just", "get", "got",
        }
        tokens = [t for t in cleaned.split() if t not in _STOP and len(t) > 2]

    return tokens


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction
# ---------------------------------------------------------------------------

def extract_top_keywords(
    texts: list[str],
    top_n: int = 30,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[tuple[str, float]]:
    """
    Return the top *top_n* (term, tfidf_weight) pairs for a corpus.

    Uses TF-IDF over unigrams and bigrams (default).
    """
    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=5000,
        min_df=2,         # ignore terms appearing in fewer than 2 docs
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # Raised when corpus is too small (< min_df docs)
        vectorizer.set_params(min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    mean_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_scores.argsort()[::-1][:top_n]
    return [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_indices]


def extract_keywords_by_bank(
    df: pd.DataFrame,
    text_col: str = "review",
    bank_col: str = "bank",
    top_n: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """Return top keywords for each bank as a dict {bank: [(term, score), ...]}."""
    result: dict[str, list[tuple[str, float]]] = {}
    for bank, group in df.groupby(bank_col):
        texts = group[text_col].fillna("").astype(str).tolist()
        result[str(bank)] = extract_top_keywords(texts, top_n=top_n)
    return result


# ---------------------------------------------------------------------------
# Theme assignment
# ---------------------------------------------------------------------------

def _build_keyword_theme_index() -> dict[str, str]:
    """Invert THEME_KEYWORDS into a flat {keyword: theme} lookup."""
    index: dict[str, str] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        for kw in keywords:
            index[kw.lower()] = theme
    return index


_KW_INDEX = _build_keyword_theme_index()


def assign_theme(text: str) -> str:
    """
    Assign the most relevant predefined theme to *text*.

    Method: count how many seed keywords from each theme appear in the
    cleaned text, return the theme with the highest count.
    """
    cleaned = clean_text(text)
    votes: Counter = Counter()

    for keyword, theme in _KW_INDEX.items():
        if keyword in cleaned:
            votes[theme] += 1

    if not votes:
        return FALLBACK_THEME

    return votes.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_themes(
    df: pd.DataFrame,
    text_col: str = "review",
) -> pd.DataFrame:
    """
    Add an `identified_theme` column to *df*.

    Returns a new DataFrame (original is not mutated).
    """
    logger.info("Assigning themes to %d reviews …", len(df))
    df = df.copy()
    df["identified_theme"] = df[text_col].fillna("").astype(str).apply(assign_theme)
    theme_counts = df["identified_theme"].value_counts()
    logger.info("Theme distribution:\n%s", theme_counts.to_string())
    return df


def theme_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot table: rows = bank, columns = theme, values = review count.
    """
    return (
        df.groupby(["bank", "identified_theme"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def keyword_theme_examples(
    df: pd.DataFrame,
    text_col: str = "review",
    bank_col: str = "bank",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    For each bank and theme, return the top TF-IDF keywords extracted from
    reviews belonging to that theme.

    Returns a DataFrame with columns: bank, theme, keywords.
    """
    rows = []
    for bank, bank_group in df.groupby(bank_col):
        for theme, theme_group in bank_group.groupby("identified_theme"):
            texts = theme_group[text_col].fillna("").astype(str).tolist()
            kws = extract_top_keywords(texts, top_n=top_n, ngram_range=(1, 2))
            rows.append(
                {
                    "bank": bank,
                    "theme": theme,
                    "keywords": ", ".join(k for k, _ in kws[:10]),
                }
            )
    return pd.DataFrame(rows)
