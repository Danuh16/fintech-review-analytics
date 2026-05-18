# scripts/

Production-ready scripts for each pipeline stage.

| Script                        | Task   | Description                                                |
| ----------------------------- | ------ | ---------------------------------------------------------- |
| `scrape_reviews.py`           | Task 1 | Scrape Google Play Store reviews for CBE, BOA, and Dashen  |
| `preprocess.py`               | Task 1 | Clean, deduplicate, and normalise the raw review dataset   |
| `analyse_sentiment_themes.py` | Task 2 | Sentiment analysis (DistilBERT / VADER) + theme assignment |
| `load_to_postgres.py`         | Task 3 | Load cleaned/analysed reviews into PostgreSQL              |

Run scripts from the project root:

```bash
# Task 1
python scripts/scrape_reviews.py
python scripts/preprocess.py

# Task 2  (use --vader flag to skip the transformer model download)
python scripts/analyse_sentiment_themes.py
python scripts/analyse_sentiment_themes.py --vader

# Task 3
python scripts/load_to_postgres.py
```
