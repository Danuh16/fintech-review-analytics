# Fintech Review Analytics

Customer Experience Analytics for Ethiopian Bank Mobile Apps — a four-part data engineering and NLP pipeline built for Omega Consultancy.

---

## Project Overview

This project scrapes, processes, and analyses Google Play Store reviews for three Ethiopian bank mobile applications:

| Bank | App Name | Play Store ID |
|---|---|---|
| Commercial Bank of Ethiopia | CBE Mobile | `com.combanketh.mobilebanking` |
| Bank of Abyssinia | BOA Mobile Banking | `com.boa.boaMobileBanking` |
| Dashen Bank | Dashen Super App | `com.dashen.dashensuperapp` |

The pipeline covers four tasks:

1. **Data Collection & Preprocessing** — scrape and clean reviews (this branch)
2. **Sentiment & Thematic Analysis** — NLP classification and theme extraction
3. **Database Engineering** — PostgreSQL schema design and population
4. **Insights & Recommendations** — visualisations and bank-specific recommendations

---

## Repository Structure

```
fintech-review-analytics/
├── .github/workflows/unittests.yml   # CI/CD: runs tests on every push
├── .vscode/settings.json
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/                          # Gitignored — never committed
├── notebooks/
│   └── README.md
├── scripts/
│   ├── scrape_reviews.py             # Task 1: scraping
│   ├── preprocess.py                 # Task 1: preprocessing
│   ├── analyse_sentiment_themes.py   # Task 2: sentiment + themes
│   └── load_to_postgres.py           # Task 3: PostgreSQL loader
├── sql/
│   └── schema.sql                    # Task 3: relational schema + verification queries
├── src/                              # Reusable modules (Tasks 2–4)
└── tests/
    └── test_preprocess.py            # Unit tests for preprocessing pipeline
```

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/<your-org>/fintech-review-analytics.git
cd fintech-review-analytics

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Scrape reviews (writes data/raw/reviews_raw.csv)
python scripts/scrape_reviews.py

# 5. Preprocess (writes data/raw/reviews_clean.csv)
python scripts/preprocess.py

# 6. Run unit tests
pytest tests/ -v

# 7. Load to PostgreSQL (Task 3)
python scripts/load_to_postgres.py
```

---

## Task 1: Data Collection & Preprocessing

### Scraping Methodology

Reviews are collected using the [`google-play-scraper`](https://github.com/JoMingyu/google-play-scraper) Python library, which calls the unofficial Google Play Store API without requiring an API key.

**Configuration:**
- **Language / region:** `en` / `et` (English, Ethiopia country code)
- **Sort order:** Newest first (`Sort.NEWEST`)
- **Batch size:** 200 reviews per API call (library maximum)
- **Target:** ≥ 400 reviews per bank (1,200 total)
- **Rate limiting:** 2-second delay between batches; 4-second delay between apps
- **Retry logic:** Up to 3 retries per failed batch with exponential back-off

**Fields collected:**

| Column | Description |
|---|---|
| `review` | Raw user review text |
| `rating` | Star rating (1–5) |
| `date` | Review posting date (YYYY-MM-DD) |
| `bank` | Bank/app name |
| `source` | Always `"Google Play"` |

### Known Limitations

- The library wraps an **unofficial API** that may return fewer than the requested count if the Play Store limits results for a given locale/language combination. If fewer than 400 reviews are returned, the date range is effectively broadened by requesting newer batches until the API is exhausted.
- Reviews are only available in English. Ethiopian users who write in Amharic are under-represented in this dataset.
- Review dates reflect when the review was posted, not the app version being reviewed.
- The Google Play Store does not expose a public official API; therefore scraping results may vary over time.

### Preprocessing Steps

1. **Load** raw CSV from `data/raw/reviews_raw.csv`
2. **Remove duplicates** — drop rows with identical `(review, rating, date, bank)` tuples
3. **Handle missing values** — drop rows missing `review` or `rating`; log counts
4. **Normalise dates** — coerce all date representations to `YYYY-MM-DD` string format; drop rows with unparseable dates
5. **Enforce dtypes** — cast `rating` to integer, strip whitespace from text columns
6. **Select columns** — output exactly `[review, rating, date, bank, source]`

### Data Quality KPIs (Task 1 targets)

| KPI | Target | Status |
|---|---|---|
| Total reviews | ≥ 1,200 | Verified at runtime |
| Reviews per bank | ≥ 400 | Verified at runtime |
| Missing data rate | < 5% | Logged during preprocessing |
| Output columns | 5 exact | Enforced by `select_columns()` |

---

## CI/CD

GitHub Actions workflow (`.github/workflows/unittests.yml`) runs on every push to `main` and all task branches:

1. Check out code
2. Set up Python 3.11
3. Cache pip dependencies
4. `pip install -r requirements.txt`
5. `pytest tests/ -v --tb=short --cov=src`

---

## Task 3: Store Cleaned Data in PostgreSQL

### 1) Install and Create Database

Install PostgreSQL locally, then create the target database:

```sql
CREATE DATABASE bank_reviews;
```

### 2) Configure Connection

The loader reads standard PostgreSQL environment variables:

- `PGHOST` (default: `localhost`)
- `PGPORT` (default: `5432`)
- `PGDATABASE` (default: `bank_reviews`)
- `PGUSER` (default: `postgres`)
- `PGPASSWORD` (default: `postgres`)

Example (Windows PowerShell):

```powershell
$env:PGHOST="localhost"
$env:PGPORT="5432"
$env:PGDATABASE="bank_reviews"
$env:PGUSER="postgres"
$env:PGPASSWORD="your_password"
```

### 3) Apply Schema + Insert Data

Run the Task 3 loader:

```bash
python scripts/load_to_postgres.py
```

The script will:

1. Read `data/raw/reviews_analysed.csv` (or fallback to `data/raw/reviews_clean.csv`)
2. Apply SQL schema from `sql/schema.sql`
3. Upsert banks into `banks`
4. Insert/update reviews into `reviews`
5. Run verification queries and print results

### 4) Verification Queries

Verification SQL is included in `sql/schema.sql` comments:

1. Count reviews per bank
2. Compute average rating per bank
3. Check nulls in key fields (`review_text`, `rating`, `review_date`, `source`)

### KPI Coverage

- Working DB connection + insert script: Implemented in `scripts/load_to_postgres.py`
- Schema file committed: `sql/schema.sql`
- Data volume requirement (>1,000) is satisfied when Task 1/2 datasets are loaded
- Verification queries executed by the loader after insertion

---

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code only |
| `task-1` | Data collection & preprocessing |
| `task-2` | Sentiment & thematic analysis |
| `task-3` | Database engineering |
| `task-4` | Insights & recommendations |

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

---

## License

MIT
