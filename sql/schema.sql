-- PostgreSQL schema for Task 3: bank_reviews
-- Usage:
--   psql -U postgres -d bank_reviews -f sql/schema.sql

BEGIN;

CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(120) NOT NULL UNIQUE,
    app_name VARCHAR(160) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id BIGINT PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE RESTRICT,
    review_text TEXT NOT NULL,
    rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score NUMERIC(6, 4),
    identified_theme VARCHAR(120),
    source VARCHAR(40) NOT NULL DEFAULT 'Google Play',
    ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_review_date ON reviews(review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment_label ON reviews(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_reviews_identified_theme ON reviews(identified_theme);

COMMIT;

-- Verification queries

-- 1) Count reviews per bank
-- SELECT b.bank_name, COUNT(*) AS review_count
-- FROM reviews r
-- JOIN banks b ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name
-- ORDER BY review_count DESC;

-- 2) Average rating per bank
-- SELECT b.bank_name, ROUND(AVG(r.rating)::numeric, 2) AS avg_rating
-- FROM reviews r
-- JOIN banks b ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name
-- ORDER BY avg_rating DESC;

-- 3) Null checks for key columns
-- SELECT
--   SUM(CASE WHEN review_text IS NULL OR review_text = '' THEN 1 ELSE 0 END) AS null_review_text,
--   SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS null_rating,
--   SUM(CASE WHEN review_date IS NULL THEN 1 ELSE 0 END) AS null_review_date,
--   SUM(CASE WHEN sentiment_label IS NULL THEN 1 ELSE 0 END) AS null_sentiment_label,
--   SUM(CASE WHEN identified_theme IS NULL THEN 1 ELSE 0 END) AS null_identified_theme
-- FROM reviews;
