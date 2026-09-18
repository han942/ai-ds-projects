CREATE TABLE IF NOT EXISTS recsys.crawl_runs (
    run_id uuid PRIMARY KEY,
    source text NOT NULL,
    source_file text NOT NULL,
    file_sha256 char(64) NOT NULL UNIQUE,
    region text NOT NULL,
    scraped_at timestamptz,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    total_rows integer NOT NULL DEFAULT 0 CHECK (total_rows >= 0),
    inserted_reviews integer NOT NULL DEFAULT 0 CHECK (inserted_reviews >= 0),
    duplicate_reviews integer NOT NULL DEFAULT 0 CHECK (duplicate_reviews >= 0),
    rejected_rows integer NOT NULL DEFAULT 0 CHECK (rejected_rows >= 0),
    error_message text
);

CREATE TABLE IF NOT EXISTS recsys.restaurants (
    restaurant_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    restaurant_key char(64) NOT NULL UNIQUE,
    canonical_name text NOT NULL,
    area text,
    address text NOT NULL,
    region text NOT NULL,
    item_avg_rating numeric(3, 2)
        CHECK (item_avg_rating IS NULL OR item_avg_rating BETWEEN 0 AND 5),
    first_seen_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recsys.app_users (
    user_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_key char(64) NOT NULL UNIQUE,
    source text NOT NULL,
    first_seen_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recsys.reviews (
    review_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_hash char(64) NOT NULL UNIQUE,
    restaurant_id bigint NOT NULL
        REFERENCES recsys.restaurants (restaurant_id),
    user_id bigint NOT NULL
        REFERENCES recsys.app_users (user_id),
    crawl_run_id uuid NOT NULL
        REFERENCES recsys.crawl_runs (run_id),
    rating numeric(2, 1) NOT NULL CHECK (rating BETWEEN 0 AND 5),
    review_text text,
    taste smallint CHECK (taste IS NULL OR taste BETWEEN 0 AND 2),
    price smallint CHECK (price IS NULL OR price BETWEEN 0 AND 2),
    service smallint CHECK (service IS NULL OR service BETWEEN 0 AND 2),
    menu text,
    reviewed_at date,
    reviewed_at_precision text NOT NULL
        CHECK (reviewed_at_precision IN (
            'exact', 'inferred_year', 'relative', 'unknown'
        )),
    scraped_at timestamptz,
    raw_date text,
    source_row_number integer NOT NULL CHECK (source_row_number > 0),
    source_file text NOT NULL,
    source_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_restaurants_region
    ON recsys.restaurants (region);
CREATE INDEX IF NOT EXISTS idx_restaurants_name
    ON recsys.restaurants (canonical_name);
CREATE INDEX IF NOT EXISTS idx_reviews_user_reviewed_at
    ON recsys.reviews (user_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_reviews_restaurant_reviewed_at
    ON recsys.reviews (restaurant_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_reviews_crawl_run
    ON recsys.reviews (crawl_run_id);

COMMENT ON SCHEMA recsys IS
    'Private training and recommendation data; not exposed through the Data API by default.';
COMMENT ON COLUMN recsys.app_users.user_key IS
    'Stable salted SHA-256 pseudonym. Raw source user names are not stored.';
COMMENT ON COLUMN recsys.reviews.content_hash IS
    'Stable deduplication key across repeated crawler exports.';
