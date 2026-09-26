"""Transactional, idempotent PostgreSQL loader for transformed CSV batches."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from rating_recsys.ingestion.transform import FileBatch


@dataclass(frozen=True)
class LoadResult:
    source_file: str
    status: str
    total_rows: int
    accepted_rows: int
    inserted_reviews: int
    duplicate_reviews: int
    rejected_rows: int


CREATE_BUFFER_SQL = """
CREATE TEMP TABLE ingest_buffer (
    restaurant_key char(64) NOT NULL,
    canonical_name text NOT NULL,
    area text,
    address text NOT NULL,
    region text NOT NULL,
    item_avg_rating numeric(3, 2),
    restaurant_metadata jsonb NOT NULL,
    user_key char(64) NOT NULL,
    user_metadata jsonb NOT NULL,
    rating numeric(2, 1) NOT NULL,
    review_text text,
    taste smallint,
    price smallint,
    service smallint,
    menu text,
    reviewed_at date,
    reviewed_at_precision text NOT NULL,
    raw_date text,
    scraped_at timestamptz NOT NULL,
    content_hash char(64) NOT NULL,
    source_row_number integer NOT NULL,
    source_file text NOT NULL,
    source_payload jsonb NOT NULL
) ON COMMIT DROP
"""


COPY_BUFFER_SQL = """
COPY ingest_buffer (
    restaurant_key, canonical_name, area, address, region,
    item_avg_rating, restaurant_metadata, user_key, user_metadata,
    rating, review_text, taste, price, service, menu,
    reviewed_at, reviewed_at_precision, raw_date, scraped_at,
    content_hash, source_row_number, source_file, source_payload
) FROM STDIN
"""


UPSERT_RESTAURANTS_SQL = """
INSERT INTO recsys.restaurants (
    restaurant_key, canonical_name, area, address, region,
    item_avg_rating, first_seen_at, last_seen_at, source_metadata
)
SELECT DISTINCT ON (restaurant_key)
    restaurant_key, canonical_name, area, address, region,
    item_avg_rating, scraped_at, scraped_at, restaurant_metadata
FROM ingest_buffer
ORDER BY restaurant_key, source_row_number DESC
ON CONFLICT (restaurant_key) DO UPDATE SET
    canonical_name = EXCLUDED.canonical_name,
    area = COALESCE(EXCLUDED.area, recsys.restaurants.area),
    address = EXCLUDED.address,
    region = EXCLUDED.region,
    item_avg_rating = COALESCE(
        EXCLUDED.item_avg_rating,
        recsys.restaurants.item_avg_rating
    ),
    first_seen_at = LEAST(
        recsys.restaurants.first_seen_at,
        EXCLUDED.first_seen_at
    ),
    last_seen_at = GREATEST(
        recsys.restaurants.last_seen_at,
        EXCLUDED.last_seen_at
    ),
    source_metadata = recsys.restaurants.source_metadata || EXCLUDED.source_metadata,
    updated_at = now()
"""


UPSERT_USERS_SQL = """
INSERT INTO recsys.app_users (
    user_key, source, first_seen_at, last_seen_at, source_metadata
)
SELECT DISTINCT ON (user_key)
    user_key, 'diningcode', scraped_at, scraped_at, user_metadata
FROM ingest_buffer
ORDER BY user_key, source_row_number DESC
ON CONFLICT (user_key) DO UPDATE SET
    first_seen_at = LEAST(
        recsys.app_users.first_seen_at,
        EXCLUDED.first_seen_at
    ),
    last_seen_at = GREATEST(
        recsys.app_users.last_seen_at,
        EXCLUDED.last_seen_at
    ),
    source_metadata = recsys.app_users.source_metadata || EXCLUDED.source_metadata,
    updated_at = now()
"""


INSERT_RESTAURANTS_ONLY_SQL = """
INSERT INTO recsys.restaurants (
    restaurant_key, canonical_name, area, address, region,
    item_avg_rating, first_seen_at, last_seen_at, source_metadata
)
SELECT DISTINCT ON (restaurant_key)
    restaurant_key, canonical_name, area, address, region,
    item_avg_rating, scraped_at, scraped_at, restaurant_metadata
FROM ingest_buffer
ORDER BY restaurant_key, source_row_number DESC
ON CONFLICT (restaurant_key) DO NOTHING
"""


INSERT_USERS_ONLY_SQL = """
INSERT INTO recsys.app_users (
    user_key, source, first_seen_at, last_seen_at, source_metadata
)
SELECT DISTINCT ON (user_key)
    user_key, 'diningcode', scraped_at, scraped_at, user_metadata
FROM ingest_buffer
ORDER BY user_key, source_row_number DESC
ON CONFLICT (user_key) DO NOTHING
"""


INSERT_REVIEWS_SQL = """
WITH inserted AS (
    INSERT INTO recsys.reviews (
        content_hash, restaurant_id, user_id, crawl_run_id,
        rating, review_text, taste, price, service, menu,
        reviewed_at, reviewed_at_precision, scraped_at, raw_date,
        source_row_number, source_file, source_payload
    )
    SELECT
        b.content_hash,
        r.restaurant_id,
        u.user_id,
        %s,
        b.rating,
        b.review_text,
        b.taste,
        b.price,
        b.service,
        b.menu,
        b.reviewed_at,
        b.reviewed_at_precision,
        b.scraped_at,
        b.raw_date,
        b.source_row_number,
        b.source_file,
        b.source_payload
    FROM ingest_buffer AS b
    JOIN recsys.restaurants AS r USING (restaurant_key)
    JOIN recsys.app_users AS u USING (user_key)
    ON CONFLICT (content_hash) DO NOTHING
    RETURNING review_id
)
SELECT count(*) FROM inserted
"""


def _existing_run(cursor, file_hash: str):
    cursor.execute(
        """
        SELECT run_id, status, total_rows, inserted_reviews,
               duplicate_reviews, rejected_rows
        FROM recsys.crawl_runs
        WHERE file_sha256 = %s
        """,
        (file_hash,),
    )
    return cursor.fetchone()


def load_batch(
    conn, batch: FileBatch, *, source: str = "diningcode",
    preserve_existing_dimensions: bool = False,
) -> LoadResult:
    """Load one file atomically and skip a previously successful file hash."""

    try:
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        raise RuntimeError("psycopg is required for database ingestion") from exc

    with conn.cursor() as cur:
        existing = _existing_run(cur, batch.file_sha256)
        if existing and existing[1] == "succeeded":
            conn.commit()
            return LoadResult(
                source_file=batch.path.name,
                status="skipped",
                total_rows=existing[2],
                accepted_rows=existing[2] - existing[5],
                inserted_reviews=existing[3],
                duplicate_reviews=existing[4],
                rejected_rows=existing[5],
            )
        if existing and existing[1] == "running":
            conn.rollback()
            raise RuntimeError(
                f"An ingestion run is already active for {batch.path.name}"
            )

        run_id = existing[0] if existing else uuid.uuid4()
        if existing:
            cur.execute(
                """
                UPDATE recsys.crawl_runs
                SET status = 'running', started_at = now(), completed_at = NULL,
                    total_rows = 0, inserted_reviews = 0,
                    duplicate_reviews = 0, rejected_rows = 0,
                    error_message = NULL
                WHERE run_id = %s
                """,
                (run_id,),
            )
        else:
            cur.execute(
                """
                INSERT INTO recsys.crawl_runs (
                    run_id, source, source_file, file_sha256,
                    region, scraped_at, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'running')
                """,
                (
                    run_id,
                    source,
                    batch.path.name,
                    batch.file_sha256,
                    batch.region,
                    batch.scraped_at,
                ),
            )
        conn.commit()

        try:
            with conn.transaction():
                cur.execute(CREATE_BUFFER_SQL)
                with cur.copy(COPY_BUFFER_SQL) as copy:
                    for row in batch.rows:
                        copy.write_row(
                            (
                                row.restaurant_key,
                                row.canonical_name,
                                row.area,
                                row.address,
                                row.region,
                                row.item_avg_rating,
                                Jsonb(row.restaurant_metadata),
                                row.user_key,
                                Jsonb(row.user_metadata),
                                row.rating,
                                row.review_text,
                                row.taste,
                                row.price,
                                row.service,
                                row.menu,
                                row.reviewed_at,
                                row.reviewed_at_precision,
                                row.raw_date,
                                row.scraped_at,
                                row.content_hash,
                                row.source_row_number,
                                row.source_file,
                                Jsonb(row.source_payload),
                            )
                        )

                cur.execute(
                    INSERT_RESTAURANTS_ONLY_SQL if preserve_existing_dimensions
                    else UPSERT_RESTAURANTS_SQL
                )
                cur.execute(
                    INSERT_USERS_ONLY_SQL if preserve_existing_dimensions
                    else UPSERT_USERS_SQL
                )
                cur.execute(INSERT_REVIEWS_SQL, (run_id,))
                inserted_reviews = cur.fetchone()[0]
                accepted_rows = len(batch.rows)
                duplicate_reviews = accepted_rows - inserted_reviews

                cur.execute(
                    """
                    UPDATE recsys.crawl_runs
                    SET status = 'succeeded', completed_at = now(),
                        total_rows = %s, inserted_reviews = %s,
                        duplicate_reviews = %s, rejected_rows = %s
                    WHERE run_id = %s
                    """,
                    (
                        batch.total_rows,
                        inserted_reviews,
                        duplicate_reviews,
                        batch.rejected_rows,
                        run_id,
                    ),
                )
        except Exception as exc:
            conn.rollback()
            with conn.transaction():
                cur.execute(
                    """
                    UPDATE recsys.crawl_runs
                    SET status = 'failed', completed_at = now(), error_message = %s,
                        total_rows = %s, rejected_rows = %s
                    WHERE run_id = %s
                    """,
                    (str(exc)[:4000], batch.total_rows, batch.rejected_rows, run_id),
                )
            raise

    return LoadResult(
        source_file=batch.path.name,
        status="succeeded",
        total_rows=batch.total_rows,
        accepted_rows=len(batch.rows),
        inserted_reviews=inserted_reviews,
        duplicate_reviews=duplicate_reviews,
        rejected_rows=batch.rejected_rows,
    )
