-- name: transaction_context
SELECT transaction_timestamp() AS observed_at,
       current_setting('transaction_read_only') AS transaction_read_only,
       current_setting('transaction_isolation') AS transaction_isolation;

-- name: schema
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'recsys'
  AND table_name IN ('reviews', 'restaurants', 'app_users', 'crawl_runs')
ORDER BY table_name, ordinal_position;

-- name: table_counts
SELECT (SELECT count(*) FROM recsys.reviews) AS reviews,
       (SELECT count(*) FROM recsys.app_users) AS users,
       (SELECT count(*) FROM recsys.restaurants) AS restaurants,
       (SELECT count(*) FROM recsys.crawl_runs) AS crawl_runs;

-- name: integrity
SELECT count(*) AS reviews,
       count(DISTINCT rv.review_id) AS distinct_review_ids,
       count(DISTINCT rv.content_hash) AS distinct_content_hashes,
       count(*) FILTER (WHERE u.user_id IS NULL) AS orphan_users,
       count(*) FILTER (WHERE rs.restaurant_id IS NULL) AS orphan_restaurants,
       count(*) FILTER (WHERE cr.run_id IS NULL) AS orphan_crawl_runs,
       count(*) FILTER (WHERE rv.rating IS NULL OR rv.rating < 0 OR rv.rating > 5) AS invalid_ratings,
       count(*) FILTER (WHERE rv.reviewed_at_precision IS NULL
          OR rv.reviewed_at_precision NOT IN ('exact', 'inferred_year', 'relative', 'unknown')) AS invalid_date_precision,
       count(*) FILTER (WHERE rv.reviewed_at IS NULL) AS missing_review_date,
       count(*) FILTER (WHERE rv.scraped_at IS NULL) AS missing_crawl_timestamp,
       min(rv.created_at) AS first_db_insert,
       max(rv.created_at) AS last_db_insert,
       min(rv.scraped_at) AS first_crawl_timestamp,
       max(rv.scraped_at) AS last_crawl_timestamp
FROM recsys.reviews rv
LEFT JOIN recsys.app_users u USING (user_id)
LEFT JOIN recsys.restaurants rs USING (restaurant_id)
LEFT JOIN recsys.crawl_runs cr ON cr.run_id = rv.crawl_run_id;

-- name: payload_completeness
SELECT source_file,
       source_payload ->> 'review_text_complete' AS reported_complete,
       count(*) AS reviews
FROM recsys.reviews
GROUP BY source_file, source_payload ->> 'review_text_complete'
ORDER BY source_file, reported_complete;

-- name: review_rows
SELECT rv.review_id, rv.user_id, rv.restaurant_id, rv.rating, rv.review_text,
       rv.taste, rv.price, rv.service, rv.menu,
       rv.reviewed_at, rv.reviewed_at_precision, rv.scraped_at,
       rv.source_file, rs.region,
       rv.source_payload ->> 'user_query' AS raw_review_text,
       COALESCE(rv.reviewed_at, (rv.scraped_at AT TIME ZONE 'Asia/Seoul')::date) AS event_date
FROM recsys.reviews rv
JOIN recsys.restaurants rs USING (restaurant_id)
ORDER BY rv.review_id;
