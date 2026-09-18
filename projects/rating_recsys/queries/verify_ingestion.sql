SELECT
    source_file,
    region,
    status,
    total_rows,
    inserted_reviews,
    duplicate_reviews,
    rejected_rows,
    completed_at
FROM recsys.crawl_runs
ORDER BY scraped_at, source_file;

SELECT
    (SELECT count(*) FROM recsys.restaurants) AS restaurants,
    (SELECT count(*) FROM recsys.app_users) AS users,
    (SELECT count(*) FROM recsys.reviews) AS reviews;

SELECT
    region,
    count(*) AS restaurants
FROM recsys.restaurants
GROUP BY region
ORDER BY restaurants DESC;

SELECT
    reviewed_at_precision,
    count(*) AS reviews
FROM recsys.reviews
GROUP BY reviewed_at_precision
ORDER BY reviews DESC;

SELECT content_hash, count(*) AS occurrences
FROM recsys.reviews
GROUP BY content_hash
HAVING count(*) > 1;
