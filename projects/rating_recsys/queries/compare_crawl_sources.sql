-- Supabase SQL Editor: compare the legacy and national Playwright cohorts.
SELECT
    cr.source,
    cr.run_id,
    cr.source_file,
    cr.status,
    cr.total_rows,
    cr.inserted_reviews,
    cr.duplicate_reviews,
    cr.rejected_rows,
    count(r.review_id) AS linked_reviews
FROM recsys.crawl_runs AS cr
LEFT JOIN recsys.reviews AS r ON r.crawl_run_id = cr.run_id
GROUP BY cr.run_id
ORDER BY cr.started_at;

-- New national crawl only. Replace source with 'diningcode' for the old data.
SELECT r.*
FROM recsys.reviews AS r
JOIN recsys.crawl_runs AS cr ON cr.run_id = r.crawl_run_id
WHERE cr.source = 'diningcode_playwright_national';

-- Exact frozen import from 2026-09-26; later resume imports get their own run_id.
SELECT count(*) AS reviews_in_this_snapshot
FROM recsys.reviews
WHERE crawl_run_id = 'b22dcc90-6dfc-46eb-a967-05e7406373a9';
