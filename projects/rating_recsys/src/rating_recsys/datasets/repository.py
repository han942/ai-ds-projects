"""Read recommendation interactions from PostgreSQL, never from legacy CSVs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rating_recsys.datasets.models import Interaction

if TYPE_CHECKING:
    from psycopg import Connection


FIRST_INTERACTIONS_SQL = """
WITH ranked_interactions AS (
    SELECT
        rv.review_id,
        rv.user_id,
        rv.restaurant_id,
        COALESCE(
            rv.reviewed_at,
            (rv.scraped_at AT TIME ZONE 'Asia/Seoul')::date
        ) AS event_date,
        rv.rating,
        rv.reviewed_at_precision,
        rs.canonical_name,
        rs.region,
        row_number() OVER (
            PARTITION BY rv.user_id, rv.restaurant_id
            ORDER BY
                COALESCE(
                    rv.reviewed_at,
                    (rv.scraped_at AT TIME ZONE 'Asia/Seoul')::date
                ),
                rv.scraped_at,
                rv.review_id
        ) AS interaction_order
    FROM recsys.reviews AS rv
    JOIN recsys.restaurants AS rs
      ON rs.restaurant_id = rv.restaurant_id
)
SELECT
    review_id,
    user_id,
    restaurant_id,
    event_date,
    rating,
    reviewed_at_precision,
    canonical_name,
    region
FROM ranked_interactions
WHERE interaction_order = 1
  AND event_date IS NOT NULL
ORDER BY user_id, event_date, review_id
"""

REVIEW_TEXTS_SQL = """
SELECT review_id, review_text
FROM recsys.reviews
WHERE review_id = ANY(%s)
ORDER BY review_id
"""


class InteractionRepository:
    """Repository for the canonical modeling interaction grain."""

    def __init__(self, connection: "Connection") -> None:
        self._connection = connection

    def fetch_first_interactions(self) -> list[Interaction]:
        """Return one first visit per user/restaurant pair.

        The database performs pair-level deduplication before records enter Python.
        No target review text or target aspect score is selected.
        """

        with self._connection.cursor() as cursor:
            cursor.execute(FIRST_INTERACTIONS_SQL, prepare=False)
            return [
                Interaction(
                    review_id=row[0],
                    user_id=row[1],
                    restaurant_id=row[2],
                    event_date=row[3],
                    rating=float(row[4]),
                    reviewed_at_precision=row[5],
                    restaurant_name=row[6],
                    region=row[7],
                )
                for row in cursor.fetchall()
            ]

    def fetch_review_texts(self, review_ids: list[int]) -> dict[int, str | None]:
        """Fetch review text by id, separately from the text-free interactions.

        Models that read text (DeepCoNN) must only use reviews dated up to
        their own cutoff; the caller is responsible for that filter.
        """

        ordered_ids = sorted(set(review_ids))
        if not ordered_ids:
            return {}
        with self._connection.cursor() as cursor:
            cursor.execute(REVIEW_TEXTS_SQL, (ordered_ids,), prepare=False)
            return {int(row[0]): row[1] for row in cursor.fetchall()}
