"""Read the modeling dataset from PostgreSQL and audit leakage-aware splits."""

from __future__ import annotations

import argparse
import json

from rating_recsys.config import get_settings
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.datasets.split import (
    build_global_temporal_split,
    build_seen_user_split,
)
from rating_recsys.db.connection import connect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimum-user-items",
        type=int,
        default=3,
        help="Minimum unique restaurants for the primary seen-user split",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Global temporal train fraction before date-boundary adjustment",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Global temporal validation fraction before date-boundary adjustment",
    )
    return parser


def _snapshot_summary(interactions) -> dict[str, object]:
    event_dates = [item.event_date for item in interactions]
    return {
        "source": "postgresql:recsys.reviews",
        "grain": "first-user-restaurant-interaction",
        "interactions": len(interactions),
        "users": len({item.user_id for item in interactions}),
        "restaurants": len({item.restaurant_id for item in interactions}),
        "date_range": {
            "minimum": min(event_dates).isoformat(),
            "maximum": max(event_dates).isoformat(),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings(require_database=True)

    with connect(settings.database_url) as connection:
        interactions = InteractionRepository(connection).fetch_first_interactions()

    if not interactions:
        raise SystemExit("No interactions found in recsys.reviews")

    seen_user = build_seen_user_split(
        interactions,
        minimum_user_items=args.minimum_user_items,
    )
    temporal = build_global_temporal_split(
        interactions,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(
        json.dumps(
            {
                "mode": "database-read",
                "snapshot": _snapshot_summary(interactions),
                "primary_seen_user": seen_user.summary(),
                "secondary_temporal": temporal.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
