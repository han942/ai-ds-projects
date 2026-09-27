"""Audit the global temporal split of the current DB without training."""

from __future__ import annotations

import argparse
import json

from rating_recsys.config import get_settings
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.db.connection import connect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
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
    settings = get_settings(require_database=True, require_user_hash_salt=False)
    with connect(settings.database_url) as connection:
        interactions = InteractionRepository(connection).fetch_first_interactions()
    if not interactions:
        raise SystemExit("No interactions found in recsys.reviews")
    split = build_global_temporal_split(
        interactions,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(
        json.dumps(
            {
                "mode": "database-read",
                "snapshot": _snapshot_summary(interactions),
                "split": split.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
