"""Validate or ingest all legacy DiningCode CSV files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rating_recsys.config import get_settings
from rating_recsys.db.connection import connect
from rating_recsys.db.migrate import apply_migrations
from rating_recsys.ingestion.loader import load_batch
from rating_recsys.ingestion.transform import (
    discover_csv_files,
    transform_file,
    unique_content_hashes,
)


DRY_RUN_SALT = "dry-run-only-not-used-for-database-identities"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing the five legacy CSV files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, normalize and profile files without connecting to PostgreSQL",
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Do not apply pending SQL migrations before ingestion",
    )
    return parser


def dry_run(files: list[Path], salt: str) -> dict:
    batches = [transform_file(path, salt) for path in files]
    all_rows = [row for batch in batches for row in batch.rows]
    unique_hashes = unique_content_hashes(all_rows)
    total_accepted = len(all_rows)

    return {
        "mode": "dry-run",
        "files": [
            {
                "source_file": batch.path.name,
                "region": batch.region,
                "scraped_at": batch.scraped_at.isoformat(),
                "total_rows": batch.total_rows,
                "accepted_rows": len(batch.rows),
                "rejected_rows": batch.rejected_rows,
                "rejection_reasons": batch.rejection_reasons,
            }
            for batch in batches
        ],
        "summary": {
            "file_count": len(batches),
            "total_rows": sum(batch.total_rows for batch in batches),
            "accepted_rows": total_accepted,
            "rejected_rows": sum(batch.rejected_rows for batch in batches),
            "unique_reviews": len(unique_hashes),
            "duplicate_rows": total_accepted - len(unique_hashes),
            "restaurants": len({row.restaurant_key for row in all_rows}),
            "users": len({row.user_key for row in all_rows}),
            "unknown_review_dates": sum(
                row.reviewed_at is None for row in all_rows
            ),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings(require_database=not args.dry_run)
    input_dir = args.input_dir or settings.input_dir
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    files = discover_csv_files(input_dir)
    if not files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    salt = settings.user_hash_salt or DRY_RUN_SALT
    if args.dry_run:
        print(json.dumps(dry_run(files, salt), ensure_ascii=False, indent=2))
        return

    if not args.skip_migrations:
        applied = apply_migrations(settings.database_url)
        if applied:
            print(json.dumps({"applied_migrations": applied}, ensure_ascii=False))

    results = []
    with connect(settings.database_url) as conn:
        for path in files:
            batch = transform_file(path, salt)
            results.append(asdict(load_batch(conn, batch)))

    print(
        json.dumps(
            {
                "mode": "database",
                "files": results,
                "summary": {
                    "file_count": len(results),
                    "total_rows": sum(item["total_rows"] for item in results),
                    "inserted_reviews": sum(
                        item["inserted_reviews"] for item in results
                    ),
                    "duplicate_reviews": sum(
                        item["duplicate_reviews"] for item in results
                    ),
                    "rejected_rows": sum(
                        item["rejected_rows"] for item in results
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
