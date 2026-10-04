"""Import a frozen snapshot of the current national Playwright crawl."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.db.connection import connect
from rating_recsys.ingestion.loader import load_batch
from rating_recsys.ingestion.transform import transform_file


CRAWLER_DATA = PROJECT_ROOT / "crawler" / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "snapshots" / "ingestion_sources"
SOURCE = "diningcode_playwright_national"


def latest_partial() -> Path:
    files = sorted(CRAWLER_DATA.glob("diningcode_playwright_national_*.csv.partial"))
    if not files:
        raise FileNotFoundError(f"No national crawler CSV partial in {CRAWLER_DATA}")
    return max(files, key=lambda path: path.stat().st_mtime_ns)


def freeze_partial(path: Path, destination_root: Path = ARTIFACTS) -> Path:
    """Copy the current complete CSV rows; reject a concurrent source change."""
    destination_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = path.name.removesuffix(".csv.partial")
    frozen = destination_root / f"{stem}_snapshot_{started}.csv"
    before = path.stat()
    shutil.copyfile(path, frozen)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        frozen.unlink()
        raise RuntimeError("Crawler CSV changed during snapshot copy; retry after it settles")
    if frozen.stat().st_size != before.st_size:
        frozen.unlink()
        raise RuntimeError("Crawler snapshot size does not match the source")
    return frozen


def _counts(cursor) -> dict[str, int]:
    cursor.execute("SELECT count(*) FROM recsys.reviews")
    total = cursor.fetchone()[0]
    cursor.execute("""
        SELECT count(*) FROM recsys.reviews AS r
        JOIN recsys.crawl_runs AS cr ON cr.run_id = r.crawl_run_id
        WHERE cr.source = 'diningcode'
    """)
    legacy = cursor.fetchone()[0]
    return {"reviews_total": total, "legacy_reviews": legacy}


def main() -> None:
    settings = get_settings(require_database=True)
    source = latest_partial()
    frozen = freeze_partial(source)
    batch = transform_file(frozen, settings.user_hash_salt)
    print(json.dumps({
        "snapshot": str(frozen), "sha256": batch.file_sha256,
        "rows": batch.total_rows, "accepted": len(batch.rows),
        "rejected": batch.rejected_rows, "rejection_reasons": batch.rejection_reasons,
    }, ensure_ascii=False), flush=True)
    if not batch.rows:
        raise RuntimeError("No valid crawler rows in frozen snapshot")
    with connect(settings.database_url) as conn:
        with conn.cursor() as cursor:
            before = _counts(cursor)
        result = load_batch(
            conn, batch, source=SOURCE, preserve_existing_dimensions=True,
        )
        with conn.cursor() as cursor:
            after = _counts(cursor)
            cursor.execute("""
                SELECT cr.run_id::text, cr.source, cr.source_file,
                       cr.status, cr.total_rows, cr.inserted_reviews,
                       cr.duplicate_reviews, cr.rejected_rows,
                       count(r.review_id) AS linked_reviews
                FROM recsys.crawl_runs AS cr
                LEFT JOIN recsys.reviews AS r ON r.crawl_run_id = cr.run_id
                WHERE cr.file_sha256 = %s
                GROUP BY cr.run_id
            """, (batch.file_sha256,))
            columns = [column.name for column in cursor.description]
            run = dict(zip(columns, cursor.fetchone()))
    if after["legacy_reviews"] != before["legacy_reviews"]:
        raise RuntimeError("Legacy review count changed unexpectedly")
    if result.status == "succeeded" and (
        after["reviews_total"] - before["reviews_total"] != result.inserted_reviews
        or run["linked_reviews"] != result.inserted_reviews
    ):
        raise RuntimeError("Post-import review counts do not reconcile")
    report = {
        "source_snapshot": str(frozen), "file_sha256": batch.file_sha256,
        "source_type": SOURCE, "load_result": asdict(result),
        "before": before, "after": after, "crawl_run": run,
        "partial_crawl": True,
    }
    report_path = frozen.with_suffix(".import.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
