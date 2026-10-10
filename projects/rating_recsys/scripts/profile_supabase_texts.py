"""Read-only profile of the actual Supabase recsys tables.

Credentials are loaded through existing application settings and never written
to results. Raw review text and entity identifiers stay in process memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from profile_review_texts import describe, quantiles, share
from rating_recsys.config import get_settings
from rating_recsys.datasets.repository import FIRST_INTERACTIONS_SQL, InteractionRepository
from rating_recsys.experiments.snapshot import dataset_digest
from rating_recsys.ingestion.transform import file_sha256, normalize_text

SEOUL = ZoneInfo("Asia/Seoul")
SQL_FILE = PROJECT_ROOT / "queries/profile_supabase_text.sql"
LEXICAL_PATTERNS = {
    "taste": r"맛|맵|매운|매워|매콤|짜다|짠|싱겁|고소|식감|신선",
    "price_value": r"가격|가성비|비싸|비싼|저렴|값|원짜리",
    "service": r"서비스|직원|친절|응대|불친절",
    "atmosphere": r"분위기|조용|시끄|인테리어|좌석|공간",
    "quantity": r"양이|양은|양도|푸짐|넉넉|적은 양",
    "waiting": r"웨이팅|대기|기다[렸리]|줄 서|줄서",
    "revisit": r"재방문|다시 방문|또 방문|단골",
    "personal_preference": r"취향|선호|입맛|호불호|좋아하|싫어하",
    "solo_dining": r"혼밥|혼자",
    "group_dining": r"회식|모임|단체",
    "family_companion": r"가족|부모|아이|남편|아내|동행",
    "date_dining": r"데이트",
    "positive_cue": r"맛있|좋다|좋았|좋아|(?<!불)만족|(?<!불)친절|(?<!비)추천|최고|훌륭|괜찮",
    "negative_cue": r"맛없|별로|아쉽|불친절|비싸|비싼|실망|불만|나쁘|최악|비추천",
}
COMPILED_PATTERNS = {key: re.compile(value) for key, value in LEXICAL_PATTERNS.items()}


def json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(type(value).__name__)


def load_queries():
    chunks = re.split(r"^-- name: (\w+)\s*$", SQL_FILE.read_text(), flags=re.MULTILINE)
    return {chunks[index]: chunks[index + 1].strip() for index in range(1, len(chunks), 2)}


def lexical_profile(rows):
    text_rows = [row for row in rows if row.review_text]
    counts = Counter()
    aspect_counts = []
    aspects = ("taste", "price_value", "service", "atmosphere", "quantity", "waiting")
    for row in text_rows:
        present = {name for name, pattern in COMPILED_PATTERNS.items() if pattern.search(row.review_text)}
        counts.update(present)
        counts["positive_and_negative_cues"] += "positive_cue" in present and "negative_cue" in present
        aspect_counts.append(sum(name in present for name in aspects))
    return {
        "text_reviews": len(text_rows),
        "lexical_presence_counts": {name: counts[name] for name in (*LEXICAL_PATTERNS, "positive_and_negative_cues")},
        "multiple_aspect_cues_ge2": sum(count >= 2 for count in aspect_counts),
        "aspect_categories": list(aspects),
        "aspect_cue_count_quantiles": quantiles(aspect_counts),
        "meaning": "regex presence; not semantic annotation, sentiment accuracy, or confirmed author preference",
    }


def cohort_profile(rows):
    result = describe(rows)
    result["users"] = result.pop("users_name_derived")
    result["restaurants"] = result.pop("restaurants_name_address_derived")
    result["lexical"] = lexical_profile(rows)
    raw_present = [row for row in rows if row.source_payload["user_query"] is not None]
    result["raw_payload_text"] = {
        "nonnull_field_rows": len(raw_present),
        "missing_field_rows": len(rows) - len(raw_present),
        "raw_normalized_matches_stored_rows": sum(
            (normalize_text(row.source_payload["user_query"]) or None) == row.review_text for row in raw_present),
        "meaning": "source_payload field, potentially normalized by collector; not original web DOM",
    }
    return result


def groups(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(getattr(row, field))].append(row)
    return {name: cohort_profile(records) for name, records in sorted(grouped.items())}


def activity_distribution(rows, field):
    counts = Counter(getattr(row, field) for row in rows)
    text_counts = Counter(getattr(row, field) for row in rows if row.review_text)
    bins = {}
    for name, low, high in (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, float("inf"))):
        entities = {key for key, count in counts.items() if low <= count <= high}
        bins[name] = {"entities": len(entities), "reviews": sum(counts[key] for key in entities),
                      "text_reviews": sum(text_counts[key] for key in entities),
                      "entities_without_text": sum(text_counts[key] == 0 for key in entities)}
    ordered = sorted(counts.values(), reverse=True)
    top_count = max(1, (len(ordered) + 99) // 100) if ordered else 0
    return {"entities": len(counts), "review_count_quantiles": quantiles(list(counts.values())),
            "bins": bins, "top_1pct_entity_count": top_count,
            "top_1pct_review_share_pct": share(sum(ordered[:top_count]), len(rows))}


def fetch_profile(cutoff):
    import psycopg
    from psycopg.rows import dict_row

    config = get_settings(require_database=True, require_user_hash_salt=False)
    queries = load_queries()
    started = time.perf_counter()
    metadata = {}
    raw_rows = []
    # One consistent live snapshot; no schema/data writes or temp tables.
    with psycopg.connect(config.database_url, connect_timeout=10, prepare_threshold=None,
                         application_name="rating_recsys_text_profile_readonly") as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SET LOCAL statement_timeout = '45s'")
            cursor.execute("SET LOCAL lock_timeout = '3s'")
            for name, sql in queries.items():
                if name == "review_rows":
                    continue
                cursor.execute(sql)
                metadata[name] = cursor.fetchall()
        with connection.cursor(name="text_profile_rows", row_factory=dict_row) as cursor:
            cursor.execute(queries["review_rows"])
            while batch := cursor.fetchmany(20000):
                raw_rows.extend(batch)
        interactions = InteractionRepository(connection).fetch_first_interactions()
        connection.rollback()
    assert metadata["transaction_context"][0]["transaction_read_only"] == "on"
    assert metadata["transaction_context"][0]["transaction_isolation"] == "repeatable read"
    assert len(raw_rows) == metadata["table_counts"][0]["reviews"]

    normalized_changed = null_count = empty_count = null_like_count = 0
    rows = []
    for record in raw_rows:
        stored_text = record["review_text"]
        text = normalize_text(stored_text)
        null_count += stored_text is None
        empty_count += stored_text is not None and not stored_text.strip()
        null_like_count += stored_text is not None and bool(stored_text.strip()) and not text
        normalized_changed += stored_text is not None and stored_text != text
        rows.append(SimpleNamespace(
            review_id=record["review_id"], user_key=record["user_id"], restaurant_key=record["restaurant_id"],
            rating=float(record["rating"]), review_text=text or None,
            taste=record["taste"], price=record["price"], service=record["service"], menu=record["menu"],
            reviewed_at=record["reviewed_at"], reviewed_at_precision=record["reviewed_at_precision"],
            scraped_at=record["scraped_at"].astimezone(SEOUL) if record["scraped_at"] else None,
            region=record["region"], source_file=record["source_file"], event_date=record["event_date"],
            source_payload={"user_query": record["raw_review_text"]},
        ))
    # describe checks date boundaries, so reject an unsupported missing crawl date
    # rather than inventing one. The preceding integrity query records the gap.
    if any(row.scraped_at is None for row in rows):
        raise ValueError("Missing crawl timestamp requires a separate date-boundary policy")
    del raw_rows
    first_ids = {interaction.review_id for interaction in interactions}
    first_rows = [row for row in rows if row.review_id in first_ids]
    train_rows = [row for row in first_rows if row.event_date <= cutoff]
    all_past_rows = [row for row in rows if row.event_date is not None and row.event_date <= cutoff]
    assert len(first_rows) == len(interactions)
    assert len(first_rows) == len({(row.user_key, row.restaurant_key) for row in rows})

    by_quarter = defaultdict(list)
    for row in rows:
        key = (f"{row.event_date.year}-Q{(row.event_date.month - 1) // 3 + 1}"
               if row.event_date else "unknown")
        by_quarter[key].append(row)
    result = {
        "schema_version": "supabase-review-text-profile-v1",
        "observed_at": metadata["transaction_context"][0]["observed_at"].astimezone(SEOUL),
        "source": "live PostgreSQL recsys schema via existing DATABASE_URL; endpoint and credentials omitted",
        "transaction": metadata["transaction_context"][0],
        "table_counts": metadata["table_counts"][0], "table_schema": metadata["schema"],
        "integrity": metadata["integrity"][0], "payload_completeness": metadata["payload_completeness"],
        "implementation_sha256": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in (
            Path(__file__), PROJECT_ROOT / "scripts/profile_review_texts.py", SQL_FILE,
            PROJECT_ROOT / "src/rating_recsys/datasets/repository.py")},
        "executed_queries": {**queries, "canonical_first_interactions": FIRST_INTERACTIONS_SQL},
        "cohort_definitions": {
            "all_reviews": "every recsys.reviews row; one review_id per DB record",
            "first_interactions": "exact InteractionRepository first user/restaurant visit rule; inferred/fallback event_date retained",
            "first_interactions_through_training_cutoff": f"first interactions with event_date <= {cutoff}; no minimum-history or label filter",
            "all_reviews_through_training_cutoff": f"all reviews with event_date <= {cutoff}; before first-visit selection",
        },
        "training_cutoff": cutoff, "canonical_snapshot_id": dataset_digest(interactions),
        "all_reviews": cohort_profile(rows),
        "first_interactions": cohort_profile(first_rows),
        "first_interactions_through_training_cutoff": cohort_profile(train_rows),
        "all_reviews_through_training_cutoff": cohort_profile(all_past_rows),
        "text_missingness": {"sql_null": null_count, "empty_or_whitespace": empty_count,
                             "null_like_nonempty": null_like_count, "changed_by_normalization": normalized_changed},
        "by_rating": groups(rows, "rating"), "by_region": groups(rows, "region"),
        "by_source_file": groups(rows, "source_file"),
        "by_event_quarter": {key: cohort_profile(value) for key, value in sorted(by_quarter.items())},
        "user_activity": activity_distribution(rows, "user_key"),
        "restaurant_activity": activity_distribution(rows, "restaurant_key"),
        "first_interaction_training_user_activity": activity_distribution(train_rows, "user_key"),
        "first_interaction_training_restaurant_activity": activity_distribution(train_rows, "restaurant_key"),
        "lexical_patterns": LEXICAL_PATTERNS,
        "limitations": [
            "DB identity is retained for grouping, but source-level name collisions and restaurant entity resolution are not independently verified.",
            "Word/regex presence is not sentiment, semantic aspect annotation, author preference, topic classification, or causal explanation.",
            "One consistent DB snapshot; source and event-quarter differences mix collection/population effects with potential temporal change.",
            "No tokenizer/model download, new embeddings, recommendation predictions, test metric analysis, or external LLM/API calls.",
            "No raw review text, entity IDs, connection endpoints, or credentials are written to the aggregate result.",
            "Exact repeated body strings are not automatically duplicate visits, spam, or copied reviews.",
            "Crawl payload completeness flags are collector metadata, not independent semantic completeness verification.",
            "Training cutoff uses historical event_date; reviews were inserted later. This is retrospective history, not proof of text availability at the cutoff.",
            "changed_by_normalization compares stored DB text with re-normalization, not raw-to-ingestion information loss.",
            "Positive/negative regex co-occurrence is not a mixed-sentiment label; contextual negation and opinion subject are not resolved.",
        ],
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    assert sum(group["rows"] for group in result["by_rating"].values()) == len(rows)
    assert sum(group["rows"] for group in result["by_region"].values()) == len(rows)
    assert sum(group["rows"] for group in result["by_source_file"].values()) == len(rows)
    assert sum(group["rows"] for group in result["by_event_quarter"].values()) == len(rows)
    assert result["all_reviews"]["rows"] == result["integrity"]["reviews"]
    for cohort in ("all_reviews", "first_interactions", "first_interactions_through_training_cutoff", "all_reviews_through_training_cutoff"):
        profile = result[cohort]
        assert sum(profile["length_bins"].values()) == profile["text_rows"]
        assert sum(profile["review_date_precision"].values()) == profile["rows"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cutoff", type=date.fromisoformat, default=date(2025, 12, 19))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/text_data/supabase_profile.json")
    args = parser.parse_args()
    result = fetch_profile(args.train_cutoff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("observed_at", "table_counts", "canonical_snapshot_id", "all_reviews", "first_interactions_through_training_cutoff", "duration_seconds")}, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Driver errors may include credentials/endpoint strings; keep logs bounded.
        print(json.dumps({"error_type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None)}), file=sys.stderr)
        raise SystemExit(1) from None
