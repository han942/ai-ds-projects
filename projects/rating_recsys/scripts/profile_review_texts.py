"""Profile local ingestion CSVs without querying the DB or exposing review text.

This is a source-file profile, not the canonical first-interaction snapshot.
Run from any directory with Python 3.11+; only the standard library is needed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rating_recsys.ingestion.transform import file_sha256, transform_file

PROFILE_SALT = "local-text-profile-not-database-identities"


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {name: None for name in ("min", "p50", "p90", "p95", "p99", "max")}
    result = {"min": ordered[0], "max": ordered[-1]}
    for name, fraction in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result[name] = round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)
    return result


def share(count, denominator):
    return round(100 * count / denominator, 4) if denominator else None


def describe(rows):
    text_rows = [row for row in rows if row.review_text]
    lengths = [len(row.review_text) for row in text_rows]
    bodies = Counter(row.review_text for row in text_rows)
    user_counts = Counter(row.user_key for row in text_rows)
    item_counts = Counter(row.restaurant_key for row in text_rows)
    users = {row.user_key for row in rows}
    items = {row.restaurant_key for row in rows}
    body_pairs = defaultdict(set)
    for row in text_rows:
        body_pairs[row.review_text].add((row.user_key, row.restaurant_key))
    markers = {
        "negation_or_dissatisfaction_cue": r"맛없|별로|아쉽|불친절|않|못|(?<!\S)안(?:\s|$)",
        "contrast_cue": r"지만|그런데|다만|반면|그래도",
        "explicit_score_or_star_cue": r"별점|[0-5](?:\.5)?\s*점|★|⭐",
        "url_cue": r"https?://|www\.",
        "html_tag_cue": r"</?[A-Za-z][^>]*>",
        "hangul_character": r"[가-힣ㄱ-ㅎㅏ-ㅣ]",
        "latin_character": r"[A-Za-z]",
    }
    return {
        "rows": len(rows), "users_name_derived": len(users), "restaurants_name_address_derived": len(items),
        "text_rows": len(text_rows), "missing_text_rows": len(rows) - len(text_rows),
        "text_coverage_pct": share(len(text_rows), len(rows)),
        "normalized_character_length": quantiles(lengths),
        "short_text_le20": sum(length <= 20 for length in lengths),
        "long_text_gt240": sum(length > 240 for length in lengths),
        "length_bins": {label: sum(low <= length <= high for length in lengths) for label, low, high in
                        (("1-20", 1, 20), ("21-100", 21, 100), ("101-240", 101, 240),
                         ("241-500", 241, 500), ("501+", 501, float("inf")))},
        "distinct_normalized_bodies": len(bodies),
        "repeated_body_excess_rows": len(text_rows) - len(bodies),
        "bodies_shared_by_distinct_user_restaurant_pairs": sum(len(pairs) > 1 for pairs in body_pairs.values()),
        "marker_review_counts": {name: sum(bool(re.search(pattern, row.review_text)) for row in text_rows)
                                 for name, pattern in markers.items()},
        "raw_multiline_text_rows": sum("\n" in str(row.source_payload.get("user_query", "")) or
                                       "\r" in str(row.source_payload.get("user_query", "")) for row in text_rows),
        "normalized_text_reviews_per_user": quantiles([user_counts[user] for user in users]),
        "normalized_text_reviews_per_restaurant": quantiles([item_counts[item] for item in items]),
        "users_without_text": sum(user_counts[user] == 0 for user in users),
        "restaurants_without_text": sum(item_counts[item] == 0 for item in items),
        "review_date_precision": dict(sorted(Counter(row.reviewed_at_precision for row in rows).items())),
        "parsed_review_date_min": min((row.reviewed_at.isoformat() for row in rows if row.reviewed_at), default=None),
        "parsed_review_date_max": max((row.reviewed_at.isoformat() for row in rows if row.reviewed_at), default=None),
        "parsed_date_after_crawl": sum(row.reviewed_at is not None and row.reviewed_at > row.scraped_at.date() for row in rows),
        "structured_attribute_coverage": {field: sum(getattr(row, field) is not None for row in rows)
                                          for field in ("taste", "price", "service", "menu")},
    }


def profile(input_dir):
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files in {input_dir}")
    batches = [transform_file(path, PROFILE_SALT) for path in files]
    accepted = [row for batch in batches for row in batch.rows]
    unique_by_hash = {}
    for row in accepted:
        unique_by_hash.setdefault(row.content_hash, row)
    unique = list(unique_by_hash.values())
    ratings, regions = defaultdict(list), defaultdict(list)
    for row in unique:
        ratings[f"{row.rating:g}"].append(row)
        regions[row.region].append(row)
    inventory = [{"file": batch.path.name, "sha256": batch.file_sha256,
                  "scraped_at": batch.scraped_at.isoformat(), "raw_rows": batch.total_rows,
                  "accepted_rows": len(batch.rows), "rejected_rows": batch.rejected_rows,
                  "rejection_reasons": batch.rejection_reasons} for batch in batches]
    assert sum(item["raw_rows"] for item in inventory) == len(accepted) + sum(item["rejected_rows"] for item in inventory)
    assert len(unique) == len(unique_by_hash) <= len(accepted)
    result = {
        "schema_version": "local-review-text-profile-v1",
        "executed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "five local ingestion source CSVs; not live DB or canonical experiment snapshot",
        "grain": "accepted source rows and distinct ingestion content_hash; no first-visit or cutoff filter",
        "normalization": "existing ingestion normalize_text; whitespace collapsed; null-like strings removed",
        "implementation_sha256": {
            "scripts/profile_review_texts.py": file_sha256(Path(__file__)),
            "src/rating_recsys/ingestion/transform.py": file_sha256(PROJECT_ROOT / "src/rating_recsys/ingestion/transform.py"),
        },
        "inventory": inventory,
        "raw_rows": sum(item["raw_rows"] for item in inventory),
        "rejected_rows": sum(item["rejected_rows"] for item in inventory),
        "accepted_source_rows": len(accepted),
        "duplicate_content_hash_excess_rows": len(accepted) - len(unique),
        "distinct_reviews": describe(unique),
        "distinct_user_restaurant_pairs": len({(row.user_key, row.restaurant_key) for row in unique}),
        "by_rating": {key: describe(value) for key, value in sorted(ratings.items(), key=lambda pair: float(pair[0]))},
        "by_region": {key: describe(value) for key, value in sorted(regions.items())},
        "limitations": [
            "No live Supabase read, canonical snapshot, tokenizer, embedding, or model evaluation.",
            "User identity uses normalized display name; identical names can merge people and rename can split people.",
            "Restaurant identity uses normalized name and address, matching ingestion; identity resolution is not verified.",
            "Content-hash deduplication is ingestion equivalence, not independently verified unique real-world visits.",
            "Unknown dates are counted, not replaced with crawl dates; inferred years and relative dates remain labeled.",
            "Regex cues are lexical presence only, not sentiment, aspect, language, spam, or leakage labels.",
            "Character lengths are not E5 token counts or measurements of actual truncation.",
            "20 and 240 character cutoffs describe length and legacy input cap; they are not quality rules.",
        ],
    }
    assert sum(group["rows"] for group in result["by_rating"].values()) == len(unique)
    assert sum(group["rows"] for group in result["by_region"].values()) == len(unique)
    assert sum(result["distinct_reviews"]["length_bins"].values()) == result["distinct_reviews"]["text_rows"]
    assert sum(result["distinct_reviews"]["review_date_precision"].values()) == len(unique)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "legacy/v1_rating_prediction/crawled_data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/text_data/local_csv_profile.json")
    args = parser.parse_args()
    result = profile(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("raw_rows", "accepted_source_rows", "rejected_rows", "duplicate_content_hash_excess_rows", "distinct_reviews")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
