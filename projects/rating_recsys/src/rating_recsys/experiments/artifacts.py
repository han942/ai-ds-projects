"""Stable JSON artifacts shared by MLflow and the local dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rating_recsys.experiments.models import Candidate, RankedCandidate, RecommendationQuery


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def query_record(query: RecommendationQuery) -> dict[str, object]:
    return {
        "query_id": query.query_id,
        "phase": query.phase,
        "user_id": query.user_id,
        "cutoff": query.cutoff.isoformat(),
        "history_depth": query.history_depth,
        "history_review_ids": [item.review_id for item in query.history],
        "history_restaurant_ids": [item.restaurant_id for item in query.history],
        "history_restaurant_names": [item.restaurant_name for item in query.history],
        "target_review_id": query.target.review_id,
        "target_restaurant_id": query.target.restaurant_id,
        "target_restaurant_name": query.target.restaurant_name,
        "target_rating": query.target.rating,
        "relevance": query.relevance,
    }


def candidate_record(candidate: Candidate) -> dict[str, object]:
    return asdict(candidate)


def ranked_record(item: RankedCandidate) -> dict[str, object]:
    return {
        **candidate_record(item.row.candidate),
        "phase": item.row.phase,
        "target_restaurant_id": item.row.target_restaurant_id,
        "relevance": item.row.relevance,
        "history_depth": item.row.history_depth,
        "features": item.row.features,
        "ranking_score": item.ranking_score,
        "final_rank": item.final_rank,
    }


def recommendation_record(item: RankedCandidate) -> dict[str, object]:
    """Return the serving-shaped output without offline target labels."""

    candidate = item.row.candidate
    return {
        "query_id": item.row.query_id,
        "user_id": item.row.user_id,
        "restaurant_id": item.row.restaurant_id,
        "restaurant_name": candidate.restaurant_name,
        "region": candidate.region,
        "candidate_sources": candidate.candidate_sources,
        "source_scores": candidate.source_scores,
        "source_ranks": candidate.source_ranks,
        "candidate_rank": candidate.candidate_rank,
        "ranking_score": item.ranking_score,
        "final_rank": item.final_rank,
    }
