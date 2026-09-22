"""Stable JSON artifacts shared by MLflow and the local dashboard."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import islice
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


def write_parquet(
    path: Path,
    records: Iterable[dict[str, object]],
    *,
    batch_size: int = 50_000,
) -> dict[str, int | str]:
    """Write large detail artifacts in deterministic Zstandard Parquet batches."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet artifacts require pyarrow from the experiment extra"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(records)
    writer = None
    schema = None
    row_count = 0
    try:
        while batch := list(islice(iterator, batch_size)):
            prepared = [_parquet_record(record) for record in batch]
            table = pa.Table.from_pylist(
                prepared,
                schema=schema,
            )
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    path,
                    table.schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=batch_size)
            row_count += len(batch)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(pa.table({}), path, compression="zstd")
    return {
        "format": "parquet",
        "compression": "zstd",
        "rows": row_count,
        "bytes": path.stat().st_size,
    }


def read_parquet(
    path: Path,
    *,
    query_id: str | None = None,
) -> list[dict[str, object]]:
    """Read a detail artifact, optionally pushing a query filter into Parquet."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet artifacts require pyarrow from the experiment extra"
        ) from exc

    filters = [("query_id", "=", query_id)] if query_id is not None else None
    table = pq.read_table(path, filters=filters)
    return [_restore_parquet_record(record) for record in table.to_pylist()]


_JSON_COLUMNS = ("source_scores", "source_ranks", "features")


def _parquet_record(record: dict[str, object]) -> dict[str, object]:
    prepared = dict(record)
    for column in _JSON_COLUMNS:
        if column in prepared:
            prepared[f"{column}_json"] = json.dumps(
                prepared.pop(column),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    sources = prepared.get("candidate_sources")
    if isinstance(sources, tuple):
        prepared["candidate_sources"] = list(sources)
    return prepared


def _restore_parquet_record(record: dict[str, object]) -> dict[str, object]:
    restored = dict(record)
    for column in _JSON_COLUMNS:
        encoded = restored.pop(f"{column}_json", None)
        if encoded is not None:
            restored[column] = json.loads(str(encoded))
    return restored


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
