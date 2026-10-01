"""Two-stage recommender experiment on a global date-cutoff split.

Protocol
--------
1. One global split by date quantiles: train (<= T1), validation window
   (T1, T2] and test window (> T2).
2. An evaluation query is one user at a window start. Its history is every
   visit of that user up to the window start and its positives are every
   window visit with relevance > 0. Users without history are counted but not
   evaluated.
3. Stage 1 is C5 = RRF(C1 item-item, C4 LightGCN). For a window, LightGCN is
   fitted on every interaction up to the window start. C0, C2 and the previous
   C3 quota union are built from the same context and reported for reference.
4. Ranker training defaults to single-positive prefix queries. Optional
   window queries hold several outcomes and freeze history, context and
   LightGCN before the calendar block starts. Targets stop at T1 while tuning
   and T2 for final refitting. Prefix queries use interactions ordered before
   the target and a graph from before its calendar block. Only queries with
   observed positives among their own Stage 1 candidates and differing labels
   become training groups; positives are never added to candidate lists.
   Optional raw-rating gain changes training labels, not evaluation relevance.
5. Tuning reads only the validation window: a LightGBM grid with early
   stopping selected by NDCG@ranking_k. LightGCN settings are fixed in the
   config (chosen earlier on the validation window).
6. The test window is evaluated exactly once, after every choice is fixed.

Each run writes ``artifacts/runs/<run_id>/`` including ``report.md``.
"""

from __future__ import annotations

import bisect
import gc
import resource
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import (
    RankingObservation,
    evaluate_rankings,
    query_scores,
)
from rating_recsys.evaluation.report import write_report
from rating_recsys.experiments.artifacts import write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import RecommendationQuery, WindowQuery
from rating_recsys.experiments.queries import (
    build_prefix_queries,
    build_training_windows,
    build_window_queries,
    global_interaction_key,
)
from rating_recsys.experiments.snapshot import (
    code_manifest,
    environment_manifest,
    freeze_snapshot,
)
from rating_recsys.ranking.features import build_feature_rows, feature_values
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
from rating_recsys.retrieval.baselines import (
    IncrementalRetrievalContext,
    RetrievalContext,
)
from rating_recsys.retrieval.hybrid import HybridCandidateGenerator
from rating_recsys.retrieval.lightgcn import LightGCN, LightGCNConfig


STAGE1 = "c5_c1_lightgcn_rrf"
CANDIDATE_STAGES = (
    "c0_popularity",
    "c1_item_item",
    "c2_region_popularity",
    "c3_rrf_union",
    "c4_lightgcn",
    STAGE1,
)
RANKING_STAGES = ("r0_candidate_order", "r1_lambdarank")
REFERENCE_NAME = "reference_untuned"
BOOTSTRAP_METRICS = ("ndcg", "recall", "precision", "map")


@dataclass(slots=True)
class TrainingArrays:
    features: np.ndarray
    labels: np.ndarray
    groups: list[int]
    summary: dict[str, int | float | str | None]


@dataclass(slots=True)
class WindowCandidates:
    ordered: dict[str, dict[str, tuple[int, ...]]]
    features: np.ndarray | None
    labels: np.ndarray | None
    row_restaurant_ids: np.ndarray | None
    group_sizes: list[int]
    catalog: set[int]
    item_popularity: dict[int, int]
    item_regions: dict[int, str]
    item_names: dict[int, str]
    positive_availability: dict[str, int | float]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, object]
    manifest: dict[str, object]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def build_context(interactions: Iterable[Interaction]) -> RetrievalContext:
    builder = IncrementalRetrievalContext()
    for item in sorted(interactions, key=global_interaction_key):
        builder.add(item)
    return builder.context


def checkpoint_start(day: date, months: int) -> date:
    """First day of the calendar block (``months`` long, from January) of ``day``."""

    return date(day.year, (day.month - 1) // months * months + 1, 1)


def fit_lightgcn(
    interactions: Sequence[Interaction], config: LightGCNConfig
) -> tuple[LightGCN | None, dict[str, object]]:
    """Fit on ``interactions``; ``None`` when the graph is too small to train."""

    started = time.perf_counter()
    try:
        model = LightGCN(config).fit(interactions)
    except ValueError as error:
        return None, {"edges": len(interactions), "skipped": str(error)}
    model.final_embeddings  # propagate once; later lookups reuse it
    return model, {
        "edges": model.edge_count,
        "users": len(model.user_ids),
        "restaurants": len(model.item_ids),
        "final_bpr_loss": model.history[-1].bpr_loss if model.history else None,
        "fit_seconds": round(time.perf_counter() - started, 3),
    }


class CheckpointedLightGCN:
    """LightGCN models for ranker training queries, refit at block starts.

    A training query dated ``d`` uses the model fitted on the interactions
    dated strictly before the start of ``d``'s calendar block, so its target
    and anything later are outside the graph. Only the latest block's model is
    held; requesting an earlier block again refits it (deterministically) and
    checks that it saw the same edges.
    """

    def __init__(
        self,
        config: LightGCNConfig,
        *,
        months: int,
        log: Callable[[str], None] = lambda _: None,
    ) -> None:
        self.config = config
        self.months = months
        self.log = log
        self.summary: dict[str, dict[str, object]] = {}
        self._block: date | None = None
        self._model: LightGCN | None = None

    def model_for(
        self,
        day: date,
        ordered_reference: Sequence[Interaction],
        reference_dates: Sequence[date],
    ) -> LightGCN | None:
        block = checkpoint_start(day, self.months)
        key = block.isoformat()
        edge_count = bisect.bisect_left(reference_dates, block)
        previous = self.summary.get(key)
        if previous is not None and previous["edges"] != edge_count:
            raise RuntimeError(
                f"Checkpoint {key} was fitted on {previous['edges']} edges, now {edge_count}"
            )
        if block == self._block:
            return self._model
        self._model, info = fit_lightgcn(ordered_reference[:edge_count], self.config)
        self._block = block
        if previous is None:
            self.summary[key] = {"checkpoint": key, "queries": 0, **info}
        self.log(
            f"lightgcn checkpoint {key}: {info['edges']} edges, "
            f"{info.get('fit_seconds', 0)}s"
        )
        return self._model

    def count_query(self, day: date) -> None:
        self.summary[checkpoint_start(day, self.months).isoformat()]["queries"] += 1

    def rows(self) -> list[dict[str, object]]:
        return [self.summary[key] for key in sorted(self.summary)]


def prefix_training_arrays(
    queries: tuple[RecommendationQuery, ...],
    reference_interactions: tuple[Interaction, ...],
    generator: HybridCandidateGenerator,
    graphs: CheckpointedLightGCN,
    *,
    feature_names: tuple[str, ...],
    log: Callable[[str], None] = lambda _: None,
    label: str = "train",
    config: ExperimentConfig | None = None,
) -> TrainingArrays:
    """Cutoff-safe single-positive LambdaRank rows for prefix queries.

    Queries are processed in global chronological order while the retrieval
    context grows, so each query sees only interactions ordered before its
    target. LightGCN comes from ``graphs`` (fitted before the query's block).
    A query becomes a training group only when its positive is among its own
    Stage 1 candidates, exactly as at evaluation. Positives are never added to
    the candidate list: an added row could only sit outside the real list
    (rank ``candidate_k + 1``), and the ranker would learn that position
    instead of the features. Rows inside a group are ordered by restaurant id
    for determinism.
    """

    config = config or ExperimentConfig()
    capacity_per_query = generator.candidate_k
    features = np.empty(
        (len(queries) * capacity_per_query, len(feature_names)), dtype=np.float32
    )
    labels = np.empty(len(queries) * capacity_per_query, dtype=np.int32)
    groups: list[int] = []
    ordered_queries = sorted(queries, key=lambda q: global_interaction_key(q.target))
    ordered_reference = sorted(reference_interactions, key=global_interaction_key)
    reference_dates = [item.event_date for item in ordered_reference]
    builder = IncrementalRetrievalContext()
    index = 0
    offset = 0
    relevant = available = retrieved = graph_queries = 0
    for processed, query in enumerate(ordered_queries, start=1):
        if processed % 10000 == 0:
            log(
                f"{label}: {processed}/{len(queries)} prefix queries; "
                f"max_rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}"
            )
        cutoff = global_interaction_key(query.target)
        while (
            index < len(ordered_reference)
            and global_interaction_key(ordered_reference[index]) < cutoff
        ):
            builder.add(ordered_reference[index])
            index += 1
        model = graphs.model_for(query.target.event_date, ordered_reference, reference_dates)
        graphs.count_query(query.target.event_date)
        ranked = (
            model.top_k(
                [query.user_id],
                [[item.restaurant_id for item in query.history]],
                generator.candidate_k,
            )[0]
            if model is not None
            else ()
        )
        graph_queries += int(bool(ranked))
        context = builder.context
        result = generator.retrieve(query, context, graph_ranked=ranked, scorer=model)
        target_label = config.training_label(query.target.rating)
        if target_label <= 0:
            continue
        relevant += 1
        available += int(result.reference.target_available)
        if not any(
            c.restaurant_id == query.target.restaurant_id for c in result.union
        ):
            continue
        retrieved += 1
        rows = sorted(
            build_feature_rows(
                query, result.union, context, include_region=generator.include_region
            ),
            key=lambda row: row.restaurant_id,
        )
        if not any(row.relevance == 0 for row in rows):
            continue
        for row in rows:
            features[offset] = [row.features[name] for name in feature_names]
            labels[offset] = target_label if row.restaurant_id == query.target.restaurant_id else 0
            offset += 1
        groups.append(len(rows))

    return TrainingArrays(
        features=features[:offset].copy(),
        labels=labels[:offset].copy(),
        groups=groups,
        summary={
            "queries": len(queries),
            "usable_groups": len(groups),
            "feature_rows": offset,
            "relevant_queries": relevant,
            "available_positive_queries": available,
            "retrieved_positive_queries": retrieved,
            "retrieved_positive_rate": retrieved / relevant if relevant else 0.0,
            "lightgcn_scored_queries": graph_queries,
            "lightgcn_scored_rate": graph_queries / len(queries) if queries else 0.0,
            "latest_target_date": (
                max(q.target.event_date for q in queries).isoformat()
                if queries
                else None
            ),
        },
    )


def window_candidates(
    queries: tuple[WindowQuery, ...],
    context: RetrievalContext,
    generator: HybridCandidateGenerator,
    *,
    graph: LightGCN | None = None,
    feature_names: tuple[str, ...] | None,
    training_config: ExperimentConfig | None = None,
) -> WindowCandidates:
    """Candidates (and optionally ranker features) for fixed-cutoff queries.

    ``graph`` must be fitted on interactions up to the window start. Without
    it, C4 is empty and C5 equals C1.
    """

    ordered: dict[str, dict[str, tuple[int, ...]]] = {
        stage: {} for stage in CANDIDATE_STAGES
    }
    graph_ranked = (
        graph.recommend(
            [q.user_id for q in queries],
            {q.user_id: [item.restaurant_id for item in q.history] for q in queries},
            generator.candidate_k,
        )
        if graph is not None
        else {}
    )
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    id_blocks: list[np.ndarray] = []
    group_sizes: list[int] = []
    catalog: set[int] = set()
    positives = available_positives = 0
    for query in queries:
        retrieval_query = query.retrieval_query()
        result = generator.retrieve(
            retrieval_query,
            context,
            graph_ranked=graph_ranked.get(query.user_id, ()),
            scorer=graph,
        )
        reference = result.reference
        for stage, ids in (
            ("c0_popularity", [c.restaurant_id for c in reference.popularity]),
            ("c1_item_item", [c.restaurant_id for c in reference.item_item]),
            ("c2_region_popularity", [c.restaurant_id for c in reference.region_popularity]),
            ("c3_rrf_union", [c.restaurant_id for c in reference.union]),
            ("c4_lightgcn", result.lightgcn),
            (STAGE1, [c.restaurant_id for c in result.union]),
        ):
            ordered[stage][query.query_id] = tuple(ids)
        catalog.update(reference.eligible_catalog)
        for restaurant_id, relevance in query.relevance_by_item.items():
            if relevance > 0:
                positives += 1
                available_positives += int(restaurant_id in context.item_counts)
        group_sizes.append(len(result.union))
        if feature_names is None:
            continue
        feature_blocks.append(
            np.asarray(
                [
                    [values[name] for name in feature_names]
                    for values in (
                        feature_values(
                            retrieval_query,
                            candidate,
                            context,
                            include_region=generator.include_region,
                        )
                        for candidate in result.union
                    )
                ],
                dtype=np.float32,
            ).reshape(len(result.union), len(feature_names))
        )
        training_labels = (
            {visit.restaurant_id: training_config.training_label(visit.rating) for visit in query.window}
            if training_config is not None else query.relevance_by_item
        )
        label_blocks.append(
            np.asarray(
                [training_labels.get(c.restaurant_id, 0) for c in result.union],
                dtype=np.int32,
            )
        )
        id_blocks.append(
            np.asarray([c.restaurant_id for c in result.union], dtype=np.int64)
        )

    has_features = feature_names is not None

    def _stack(blocks: list[np.ndarray], width: int | None, dtype) -> np.ndarray:
        if blocks:
            return np.concatenate(blocks)
        return np.empty((0, width) if width is not None else (0,), dtype=dtype)

    return WindowCandidates(
        ordered=ordered,
        features=(
            _stack(feature_blocks, len(feature_names), np.float32)
            if has_features
            else None
        ),
        labels=_stack(label_blocks, None, np.int32) if has_features else None,
        row_restaurant_ids=_stack(id_blocks, None, np.int64) if has_features else None,
        group_sizes=group_sizes,
        catalog=catalog,
        item_popularity=dict(context.item_counts),
        item_regions=dict(context.item_regions),
        item_names=dict(context.item_names),
        positive_availability={
            "relevant_positives": positives,
            "available_positives": available_positives,
            "rate": available_positives / positives if positives else 0.0,
        },
    )


def window_training_arrays(
    reference_interactions: tuple[Interaction, ...],
    generator: HybridCandidateGenerator,
    graphs: CheckpointedLightGCN,
    *,
    config: ExperimentConfig,
    through: date,
    phase: str = "train",
    log: Callable[[str], None] = lambda _: None,
) -> TrainingArrays:
    """Multiple observed restaurants per query, with cutoff-safe real candidates.

    No target is injected. All labels are outcomes from the following window;
    all features and both retrieval sources are frozen before its start.
    """
    rows = tuple(sorted(reference_interactions, key=global_interaction_key))
    dates = [row.event_date for row in rows]
    builder = IncrementalRetrievalContext()
    index = 0
    feature_blocks, label_blocks, groups = [], [], []
    query_count = relevant = available = retrieved = scored = 0
    multiple = distinct = preference_pairs = observed_count = 0
    for start, queries in build_training_windows(rows, config=config, through=through, phase=phase):
        while index < len(rows) and rows[index].event_date < start:
            builder.add(rows[index])
            index += 1
        graph = graphs.model_for(start, rows, dates)
        candidates = window_candidates(
            queries, builder.context, generator, graph=graph,
            feature_names=config.feature_names, training_config=config,
        )
        offset = 0
        catalog_ids = set(builder.context.item_counts)
        for query, size in zip(queries, candidates.group_sizes, strict=True):
            graphs.count_query(start)
            query_count += 1
            scored += int(bool(candidates.ordered["c4_lightgcn"][query.query_id]))
            labels = candidates.labels[offset:offset + size]
            values = candidates.features[offset:offset + size]
            offset += size
            positives = {
                visit.restaurant_id for visit in query.window
                if config.training_label(visit.rating) > 0
            }
            relevant += int(bool(positives))
            available += int(bool(positives & catalog_ids))
            retrieved += int(bool(size and np.any(labels > 0)))
            if not size or labels.max() <= 0 or labels.max() == labels.min():
                continue
            groups.append(size)
            feature_blocks.append(values)
            label_blocks.append(labels)
            observed = labels[labels > 0]
            observed_count += len(observed)
            multiple += int(len(observed) > 1)
            distinct += int(len(set(observed)) > 1)
            # Counts actual rating comparisons even in the three-bin control.
            selected = set(candidates.ordered[STAGE1][query.query_id])
            ratings = [visit.rating for visit in query.window if visit.restaurant_id in selected]
            preference_pairs += sum(a != b for i, a in enumerate(ratings) for b in ratings[i + 1:])
        log(f"{phase} window {start}: {len(queries)} queries, cumulative usable groups {len(groups)}")
    if not groups:
        raise ValueError("Training windows have no candidate groups with different labels")
    return TrainingArrays(
        features=np.concatenate(feature_blocks), labels=np.concatenate(label_blocks), groups=groups,
        summary={
            "mode": "window", "label_mode": config.ranker_label_mode,
            "queries": query_count, "usable_groups": len(groups), "feature_rows": sum(groups),
            "relevant_queries": relevant, "available_positive_queries": available,
            "retrieved_positive_queries": retrieved,
            "retrieved_positive_rate": retrieved / relevant if relevant else 0.0,
            "lightgcn_scored_queries": scored,
            "lightgcn_scored_rate": scored / query_count if query_count else 0.0,
            "multi_positive_groups": multiple, "distinct_observed_label_groups": distinct,
            "observed_preference_pairs": preference_pairs,
            "mean_retrieved_positives_per_group": observed_count / len(groups),
            "latest_target_date": max(row.event_date for row in rows if row.event_date <= through).isoformat(),
        },
    )


def rank_window(
    ranker: LightGBMLambdaRanker,
    queries: tuple[WindowQuery, ...],
    candidates: WindowCandidates,
) -> dict[str, tuple[tuple[int, float], ...]]:
    """Order by ranker score, preserving each query's C5 order for exact ties."""

    if candidates.features is None or candidates.row_restaurant_ids is None:
        raise ValueError("Window candidates were built without features")
    scores = (
        ranker.predict(candidates.features) if len(candidates.features) else np.empty(0)
    )
    ranked: dict[str, tuple[tuple[int, float], ...]] = {}
    offset = 0
    for query, size in zip(queries, candidates.group_sizes, strict=True):
        ids = candidates.row_restaurant_ids[offset : offset + size]
        values = scores[offset : offset + size]
        c5_ranks = {
            restaurant_id: rank
            for rank, restaurant_id in enumerate(candidates.ordered[STAGE1][query.query_id])
        }
        ranked[query.query_id] = tuple(
            sorted(
                ((int(i), float(s)) for i, s in zip(ids, values, strict=True)),
                key=lambda pair: (-pair[1], c5_ranks[pair[0]], pair[0]),
            )
        )
        offset += size
    return ranked


def positive_eval_set(
    candidates: WindowCandidates,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Validation groups with at least one retrieved positive, for early stopping."""

    if candidates.features is None or candidates.labels is None:
        raise ValueError("Window candidates were built without features")
    row_indexes: list[np.ndarray] = []
    groups: list[int] = []
    offset = 0
    for size in candidates.group_sizes:
        labels = candidates.labels[offset : offset + size]
        if size and labels.max() > 0:
            row_indexes.append(np.arange(offset, offset + size))
            groups.append(size)
        offset += size
    if not groups:
        raise ValueError("Validation window has no retrieved positive for early stopping")
    indexes = np.concatenate(row_indexes)
    return candidates.features[indexes], candidates.labels[indexes], groups


def observations(
    queries: Iterable[WindowQuery],
    ordered: dict[str, tuple[int, ...]],
) -> tuple[RankingObservation, ...]:
    return tuple(
        RankingObservation(
            query_id=query.query_id,
            user_id=query.user_id,
            relevance_by_item=query.relevance_by_item,
            ordered_restaurant_ids=ordered.get(query.query_id, ()),
        )
        for query in queries
    )


def stage_metrics(
    queries: tuple[WindowQuery, ...],
    candidates: WindowCandidates,
    ranked: dict[str, tuple[tuple[int, float], ...]],
    config: ExperimentConfig,
) -> dict[str, object]:
    """C0-C5 at candidate cutoffs; R0 (C5 order) and R1 at ranking cutoffs."""

    common = {
        "catalog_ids": candidates.catalog,
        "item_popularity": candidates.item_popularity,
        "item_regions": candidates.item_regions,
    }
    metrics: dict[str, object] = {
        "positive_availability": candidates.positive_availability
    }
    for stage in CANDIDATE_STAGES:
        metrics[stage] = evaluate_rankings(
            observations(queries, candidates.ordered[stage]),
            cutoffs=config.candidate_cutoffs,
            **common,
        )
    metrics["r0_candidate_order"] = evaluate_rankings(
        observations(queries, candidates.ordered[STAGE1]),
        cutoffs=config.ranking_cutoffs,
        **common,
    )
    metrics["r1_lambdarank"] = evaluate_rankings(
        observations(queries, ranked_ids(ranked)),
        cutoffs=config.ranking_cutoffs,
        **common,
    )
    return metrics


def paired_bootstrap(
    queries: tuple[WindowQuery, ...],
    treatment: dict[str, tuple[int, ...]],
    baseline: dict[str, tuple[int, ...]],
    *,
    cutoff: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Paired bootstrap of per-query differences (treatment − baseline)."""

    relevant = tuple(query for query in queries if query.relevant)
    if not relevant:
        return {"queries": 0}
    treated = observations(relevant, treatment)
    base = observations(relevant, baseline)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(relevant), size=(samples, len(relevant)))
    result: dict[str, object] = {
        "queries": len(relevant),
        "cutoff": cutoff,
        "samples": samples,
    }
    for metric in BOOTSTRAP_METRICS:
        t = np.array([query_scores(o, cutoff)[metric] for o in treated])
        b = np.array([query_scores(o, cutoff)[metric] for o in base])
        deltas = (t - b)[indexes].mean(axis=1)
        low, high = np.percentile(deltas, [2.5, 97.5])
        result[f"{metric}_at_{cutoff}"] = {
            "treatment": float(t.mean()),
            "baseline": float(b.mean()),
            "delta": float((t - b).mean()),
            "ci95": [float(low), float(high)],
            "wins": int((t > b).sum()),
            "losses": int((t < b).sum()),
            "ties": int((t == b).sum()),
        }
    return result


def select_ranker(
    rows: list[dict[str, object]], config: ExperimentConfig
) -> dict[str, object]:
    """Highest validation R1 NDCG@ranking_k; ties by Recall then grid order."""

    k = config.ranking_k
    return max(
        enumerate(rows),
        key=lambda pair: (pair[1][f"ndcg_at_{k}"], pair[1][f"recall_at_{k}"], -pair[0]),
    )[1]


def ranked_ids(
    ranked: dict[str, tuple[tuple[int, float], ...]],
) -> dict[str, tuple[int, ...]]:
    return {
        query_id: tuple(restaurant_id for restaurant_id, _ in items)
        for query_id, items in ranked.items()
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    interactions: Iterable[Interaction],
    *,
    project_root: Path,
    artifacts_root: Path,
    config: ExperimentConfig | None = None,
    label: str | None = None,
    allow_dirty: bool = True,
    log: Callable[[str], None] | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    emit = log or (lambda message: print(message, file=sys.stderr, flush=True))
    all_interactions = tuple(interactions)
    if not all_interactions:
        raise ValueError("Cannot run an experiment without interactions")
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now
        emit(f"[{name}] {timings[name]:.1f}s")

    source = code_manifest(project_root, allow_dirty=allow_dirty)
    snapshot_path, snapshot = freeze_snapshot(
        all_interactions, artifacts_root / "snapshots"
    )
    try:
        snapshot_location = str(snapshot_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        snapshot_location = str(snapshot_path)
    created_at = datetime.now(timezone.utc)
    run_id = (
        created_at.strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{str(snapshot['dataset_snapshot_id'])[:8]}"
    )
    run_dir = artifacts_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    dirty_diff = str(source.pop("git_diff", ""))
    if dirty_diff:
        (run_dir / "source.diff").write_text(dirty_diff, encoding="utf-8")
        source["git_diff_artifact"] = "source.diff"

    # ---- Split and evaluation queries -------------------------------------
    split = build_global_temporal_split(
        all_interactions,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    t1, t2 = split.train_cutoff, split.validation_cutoff
    validation_queries, validation_new = build_window_queries(
        split.train, split.validation, config=config, phase="validation", cutoff=t1
    )
    test_history = split.train + split.validation
    test_queries, test_new = build_window_queries(
        test_history, split.test, config=config, phase="test", cutoff=t2
    )
    feature_names = config.feature_names
    lap("01_split_and_window_queries")

    # ---- Step A: validation-window LightGCN (edges <= T1) -----------------
    generator = _generator(config)
    graph_config = config.lightgcn_config
    validation_graph, validation_graph_info = fit_lightgcn(split.train, graph_config)
    lap("02_fit_validation_lightgcn")

    # ---- Step B: tuning-phase training rows (targets <= T1) ---------------
    checkpoints = CheckpointedLightGCN(
        graph_config, months=config.lightgcn_checkpoint_months, log=emit
    )
    if config.ranker_training_mode == "window":
        train_arrays = window_training_arrays(
            split.train, generator, checkpoints, config=config, through=t1, log=emit,
        )
    else:
        train_queries = build_prefix_queries(split.train, config=config, phase="train")
        _assert_targets_through(train_queries, t1)
        train_arrays = prefix_training_arrays(
            train_queries, split.train, generator, checkpoints,
            feature_names=feature_names, log=emit, label="train_prefix", config=config,
        )
    lap("03_build_train_rows")

    validation_context = build_context(split.train)
    validation_candidates = window_candidates(
        validation_queries, validation_context, generator,
        graph=validation_graph, feature_names=feature_names,
    )
    eval_set = positive_eval_set(validation_candidates)
    lap("04_build_validation_rows")

    # ---- Step C: LightGBM grid with early stopping on validation ----------
    ranker_rows: list[dict[str, object]] = []
    fitted: dict[str, LightGBMLambdaRanker] = {}
    for params in _ranker_grid(config):
        started = time.perf_counter()
        ranker = _ranker(config, params)
        ranker.fit(
            train_arrays.features,
            train_arrays.labels,
            train_arrays.groups,
            eval_set=eval_set,
            early_stopping_rounds=(
                config.early_stopping_rounds if params["early_stopping"] else None
            ),
        )
        ranked = rank_window(ranker, validation_queries, validation_candidates)
        r1 = evaluate_rankings(
            observations(validation_queries, ranked_ids(ranked)),
            cutoffs=config.ranking_cutoffs,
            catalog_ids=validation_candidates.catalog,
        )
        if params["early_stopping"]:
            used_iterations = ranker.best_iteration or int(params["n_estimators"])
        else:
            used_iterations = int(params["n_estimators"])
        row = {
            **params,
            "best_iteration": used_iterations,
            "fit_seconds": round(time.perf_counter() - started, 3),
            "lightgbm_validation_ndcg_curve": ranker.validation_curve,
            **{
                key: value
                for key, value in r1.items()
                if key.split("_at_")[0] in ("recall", "precision", "ndcg", "map", "mrr")
            },
        }
        ranker_rows.append(row)
        fitted[str(params["name"])] = ranker
        emit(
            f"{params['name']}: iters={used_iterations} val NDCG@{config.ranking_k}="
            f"{row[f'ndcg_at_{config.ranking_k}']:.4f} ({row['fit_seconds']:.0f}s)"
        )
    chosen_ranker = select_ranker(ranker_rows, config)
    validation_ranker = fitted[str(chosen_ranker["name"])]
    del fitted
    validation_ranked = rank_window(
        validation_ranker, validation_queries, validation_candidates
    )
    validation_metrics = stage_metrics(
        validation_queries, validation_candidates, validation_ranked, config
    )
    _write_window_artifacts(
        run_dir, "validation", validation_queries, validation_ranked,
        validation_candidates, config,
    )
    del eval_set, validation_candidates, validation_context, validation_ranker
    del validation_graph
    gc.collect()
    lap("05_tune_lightgbm_on_validation")

    # ---- Step D: final refit on train + validation window (targets <= T2) -
    if config.ranker_training_mode == "window":
        # Rebuild the partial T1 calendar window through T2 exactly once.
        train_summary = train_arrays.summary
        del train_arrays
        refit_arrays = window_training_arrays(
            test_history, generator, checkpoints, config=config, through=t2, phase="refit", log=emit,
        )
        final_features, final_labels, final_groups = (
            refit_arrays.features, refit_arrays.labels, refit_arrays.groups,
        )
        training_summary = {"train_window": train_summary, "refit_all_windows": refit_arrays.summary}
        del refit_arrays
    else:
        refit_queries = tuple(
            query
            for query in build_prefix_queries(test_history, config=config, phase="refit")
            if query.target.event_date > t1
        )
        _assert_targets_through(refit_queries, t2)
        refit_arrays = prefix_training_arrays(
            refit_queries, test_history, generator, checkpoints,
            feature_names=feature_names, log=emit, label="refit_prefix", config=config,
        )
        final_features = np.concatenate([train_arrays.features, refit_arrays.features])
        final_labels = np.concatenate([train_arrays.labels, refit_arrays.labels])
        final_groups = train_arrays.groups + refit_arrays.groups
        training_summary = {
            "train_prefix": train_arrays.summary,
            "refit_validation_window_prefix": refit_arrays.summary,
        }
        del train_arrays, refit_arrays
    gc.collect()
    final_params = {
        "num_leaves": chosen_ranker["num_leaves"],
        "min_child_samples": chosen_ranker["min_child_samples"],
        "n_estimators": int(chosen_ranker["best_iteration"]),
    }
    final_ranker = _ranker(config, final_params)
    final_ranker.fit(final_features, final_labels, final_groups)
    final_ranker.save(run_dir / "model.txt")
    feature_importance = final_ranker.feature_importance()
    del final_features, final_labels, final_groups
    gc.collect()
    lap("06_final_refit")

    # ---- Step E: the single test evaluation --------------------------------
    test_graph, test_graph_info = fit_lightgcn(test_history, graph_config)
    test_context = build_context(test_history)
    test_candidates = window_candidates(
        test_queries, test_context, generator,
        graph=test_graph, feature_names=feature_names,
    )
    del test_graph
    test_ranked = rank_window(final_ranker, test_queries, test_candidates)
    test_metrics = stage_metrics(test_queries, test_candidates, test_ranked, config)
    test_metrics["bootstrap_r1_minus_r0"] = paired_bootstrap(
        test_queries,
        ranked_ids(test_ranked),
        test_candidates.ordered[STAGE1],
        cutoff=config.ranking_k,
        samples=config.bootstrap_samples,
        seed=config.random_seed,
    )
    # Evidence for the Stage 1 switch: C5 against the previous C3 union.
    test_metrics["bootstrap_c5_minus_c3"] = paired_bootstrap(
        test_queries,
        test_candidates.ordered[STAGE1],
        test_candidates.ordered["c3_rrf_union"],
        cutoff=config.candidate_k,
        samples=config.bootstrap_samples,
        seed=config.random_seed,
    )
    _write_window_artifacts(
        run_dir, "test", test_queries, test_ranked, test_candidates, config
    )
    lap("07_single_test_evaluation")

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "selection": {
            "rule": {
                "ranker": (
                    f"validation R1 NDCG@{config.ranking_k} 최대 "
                    f"(동률이면 Recall@{config.ranking_k}, 그다음 grid 순서)"
                ),
                "final_refit": (
                    "선택한 설정, 트리 수 = validation best iteration, "
                    f"train + validation의 {config.ranker_training_mode} query로 재학습"
                ),
            },
            "ranker_grid": ranker_rows,
            "chosen_ranker": chosen_ranker["name"],
            "final_ranker_params": final_params,
        },
        "training": training_summary,
        "lightgcn": {
            "config": graph_config.to_dict(),
            "checkpoint_months": config.lightgcn_checkpoint_months,
            "window_models": {
                "validation": validation_graph_info,
                "test": test_graph_info,
            },
            "training_checkpoints": checkpoints.rows(),
        },
        "feature_importance": feature_importance,
    }
    environment = environment_manifest()
    manifest = {
        "run_id": run_id,
        "label": label,
        "created_at": created_at.isoformat(),
        "config": config.to_dict(),
        "snapshot": {**snapshot, "path": snapshot_location},
        "code": source,
        "environment": {
            "python": environment["python"],
            "platform": environment["platform"],
            "packages": environment["packages"],
        },
        "feature_schema": list(feature_names),
        "split": split.summary(),
        "windows": {
            "validation": _window_summary(validation_queries, validation_new, t1),
            "test": _window_summary(test_queries, test_new, t2),
        },
        "leakage_checks": {
            "lightgcn_validation_graph_through": t1.isoformat(),
            "lightgcn_test_graph_through": t2.isoformat(),
            "lightgcn_training_query_graph": (
                "interactions dated before the start of the query's "
                f"{config.lightgcn_checkpoint_months}-month calendar block"
            ),
            "tuning_training_targets_through": t1.isoformat(),
            "refit_training_targets_through": t2.isoformat(),
            "ranker_training_groups": (
                f"{config.ranker_training_mode} queries with observed positives among their own "
                "Stage 1 candidates and different labels; no positive is added to a candidate list"
            ),
            "ranker_training_label_mode": config.ranker_label_mode,
            "evaluation_labels": "fixed relevance thresholds, independent of training labels",
            "validation_context_through": t1.isoformat(),
            "test_context_through": t2.isoformat(),
            "test_evaluations": 1,
        },
        "timings_seconds": timings,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    write_report(run_dir)
    return ExperimentResult(run_id=run_id, run_dir=run_dir, metrics=metrics, manifest=manifest)


def _ranker_grid(config: ExperimentConfig) -> list[dict[str, object]]:
    return [
        {
            "name": REFERENCE_NAME,
            "num_leaves": config.reference_num_leaves,
            "min_child_samples": config.reference_min_child_samples,
            "n_estimators": config.reference_estimators,
            "early_stopping": False,
        }
    ] + [
        {
            "name": f"leaves{leaves}_minchild{min_child}",
            "num_leaves": leaves,
            "min_child_samples": min_child,
            "n_estimators": config.max_estimators,
            "early_stopping": True,
        }
        for leaves in config.num_leaves_grid
        for min_child in config.min_child_samples_grid
    ]


def _generator(config: ExperimentConfig) -> HybridCandidateGenerator:
    return HybridCandidateGenerator(
        candidate_k=config.candidate_k,
        rrf_constant=config.rrf_constant,
        include_region=config.include_region,
        legacy_c3_quota=config.legacy_c3_quota,
    )


def _ranker(config: ExperimentConfig, params: dict[str, object]) -> LightGBMLambdaRanker:
    return LightGBMLambdaRanker(
        random_seed=config.random_seed,
        ranking_k=config.ranking_k,
        feature_names=config.feature_names,
        n_estimators=int(params["n_estimators"]),
        learning_rate=config.learning_rate,
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=config.reg_lambda,
        n_jobs=config.n_jobs,
        label_gain=config.label_gain,
    )


def _assert_targets_through(
    queries: tuple[RecommendationQuery, ...],
    cutoff: date,
) -> None:
    late = [query.query_id for query in queries if query.target.event_date > cutoff]
    if late:
        raise RuntimeError(
            f"{len(late)} training queries have targets after {cutoff}: {late[:3]}"
        )


def _window_summary(
    queries: tuple[WindowQuery, ...],
    new_users: int,
    cutoff: date,
) -> dict[str, object]:
    positives = [
        sum(value > 0 for value in query.relevance_by_item.values())
        for query in queries
        if query.relevant
    ]
    return {
        "history_through": cutoff.isoformat(),
        "seen_users": len(queries),
        "evaluated_users": len(positives),
        "new_users_not_evaluated": new_users,
        "relevant_positives": sum(positives),
        "mean_relevant_per_user": sum(positives) / len(positives) if positives else 0.0,
    }


def _write_window_artifacts(
    run_dir: Path,
    phase: str,
    queries: tuple[WindowQuery, ...],
    ranked: dict[str, tuple[tuple[int, float], ...]],
    candidates: WindowCandidates,
    config: ExperimentConfig,
) -> None:
    candidate_rank = {
        query_id: {restaurant_id: rank for rank, restaurant_id in enumerate(ids, 1)}
        for query_id, ids in candidates.ordered[STAGE1].items()
    }
    write_jsonl(
        run_dir / f"queries_{phase}.jsonl",
        (
            {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "cutoff": query.cutoff.isoformat(),
                "history": [
                    {
                        "restaurant_id": item.restaurant_id,
                        "restaurant_name": item.restaurant_name,
                        "event_date": item.event_date.isoformat(),
                        "rating": item.rating,
                    }
                    for item in query.history
                ],
                "window": [
                    {
                        "restaurant_id": item.restaurant_id,
                        "restaurant_name": item.restaurant_name,
                        "event_date": item.event_date.isoformat(),
                        "rating": item.rating,
                        "relevance": query.relevance_by_item[item.restaurant_id],
                    }
                    for item in query.window
                ],
            }
            for query in queries
        ),
    )
    write_jsonl(
        run_dir / f"recommendations_{phase}.jsonl",
        (
            {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "recommendations": [
                    {
                        "final_rank": rank,
                        "restaurant_id": restaurant_id,
                        "restaurant_name": candidates.item_names.get(restaurant_id),
                        "region": candidates.item_regions.get(restaurant_id),
                        "ranking_score": score,
                        "candidate_rank": candidate_rank[query.query_id].get(
                            restaurant_id
                        ),
                    }
                    for rank, (restaurant_id, score) in enumerate(
                        ranked.get(query.query_id, ())[: config.ranking_k], start=1
                    )
                ],
            }
            for query in queries
        ),
    )
