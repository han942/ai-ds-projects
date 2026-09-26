"""Local, cutoff-safe candidate retrieval comparison with quarterly LightGCN."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from shutil import copy2
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.datasets.split import build_seen_user_split
from rating_recsys.db.connection import connect
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings
from rating_recsys.experiments.artifacts import write_json
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_holdout_queries, global_interaction_key
from rating_recsys.experiments.snapshot import code_manifest, write_snapshot
from rating_recsys.retrieval.baselines import BaselineCandidateGenerator, IncrementalRetrievalContext
from rating_recsys.retrieval.lightgcn import LIGHTGCN, LightGCN, LightGCNConfig, LightGCNRRF


CUTOFFS = (20, 50, 100)


def quarter_start(value: date) -> date:
    return date(value.year, 1 + 3 * ((value.month - 1) // 3), 1)


def _rank(ids: tuple[int, ...], target: int) -> int:
    try:
        return ids.index(target) + 1
    except ValueError:
        return 0


def _observations(queries, ranked_ids):
    return tuple(
        RankingObservation(
            query_id=query.query_id, user_id=query.user_id,
            target_restaurant_id=query.target.restaurant_id,
            relevance=query.relevance, history_depth=query.history_depth,
            ordered_restaurant_ids=ranked_ids[query.query_id],
        )
        for query in queries
    )


def _paired_interval(queries, baseline, treatment, cutoff: int) -> dict[str, object]:
    relevant = [query for query in queries if query.relevance > 0]
    base = np.fromiter(
        (_rank(baseline[q.query_id], q.target.restaurant_id) in range(1, cutoff + 1)
         for q in relevant), count=len(relevant), dtype=np.float32,
    )
    changed = np.fromiter(
        (_rank(treatment[q.query_id], q.target.restaurant_id) in range(1, cutoff + 1)
         for q in relevant), count=len(relevant), dtype=np.float32,
    )
    delta = changed - base
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(5):
        samples.extend(delta[rng.integers(len(delta), size=(1000, len(delta)))].mean(axis=1))
    return {
        "baseline": float(base.mean()), "lightgcn": float(changed.mean()),
        "delta": float(delta.mean()),
        "paired_bootstrap_95pct": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "lightgcn_only_hits": int(((changed == 1) & (base == 0)).sum()),
        "baseline_only_hits": int(((base == 1) & (changed == 0)).sum()),
    }


def _evaluate_phase(
    *, phase: str, queries, reference: tuple[Interaction, ...],
    run_dir: Path, config: ExperimentConfig, graph_config: LightGCNConfig,
) -> dict[str, object]:
    context = IncrementalRetrievalContext()
    generator = BaselineCandidateGenerator(
        candidate_k=config.candidate_k, rrf_constant=config.rrf_constant,
        include_region=True,
    )
    fuser = LightGCNRRF(candidate_k=config.candidate_k, rrf_constant=config.rrf_constant)
    ordered_reference = sorted(reference, key=global_interaction_key)
    ordered_queries = sorted(queries, key=lambda q: global_interaction_key(q.target))
    checkpoints = sorted({quarter_start(query.cutoff) for query in queries})
    models = {}
    checkpoint_info = []
    for checkpoint in checkpoints:
        training = tuple(row for row in reference if row.event_date < checkpoint)
        started = perf_counter()
        model = LightGCN(graph_config).fit(training) if len(training) >= 10 else None
        if model is not None and not model.user_embeddings.size:
            model = None
        models[checkpoint] = model
        checkpoint_info.append({
            "checkpoint": checkpoint.isoformat(), "training_edges": len(training),
            "users": len(model.user_ids) if model else 0,
            "items": len(model.item_ids) if model else 0,
            "bpr_loss": model.final_loss if model else None,
            "training_seconds": perf_counter() - started,
        })
        print(f"{phase} {checkpoint}: {len(training)} edges, {checkpoint_info[-1]['training_seconds']:.1f}s", flush=True)
    ranked = {name: {} for name in (
        "popularity", "cosine_cooccurrence", "lightgcn", "rrf_c0_c1", "baseline_c3", "rrf_c0_c1_lightgcn",
    )}
    eligible = set()
    graph_user_queries = 0
    graph_item_queries = 0
    source_hits = Counter()
    source_unique_hits = Counter()
    source_candidates = Counter()
    started = perf_counter()
    index = 0
    output = run_dir / f"candidates_{phase}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for query in ordered_queries:
            cutoff_key = global_interaction_key(query.target)
            while index < len(ordered_reference) and global_interaction_key(ordered_reference[index]) < cutoff_key:
                context.add(ordered_reference[index])
                index += 1
            result, current = generator.retrieve_from_context(query, context.context)
            model = models[quarter_start(query.cutoff)]
            fused, graph_ids = fuser.fuse(query, result, current, model)
            if model is not None and query.user_id in model.user_ids:
                graph_user_queries += 1
            if graph_ids:
                graph_item_queries += 1
            eligible.update(result.eligible_catalog)
            ranked["popularity"][query.query_id] = tuple(x.restaurant_id for x in result.popularity)
            ranked["cosine_cooccurrence"][query.query_id] = tuple(x.restaurant_id for x in result.item_item)
            ranked["lightgcn"][query.query_id] = graph_ids
            ranked["baseline_c3"][query.query_id] = tuple(x.restaurant_id for x in result.union)
            ranked["rrf_c0_c1_lightgcn"][query.query_id] = tuple(x.restaurant_id for x in fused)
            base_ranks = {}
            for ranked_source in (result.popularity, result.item_item):
                for candidate in ranked_source:
                    item = candidate.restaurant_id
                    base_ranks[item] = base_ranks.get(item, 0.0) + 1 / (
                        config.rrf_constant + candidate.candidate_rank
                    )
            ranked["rrf_c0_c1"][query.query_id] = tuple(
                sorted(base_ranks, key=lambda item: (-base_ranks[item], item))[:config.candidate_k]
            )
            if query.relevance > 0:
                for candidate in fused:
                    for source in candidate.candidate_sources:
                        source_candidates[source] += 1
                    if candidate.restaurant_id == query.target.restaurant_id:
                        for source in candidate.candidate_sources:
                            source_hits[source] += 1
                        if len(candidate.candidate_sources) == 1:
                            source_unique_hits[candidate.candidate_sources[0]] += 1
            handle.write(json.dumps({
                "query_id": query.query_id,
                "cutoff": query.cutoff.isoformat(),
                "checkpoint": quarter_start(query.cutoff).isoformat(),
                "target_available": result.target_available,
                "popularity": ranked["popularity"][query.query_id],
                "cosine_cooccurrence": ranked["cosine_cooccurrence"][query.query_id],
                "lightgcn": graph_ids,
                "baseline_c3": ranked["baseline_c3"][query.query_id],
                "rrf_c0_c1": ranked["rrf_c0_c1"][query.query_id],
                "rrf_c0_c1_lightgcn": ranked["rrf_c0_c1_lightgcn"][query.query_id],
                "fused_source_ranks": {str(c.restaurant_id): c.source_ranks for c in fused},
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    item_popularity = dict(Counter(row.restaurant_id for row in reference))
    item_regions = {row.restaurant_id: row.region for row in reference}
    scores = {
        name: evaluate_rankings(
            _observations(queries, ids), cutoffs=CUTOFFS,
            catalog_ids=eligible, item_popularity=item_popularity,
            item_regions=item_regions,
        )
        for name, ids in ranked.items()
    }
    return {
        "queries": len(queries), "relevant_queries": sum(q.relevance > 0 for q in queries),
        "graph_known_user_queries": graph_user_queries,
        "graph_scored_queries": graph_item_queries,
        "eligible_catalog_size": len(eligible),
        "candidate_generation_seconds_including_baselines": perf_counter() - started,
        "checkpoints": checkpoint_info, "sources": scores,
        "lightgcn_contribution": {
            source: {"retrieved_targets": source_hits[source],
                     "unique_targets": source_unique_hits[source],
                     "candidate_appearances": source_candidates[source]}
            for source in ("popularity", "item_item", LIGHTGCN)
        },
        "paired_vs_baseline_c3": {
            f"recall_at_{cutoff}": _paired_interval(
                queries, ranked["baseline_c3"], ranked["rrf_c0_c1_lightgcn"], cutoff
            )
            for cutoff in CUTOFFS
        },
        "paired_vs_c0_c1": {
            f"recall_at_{cutoff}": _paired_interval(
                queries, ranked["rrf_c0_c1"], ranked["rrf_c0_c1_lightgcn"], cutoff
            )
            for cutoff in CUTOFFS
        },
    }


def run_candidate_experiment(interactions: Iterable[Interaction], *, artifacts_root: Path) -> Path:
    all_interactions = tuple(interactions)
    if not all_interactions:
        raise ValueError("No interactions")
    config = ExperimentConfig()
    graph_config = LightGCNConfig()
    primary = build_seen_user_split(all_interactions, minimum_user_items=config.minimum_user_items)
    train = primary.train
    test_history = primary.train + primary.validation
    phases = {
        "validation": (build_holdout_queries(train, primary.validation, config=config, phase="validation"), train),
        "test": (build_holdout_queries(test_history, primary.test, config=config, phase="test"), test_history),
    }
    now = datetime.now(timezone.utc)
    from rating_recsys.experiments.snapshot import dataset_digest
    run_dir = artifacts_root / "comparisons" / f"lightgcn_{now.strftime('%Y%m%dT%H%M%S%fZ')}-{dataset_digest(all_interactions)[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = write_snapshot(all_interactions, run_dir / "dataset.jsonl")
    write_json(run_dir / "config.json", {
        "candidate_k": config.candidate_k, "rrf_constant": config.rrf_constant,
        "graph": asdict(graph_config),
        "model_checkpoint_frequency": "calendar_quarter",
        "graph_edge_rule": "reference interaction event_date < checkpoint start",
        "query_context_rule": "reference interaction global key < query target key",
        "candidate_policy": "Top-100 per source then unweighted three-source RRF Top-100",
        "baseline_policy": "C0+C1 Top-50 preserved, then C0+C1+C2 RRF to Top-100",
        "ranker": "not evaluated; candidate generation experiment",
    })
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    dirty_diff = str(source.pop("git_diff", ""))
    if dirty_diff:
        (run_dir / "source.diff").write_text(dirty_diff, encoding="utf-8")
        source["git_diff_artifact"] = "source.diff"
    write_json(run_dir / "code.json", source)
    source_dir = run_dir / "source"
    source_dir.mkdir()
    for module in ("retrieval/lightgcn.py", "experiments/compare_lightgcn.py"):
        copy2(PROJECT_ROOT / "src" / "rating_recsys" / module,
              source_dir / Path(module).name)
    results = {}
    for phase, (queries, reference) in phases.items():
        results[phase] = _evaluate_phase(
            phase=phase, queries=queries, reference=reference, run_dir=run_dir,
            config=config, graph_config=graph_config,
        )
        write_json(run_dir / f"metrics_{phase}.json", results[phase])
    write_json(run_dir / "manifest.json", {
        "created_at": now.isoformat(), "snapshot": snapshot,
        "primary_split": primary.summary(),
        "artifacts": ["config.json", "code.json", "source.diff", "source/lightgcn.py", "source/compare_lightgcn.py", "dataset.jsonl", "metrics_validation.json", "metrics_test.json", "candidates_validation.jsonl", "candidates_test.jsonl"],
        "mlflow_enabled": False,
    })
    return run_dir


def main() -> None:
    settings = get_settings(require_database=True, require_user_hash_salt=False)
    with connect(settings.database_url) as connection:
        interactions = InteractionRepository(connection).fetch_first_interactions()
    print(f"Loaded {len(interactions)} interactions", flush=True)
    run_dir = run_candidate_experiment(interactions, artifacts_root=PROJECT_ROOT / "artifacts")
    print(f"Completed: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
