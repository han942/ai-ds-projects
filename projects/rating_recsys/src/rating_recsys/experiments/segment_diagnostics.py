"""Recompute validation sparsity segments from saved candidate ranks only."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings
from rating_recsys.evaluation.segments import analyze_segments, validate_saved_rows
from rating_recsys.experiments.artifacts import read_json, read_jsonl, write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.prepared import _file_digest
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import dataset_digest, load_snapshot

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/text_candidate_views/20261010T234421199403Z-e7896add")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/diagnostics/segments")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    m = read_json(args.source_run / "metrics.json")
    if m["status"] != "complete" or m["test_evaluated"]:
        raise ValueError("Expected a complete validation-only source run")
    snapshot = Path(m["sources"]["snapshot"]["path"])
    if _file_digest(snapshot) != m["sources"]["snapshot"]["sha256"]:
        raise ValueError("Source snapshot hash differs")
    rank_path = args.source_run / "candidates_validation.jsonl"
    if _file_digest(rank_path) != m["artifacts_sha256"][rank_path.name]:
        raise ValueError("Saved ranks hash differs")
    rows = load_snapshot(snapshot)
    if dataset_digest(rows) != m["snapshot_id"]:
        raise ValueError("Snapshot identity differs")
    config = dict(m["experiment"])
    for key in ("num_leaves_grid", "min_child_samples_grid"):
        config[key] = tuple(config[key])
    config = ExperimentConfig(**config)
    if config.candidate_k != 100 or config.ranking_k != 10:
        raise ValueError("Diagnostics require source candidate@100 and rank@10")
    split = build_global_temporal_split(rows, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction)
    if split.train_cutoff.isoformat() != m["train_cutoff"] or split.validation_cutoff.isoformat() != m["validation_through"]:
        raise ValueError("Source cutoffs differ")
    queries, new = build_window_queries(split.train, split.validation, config=config, phase="validation", cutoff=split.train_cutoff)
    if len(queries) != m["validation_queries"] or new != m["new_validation_users_excluded"]:
        raise ValueError("Source query population differs")
    arms = tuple(m["metrics"])
    ranks = validate_saved_rows(queries, read_jsonl(rank_path), arms)
    popularity = Counter(row.restaurant_id for row in split.train)
    for arm in arms:
        observations = [RankingObservation(q.query_id, q.user_id, q.relevance_by_item, ranks[arm][q.query_id]) for q in queries]
        metrics = evaluate_rankings(observations, cutoffs=(10,100), catalog_ids=popularity)
        for key in ("queries", "evaluated_queries", "recall_at_100", "ndcg_at_10"):
            if abs(metrics[key] - m["metrics"][arm][key]) > 1e-12:
                raise ValueError(f"Overall source metric differs: {arm}/{key}")
    result, changes = analyze_segments(queries, ranks, popularity, samples=args.samples, seed=args.seed)
    result.update(version="saved-validation-segments-v1", status="complete", source_run=str(args.source_run.resolve()),
                  snapshot_id=m["snapshot_id"], train_cutoff=m["train_cutoff"], validation_through=m["validation_through"],
                  source_ranks_sha256=_file_digest(rank_path), source_metrics_sha256=_file_digest(args.source_run / "metrics.json"),
                  new_validation_users_excluded=new, train_rows=len(split.train), catalog=len(popularity),
                  source_metrics_reproduced=True, source_labels_rebuilt=True,
                  new_training=False, new_predictions=False, test_evaluated=False, embedding_calls=0, api_calls=0)
    run = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + m["snapshot_id"][:8])
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "metrics.json", result)
    write_jsonl(run / "positive_changes.jsonl", changes)
    for path in (Path(__file__), PROJECT_ROOT / "src/rating_recsys/evaluation/segments.py"):
        shutil.copy2(path, run / path.name)
    print(run, flush=True)
    return result

if __name__ == "__main__":
    main()
