"""Fixed-budget, matched behavior versus behavior+E5 towers, validation only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings, query_scores
from rating_recsys.evaluation.segments import analyze_segments
from rating_recsys.experiments.artifacts import read_json, write_json, write_jsonl
from rating_recsys.experiments.candidate_models import TwoTowerCandidate
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.review_multivector_cli import (
    load_prepared, protect_ranker, paired_change, candidate_changes, validate_rankings,
)
from rating_recsys.experiments.snapshot import (
    code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path,
)
from rating_recsys.retrieval.two_tower import BehavioralProfiles, CachedProfiles, TwoTower, TwoTowerConfig

VERSION = "matched-two-tower-behavior-text-v1"


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def conditional_seed_mean_change(queries, pairs, cutoff, metric, samples=2000, seed=42):
    """Resample users after averaging paired deltas over the fixed training seeds."""
    deltas = []
    for query in queries:
        if not query.relevant:
            continue
        differences = []
        for before, after in pairs:
            observations = [RankingObservation(query.query_id, query.user_id, query.relevance_by_item,
                                               tuple(ranks[query.query_id])) for ranks in (before, after)]
            differences.append(query_scores(observations[1], cutoff)[metric]
                               - query_scores(observations[0], cutoff)[metric])
        deltas.append(float(np.mean(differences)))
    if not deltas:
        return {"mean_difference": None, "bootstrap_95pct": None, "paired_users": 0}
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(len(values), size=len(values))].mean()) for _ in range(samples)]
    return {"mean_difference": float(values.mean()), "bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
            "paired_users": len(values), "training_seeds": len(pairs), "samples": samples,
            "interpretation": "user bootstrap conditional on these training seeds; no independent user×seed pooling"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=PROJECT_ROOT / "artifacts/snapshots/e7896add5b4b5939.jsonl")
    p.add_argument("--baseline-run", type=Path, default=PROJECT_ROOT / "artifacts/runs/20261004T125314601899Z-e7896add")
    p.add_argument("--prepared-validation", type=Path, default=PROJECT_ROOT / "artifacts/prepared/e7896add5b4b5939/validation-1d1692022f92bc38/manifest.json")
    p.add_argument("--embedding-run", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/two_tower_ablation")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--threads", type=int, default=4)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epochs < 1 or args.threads < 1 or len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        raise ValueError("Require positive budget and unique nonnegative seeds")
    started = time.perf_counter()
    baseline = read_json(args.baseline_run / "manifest.json")
    settings = dict(baseline["config"])
    for key in ("num_leaves_grid", "min_child_samples_grid"):
        settings[key] = tuple(settings[key])
    exp = ExperimentConfig(**settings)
    if exp.candidate_k != 100 or exp.ranking_k != 10:
        raise ValueError("Matched diagnostics require candidate@100 and rank@10")
    rows = load_snapshot(args.snapshot)
    snapshot_id = dataset_digest(rows)
    if snapshot_id != baseline["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Snapshot differs from baseline")
    split = build_global_temporal_split(rows, train_fraction=exp.train_fraction,
                                        validation_fraction=exp.validation_fraction)
    queries, new_users = build_window_queries(split.train, split.validation, config=exp,
                                               phase="validation", cutoff=split.train_cutoff)
    _, c5 = load_prepared(args.prepared_validation, snapshot_id, split, exp, queries)
    texts, text_meta = load_review_texts(review_texts_path(args.snapshot), rows)
    candidate = TwoTowerCandidate()
    candidate.embedding_run = args.embedding_run
    candidate.prepare_experiment(exp, {"dataset_snapshot_id": snapshot_id}, text_meta)
    # Review bodies outside the train context are not made available to fitting.
    train_texts = {r.review_id: texts[r.review_id] for r in split.train}
    catalog = {r.restaurant_id for r in split.train}
    user_ids = [q.user_id for q in queries]
    excluded = {q.user_id: {r.restaurant_id for r in q.history} for q in queries}
    popularity = Counter(r.restaurant_id for r in split.train)
    run = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + snapshot_id[:8])
    run.mkdir(parents=True, exist_ok=False)
    protected = protect_ranker(args.baseline_run)
    cache_before = digest_file(candidate.embedding.cache_path)
    state = {
        "version": VERSION, "status": "training", "snapshot_id": snapshot_id,
        "train_cutoff": split.train_cutoff.isoformat(), "validation_through": split.validation_cutoff.isoformat(),
        "train_rows": len(split.train), "validation_queries": len(queries),
        "relevant_validation_queries": sum(bool(q.relevant) for q in queries),
        "new_validation_users_excluded": new_users, "catalog": len(catalog),
        "experiment": exp.to_dict(), "seeds": args.seeds, "epochs": args.epochs,
        "selection": "fixed final epoch for both arms; all seeds reported; no early stopping or winner activation",
        "evidence_status": "exploratory-repeated-validation", "test_evaluated": False,
        "ranker_trained": False, "ranker_predicted": False, "api_calls": 0, "db_queries": 0,
        "embedding_inference": 0, "encoder_loaded": False, "arms": {}, "paired": {},
        "segments": {},
        "frozen_embedding_cache_sha256_before": cache_before,
        "command": [sys.executable, "-m", "rating_recsys.experiments.two_tower_ablation_cli", *(argv or sys.argv[1:])],
        "provenance": candidate.provenance,
        "sources": {name: {"path": str(path.resolve()), "sha256": digest_file(path)} for name, path in (
            ("snapshot", args.snapshot), ("review_texts", review_texts_path(args.snapshot)),
            ("prepared", args.prepared_validation), ("baseline", args.baseline_run / "manifest.json"),
            ("embedding_source", args.embedding_run / "manifest.json"))},
    }
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    (run / "source.diff").write_text(source.pop("git_diff"))
    state["code"] = source
    for name in ("retrieval/two_tower.py", "experiments/two_tower_ablation_cli.py"):
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / "src/rating_recsys" / name, destination)
    arms = {"c5": c5}
    def score(ranks):
        return evaluate_rankings([RankingObservation(q.query_id, q.user_id, q.relevance_by_item,
                                                     tuple(ranks[q.query_id])) for q in queries],
                                  cutoffs=(10, 20, 50, 100), catalog_ids=catalog, item_popularity=popularity)
    state["c5"] = score(c5)
    def update():
        write_json(run / "progress.json", state)
    def save_ranks():
        write_jsonl(run / "candidates_validation.jsonl", ({"query_id": q.query_id, "user_id": q.user_id,
            "relevance_by_item": q.relevance_by_item, **{name: value[q.query_id] for name, value in arms.items()}}
            for q in queries))
    print(run, flush=True)
    update()
    try:
        with CachedProfiles(candidate.embedding, candidate.formatter) as text_reader, threadpool_limits(limits=1, user_api="blas"):
            for seed in args.seeds:
                summaries = {}
                for mode in ("behavior", "behavior_text"):
                    name = f"{mode}_seed{seed}"
                    part_started = time.perf_counter()
                    model_config = TwoTowerConfig(input_mode=mode, epochs=args.epochs, seed=seed,
                                                   threads=args.threads, window_months=exp.lightgcn_checkpoint_months)
                    curve = []
                    reader = BehavioralProfiles(mode, text_reader if mode == "behavior_text" else None)
                    def callback(stats, fitted):
                        ranked = fitted.recommend(user_ids, excluded, exp.candidate_k)
                        metrics = score({q.query_id: ranked[q.user_id] for q in queries})
                        curve.append({**asdict(stats), "recall_at_100": metrics["recall_at_100"],
                                      "ndcg_at_10": metrics["ndcg_at_10"]})
                        print(f"{name} epoch {stats.epoch}/{args.epochs}: Recall100={metrics['recall_at_100']:.6f} NDCG10={metrics['ndcg_at_10']:.6f}", flush=True)
                        state["active_arm"] = {"name": name, "curve": curve}
                        update()
                        return False
                    fitted = TwoTower(model_config, embedding_config=candidate.embedding,
                        formatter=candidate.formatter, profiles=reader, eval_user_ids=user_ids,
                        experiment_config=exp).fit(split.train, train_texts if mode == "behavior_text" else {}, callback=callback)
                    rankings = fitted.recommend(user_ids, excluded, exp.candidate_k)
                    arms[name] = {q.query_id: rankings[q.user_id] for q in queries}
                    validate_rankings(queries, {name: arms[name]}, catalog, exp.candidate_k)
                    summary = {"config": model_config.to_dict(), "curve": curve, "metrics": score(arms[name]),
                        "metadata": fitted.metadata, "training_audit": fitted.training_audit,
                        "checkpoint": fitted.save(run / f"{name}.pt"), "seconds": time.perf_counter() - part_started}
                    summaries[mode] = summary
                    state["arms"][name] = summary
                    save_ranks()
                    update()
                left, right = summaries["behavior"]["metadata"], summaries["behavior_text"]["metadata"]
                matched = {key: left[key] == right[key] for key in (
                    "initial_parameters_sha256", "parameter_count", "training_sequence_sha256", "training_positives")}
                if not all(matched.values()):
                    raise ValueError("Training conditions are not matched")
                before, after = (arms[f"{mode}_seed{seed}"] for mode in ("behavior", "behavior_text"))
                state["paired"][str(seed)] = {"matched": matched,
                    "recall_at_100": paired_change(queries, before, after, 100, "recall"),
                    "ndcg_at_10": paired_change(queries, before, after, 10, "ndcg"),
                    "candidate_changes": candidate_changes(queries, before, after)[0]}
                segments, changes = analyze_segments(queries, {"behavior": before, "behavior_text": after},
                                                     popularity, baseline="behavior")
                segment_path = run / f"segments_seed{seed}.json"
                write_json(segment_path, segments)
                write_jsonl(run / f"positive_changes_seed{seed}.jsonl", changes)
                state["segments"][str(seed)] = {"path": segment_path.name, "sha256": digest_file(segment_path)}
                update()
        pairs = [(arms[f"behavior_seed{seed}"], arms[f"behavior_text_seed{seed}"]) for seed in args.seeds]
        state["seed_mean_paired"] = {"recall_at_100": conditional_seed_mean_change(queries, pairs, 100, "recall"),
                                      "ndcg_at_10": conditional_seed_mean_change(queries, pairs, 10, "ndcg")}
        state["ranker_unchanged"] = protected == protect_ranker(args.baseline_run)
        state["frozen_embedding_cache_sha256_after"] = digest_file(candidate.embedding.cache_path)
        if not state["ranker_unchanged"] or state["frozen_embedding_cache_sha256_after"] != cache_before:
            raise ValueError("Protected source artifact changed")
        state.update(status="complete", elapsed_seconds=time.perf_counter() - started)
        state.pop("active_arm", None)
        state["artifacts_sha256"] = {p.name: digest_file(p) for p in sorted(run.glob("*.pt"))}
        state["artifacts_sha256"]["candidates_validation.jsonl"] = digest_file(run / "candidates_validation.jsonl")
        write_json(run / "metrics.json", state)
        update()
        print(json.dumps({"run": str(run), "seed_mean_paired": state["seed_mean_paired"]}, ensure_ascii=False, indent=2))
        return state
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        update()
        raise


def console_main():
    main()
    return 0


if __name__ == "__main__":
    console_main()
