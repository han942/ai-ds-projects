"""Candidate-only validation: same frozen E5 bank, mean vs review matching.

Reads an existing local review Transformer checkpoint solely for its E5 store.
No ranker fitting/prediction, neural training, E5 inference, DB/API call, or
test evaluation occurs. All candidate arms and fusion settings are fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
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
from rating_recsys.experiments.artifacts import read_json, write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import canonical_json, code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path
from rating_recsys.experiments.prepared import _file_digest
from rating_recsys.retrieval.hybrid import rrf
from rating_recsys.retrieval.review_multivector import (
    ReviewMultiVectorConfig, ReviewMultiVectorRetriever, build_review_bank,
    review_vectors_from_store, supplement_candidates,
)

VERSION = "review-multivector-candidate-v1"


def log(value):
    print(value, flush=True)


def compatible_prepared(manifest, snapshot_id, split, config):
    identity = manifest["identity"]
    return all((
        identity.get("kind") == "validation",
        identity.get("snapshot_id") == snapshot_id,
        identity.get("cutoffs") == split.summary()["cutoffs"],
        identity.get("candidate_k") == config.candidate_k,
        identity.get("rrf_constant") == config.rrf_constant,
        identity.get("satisfaction") == {name: value for name, value in config.to_dict().items()
                                         if name.startswith("satisfaction_")},
        identity.get("lightgcn") == config.lightgcn_config.to_dict(),
        identity.get("features") == list(config.feature_names),
        identity.get("region_mode") == config.region_mode,
        identity.get("relevance_thresholds") == [config.relevance_low_threshold, config.relevance_high_threshold],
        identity.get("rating_shrinkage_strength") == config.rating_shrinkage_strength,
    ))


def load_prepared(path, snapshot_id, split, config, queries):
    manifest = read_json(path)
    if not compatible_prepared(manifest, snapshot_id, split, config):
        raise ValueError("Prepared validation conditions differ from the baseline")
    unhashed = {key: value for key, value in manifest.items() if key != "metadata_sha256"}
    import hashlib
    if hashlib.sha256(canonical_json(unhashed).encode()).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("Prepared manifest checksum mismatch")
    if _file_digest(path.parent / "data.npz") != manifest["arrays_sha256"]:
        raise ValueError("Prepared array checksum mismatch")
    ordered = manifest["data"]["ordered"]
    c5 = {key: tuple(value) for key, value in ordered["c5_c1_lightgcn_rrf"].items()}
    if set(c5) != {query.query_id for query in queries}:
        raise ValueError("Prepared query population differs")
    for query_id, ranks in c5.items():
        reproduced = rrf([ordered["c1_item_item"][query_id], ordered["c4_lightgcn"][query_id]],
                         k=config.candidate_k, constant=config.rrf_constant)
        if ranks != reproduced:
            raise ValueError("Stored C5 differs from C1 + LightGCN RRF")
    return manifest, c5


def find_prepared(root, snapshot_id, split, config):
    for path in sorted((root / snapshot_id[:16]).glob("validation-*/manifest.json")):
        if compatible_prepared(read_json(path), snapshot_id, split, config):
            return path
    raise FileNotFoundError("No compatible prepared validation candidates; do not rebuild the ranker")


def protect_ranker(baseline_run):
    paths = [baseline_run / "model.txt", *sorted((PROJECT_ROOT / "src/rating_recsys/ranking").glob("*.py"))]
    return {str(path.resolve()): _file_digest(path) for path in paths}


def paired_change(queries, before, after, cutoff, metric, *, samples=2000, seed=42):
    differences = []
    for query in queries:
        if not query.relevant:
            continue
        def value(ranks):
            row = RankingObservation(query.query_id, query.user_id, query.relevance_by_item, tuple(ranks))
            return query_scores(row, cutoff)[metric]
        differences.append(value(after[query.query_id]) - value(before[query.query_id]))
    values = np.asarray(differences, dtype=np.float64)
    if not len(values):
        return {"mean_difference": None, "bootstrap_95pct": None, "paired_users": 0}
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(samples)])
    return {"mean_difference": float(values.mean()), "bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
            "paired_users": len(values), "samples": samples, "seed": seed}


def candidate_changes(queries, baseline, after):
    rows = []
    for query in queries:
        truth = {item for item, grade in query.relevance_by_item.items() if grade > 0}
        before, new = set(baseline[query.query_id]), set(after[query.query_id])
        gained, lost = truth & (new-before), truth & (before-new)
        rows.append({"query_id": query.query_id, "added_candidates": len(new-before),
                     "removed_candidates": len(before-new), "gained_relevant_items": len(gained),
                     "lost_relevant_items": len(lost), "relevant_items": len(truth)})
    valid = [row for row in rows if row["relevant_items"]]
    summary = {"queries": len(rows), "evaluated_queries": len(valid),
               "queries_with_relevant_gain": sum(row["gained_relevant_items"] > 0 for row in valid),
               "queries_with_relevant_loss": sum(row["lost_relevant_items"] > 0 for row in valid),
               "gained_relevant_items": sum(row["gained_relevant_items"] for row in valid),
               "lost_relevant_items": sum(row["lost_relevant_items"] for row in valid),
               "mean_added_candidates_all_queries": float(np.mean([row["added_candidates"] for row in rows])),
               "mean_removed_candidates_all_queries": float(np.mean([row["removed_candidates"] for row in rows])),
               "macro_recall_gain": float(np.mean([row["gained_relevant_items"]/row["relevant_items"] for row in valid])),
               "macro_recall_loss": float(np.mean([row["lost_relevant_items"]/row["relevant_items"] for row in valid]))}
    return summary, rows


def validate_rankings(queries, arms, catalog, k):
    for name, rankings in arms.items():
        if set(rankings) != {query.query_id for query in queries}:
            raise ValueError(f"Query IDs differ in {name}")
        for query in queries:
            ids = rankings[query.query_id]
            seen = {row.restaurant_id for row in query.history}
            if len(ids) > k or len(ids) != len(set(ids)) or set(ids)-catalog or set(ids)&seen:
                raise ValueError(f"Invalid candidate list in {name}")


def count_bias(bank, scores, rankings):
    counts = (bank.item_review_ids >= 0).sum(1)
    bins = (("missing", counts == 0), ("1-2", (counts >= 1) & (counts <= 2)),
            ("3-9", (counts >= 3) & (counts <= 9)), ("10-30", counts >= 10))
    result = {}
    for name, values in scores.items():
        exposures = Counter(item for ordered in rankings[name].values() for item in ordered)
        rows = []
        for label, mask in bins:
            finite = values[:, mask]
            finite = finite[np.isfinite(finite)]
            rows.append({"review_count_bin": label, "catalog_items": int(mask.sum()),
                         "mean_unmasked_catalog_score": float(finite.mean()) if len(finite) else None,
                         "top100_exposures": sum(exposures[int(item)] for item in bank.item_ids[mask])})
        result[name] = rows
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot", type=Path, default=PROJECT_ROOT / "artifacts/snapshots/e7896add5b4b5939.jsonl")
    result.add_argument("--baseline-run", type=Path, default=PROJECT_ROOT / "artifacts/runs/20261004T125314601899Z-e7896add")
    result.add_argument("--prepared-validation", type=Path, help="Compatible manifest; otherwise discovered by identity")
    result.add_argument("--frozen-checkpoint", type=Path, help="Local Transformer checkpoint; only its E5 store is used")
    result.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/review_multivector")
    result.add_argument("--matching-k", type=int, default=3)
    result.add_argument("--text-slots", type=int, default=10)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--user-batch-size", type=int, default=16)
    result.add_argument("--item-batch-size", type=int, default=256)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = ReviewMultiVectorConfig(matching_k=args.matching_k, user_batch_size=args.user_batch_size,
                                    item_batch_size=args.item_batch_size)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    started = time.perf_counter()
    baseline_manifest = read_json(args.baseline_run / "manifest.json")
    baseline_config = dict(baseline_manifest["config"])
    for key in ("num_leaves_grid", "min_child_samples_grid"):
        baseline_config[key] = tuple(baseline_config[key])
    experiment = ExperimentConfig(**baseline_config)
    if not 0 <= args.text_slots <= experiment.candidate_k:
        raise ValueError("text_slots exceeds candidate budget")
    protected = protect_ranker(args.baseline_run)
    rows = load_snapshot(args.snapshot)
    digest = dataset_digest(rows)
    if digest != baseline_manifest["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Snapshot differs from the baseline")
    split = build_global_temporal_split(rows, train_fraction=experiment.train_fraction,
                                        validation_fraction=experiment.validation_fraction)
    queries, new_users = build_window_queries(split.train, split.validation, config=experiment,
                                              phase="validation", cutoff=split.train_cutoff)
    prepared_path = args.prepared_validation or find_prepared(PROJECT_ROOT / "artifacts/prepared", digest, split, experiment)
    prepared, c5 = load_prepared(prepared_path, digest, split, experiment, queries)
    catalog = {row.restaurant_id for row in split.train}
    if set(prepared["data"]["catalog"]) != catalog:
        raise ValueError("Prepared past catalog differs")
    if args.frozen_checkpoint is None:
        checkpoints = sorted((PROJECT_ROOT / "artifacts/comparisons/review_transformer").glob("*/review_transformer_validation.pt"))
        if not checkpoints:
            raise FileNotFoundError("A local frozen E5 review store is required")
        args.frozen_checkpoint = checkpoints[-1]
    run = args.output_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + digest[:8])
    run.mkdir(parents=True, exist_ok=False)
    state = {"version": VERSION, "status": "preparing", "created_at": datetime.now(timezone.utc).isoformat(),
             "snapshot_id": digest, "snapshot_sha256": _file_digest(args.snapshot),
             "review_config": config.to_dict(), "evaluation_config": experiment.to_dict(),
             "primary_metric": (f"candidate Recall@{experiment.candidate_k} on all "
                                f"{sum(bool(query.relevant) for query in queries):,} relevant validation queries"),
             "prespecified_comparison": "late vs mean on the identical frozen bank; all fixed arms reported, no grid selection",
             "fusion": {"text_slots": args.text_slots, "rrf_constant": experiment.rrf_constant},
             "sources": {"snapshot": str(args.snapshot.resolve()), "baseline_run": str(args.baseline_run.resolve()),
                         "prepared_validation": str(prepared_path.resolve()), "prepared_sha256": _file_digest(prepared_path),
                         "frozen_checkpoint": str(args.frozen_checkpoint.resolve()), "frozen_checkpoint_sha256": _file_digest(args.frozen_checkpoint)},
             "ranker_hashes_before": protected, "ranker_trained": False, "ranker_predicted": False,
             "test_evaluated": False, "neural_training_steps": 0, "embedding_inference_calls": 0,
             "db_queries": 0, "api_calls": 0, "blas_threads": args.threads,
             "train_cutoff": split.train_cutoff.isoformat(), "validation_through": split.validation_cutoff.isoformat(),
             "train_rows": len(split.train), "validation_rows": len(split.validation), "catalog": len(catalog),
             "validation_queries": len(queries), "new_validation_users_excluded": new_users,
             "command": [sys.executable, "-m", "rating_recsys.experiments.review_multivector_cli", *(argv or sys.argv[1:])]}
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    (run / "source.diff").write_text(source.pop("git_diff"))
    state["code"] = source
    for name in ("retrieval/review_multivector.py", "experiments/review_multivector_cli.py"):
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / "src/rating_recsys" / name, destination)
    write_json(run / "progress.json", state)
    log(f"[candidate-only review matching] {run}")
    try:
        import hashlib
        import torch
        torch.set_num_threads(args.threads)
        texts, text_metadata = load_review_texts(review_texts_path(args.snapshot), rows)
        saved = torch.load(args.frozen_checkpoint, map_location="cpu", weights_only=False)
        store = saved["store"]
        past_reviews = {row.review_id for row in split.train}
        records = store.manifest["records"]
        if set(store.review_index) != {record["review_id"] for record in records} or not set(store.review_index) <= past_reviews:
            raise ValueError("Frozen review store contains non-training or inconsistent reviews")
        if hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != store.manifest["input_manifest_sha256"]:
            raise ValueError("Frozen input manifest checksum differs")
        for record in records:
            text = texts[record["review_id"]]
            if not text or any(text[block["start"]:block["end"]] != block["text"] for block in record["blocks"]):
                raise ValueError("Frozen review spans differ from the current sidecar")
        state["review_texts"] = text_metadata
        state["frozen_store"] = {name: store.manifest[name] for name in ("input_manifest_sha256", "cache_identity", "block_config", "reviews")}
        query_vectors, item_vectors = review_vectors_from_store(store)
        bank = build_review_bank(split.train, split.train_cutoff, store.review_index, query_vectors, item_vectors,
                                 query_users=[query.user_id for query in queries], config=config)
        del saved, store, texts, records, query_vectors, item_vectors
        gc.collect()
        bank.save(run / "review_bank.npz")
        write_json(run / "profiles_validation.json", bank.audit)
        state.update(status="scoring", profiles=bank.audit, preparation_seconds=time.perf_counter()-started)
        write_json(run / "progress.json", state)
        retriever = ReviewMultiVectorRetriever(bank, config)
        score_started = time.perf_counter()
        def progress(completed, total):
            state.update(scored_users=completed, scoring_seconds=time.perf_counter()-score_started)
            write_json(run / "progress.json", state)
            if completed == total or completed % 128 == 0 or completed == config.user_batch_size:
                log(f"review matching: {completed}/{total} users; {time.perf_counter()-score_started:.1f}s")
        with threadpool_limits(limits=args.threads, user_api="blas"):
            scores = retriever.score_all(progress=progress)
        state["scoring_seconds"] = time.perf_counter()-score_started
        np.savez_compressed(run / "catalog_scores.npz", **scores)
        exclude = {query.user_id: {row.restaurant_id for row in query.history} for query in queries}
        text_rankings = {name: retriever.rankings(value, exclude, k=experiment.candidate_k) for name, value in scores.items()}
        arms = {"c5": c5}
        for name, user_rankings in text_rankings.items():
            ranks = {query.query_id: user_rankings[query.user_id] for query in queries}
            arms[name] = ranks
            arms[f"c5_{name}_quota"] = {query.query_id: supplement_candidates(c5[query.query_id], ranks[query.query_id],
                                                     k=experiment.candidate_k, text_slots=args.text_slots) for query in queries}
            arms[f"c5_{name}_rrf"] = {query.query_id: rrf([c5[query.query_id], ranks[query.query_id]],
                                                     k=experiment.candidate_k, constant=experiment.rrf_constant) for query in queries}
        validate_rankings(queries, arms, catalog, experiment.candidate_k)
        if any(len(ranks[query.query_id]) != len(c5[query.query_id]) for name, ranks in arms.items()
               if name.startswith("c5_") for query in queries):
            raise ValueError("Fusion candidate budget differs from C5")
        popularity = Counter(row.restaurant_id for row in split.train)
        metrics = {name: evaluate_rankings([RankingObservation(query.query_id, query.user_id, query.relevance_by_item,
                                            tuple(ranks[query.query_id])) for query in queries],
                                            cutoffs=(10, 20, 50, 100), catalog_ids=catalog,
                                            item_popularity=popularity) for name, ranks in arms.items()}
        write_jsonl(run / "candidates_validation.jsonl", ({"query_id": query.query_id, "user_id": query.user_id,
                   "relevance_by_item": query.relevance_by_item, **{name: ranks[query.query_id] for name, ranks in arms.items()}}
                   for query in queries))
        comparisons = [("mean", "late"), ("c5_mean_quota", "c5_late_quota"), ("c5_mean_rrf", "c5_late_rrf")]
        comparisons += [("c5", name) for name in arms if name != "c5"]
        paired = {f"{after}_minus_{before}": {
                    "recall_at_100": paired_change(queries, arms[before], arms[after], 100, "recall"),
                    "ndcg_at_10": paired_change(queries, arms[before], arms[after], 10, "ndcg")}
                    for before, after in comparisons}
        changes = {}
        for name, ranks in arms.items():
            if name == "c5":
                continue
            changes[name], per_query = candidate_changes(queries, c5, ranks)
            write_jsonl(run / f"{name}.candidate_changes.jsonl", per_query)
        state.update(status="complete", metrics=metrics, paired_comparisons=paired, candidate_changes_vs_c5=changes,
                     review_count_bias=count_bias(bank, scores, text_rankings), seconds=time.perf_counter()-started,
                     ranker_hashes_after=protect_ranker(args.baseline_run))
        if state["ranker_hashes_after"] != protected:
            raise ValueError("Ranker files changed during candidate-only comparison")
        state["ranker_unchanged"] = True
        state["limitations"] = [
            "Validation exploration; no new independent holdout and no final ranker metric.",
            "All ratings retained to isolate aggregation; matching topics does not establish preferred sentiment.",
            "Top-k review matching still has restaurant-review-count opportunity bias despite caps.",
            "Frozen E5 block means reused; no learned review attention or encoder fine-tuning.",
            "Standalone text lists remain empty for missing text; fusion falls back to unchanged C5.",
            "Final scores use full historical catalog; no approximate index or production latency claim.",
        ]
        state["artifacts_sha256"] = {name: _file_digest(run / name) for name in (
            "review_bank.npz", "catalog_scores.npz", "profiles_validation.json", "candidates_validation.jsonl")}
        write_json(run / "metrics.json", state)
        write_json(run / "progress.json", state)
        log(json.dumps({"run": str(run), "metrics": {name: {key: values[key] for key in ("recall_at_100", "ndcg_at_10")}
                                                     for name, values in metrics.items()},
                        "seconds": state["seconds"], "ranker_unchanged": True}, ensure_ascii=False))
        return run
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), seconds=time.perf_counter()-started)
        write_json(run / "progress.json", state)
        raise


def console_main():
    main()


if __name__ == "__main__":
    console_main()
