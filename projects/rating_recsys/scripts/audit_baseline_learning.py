"""Inspect a saved baseline and fit one fixed validation-only learning curve.

This diagnostic preserves the historical artifacts and does not evaluate test,
refit at T2, search a parameter grid, or adopt a model. Historical prepared rows
are loaded explicitly after checksum validation; this is not a current-cache hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import WindowCandidates, ranker_eval_kwargs
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import canonical_json, load_snapshot
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
from rating_recsys.retrieval.lightgcn import LightGCN


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepared(root, entry):
    folder = root / entry["path"]
    meta = read_json(folder / "manifest.json")
    checksum = meta.pop("metadata_sha256")
    if hashlib.sha256(canonical_json(meta).encode()).hexdigest() != checksum:
        raise ValueError("Prepared metadata checksum differs")
    if sha256(folder / "data.npz") != meta["arrays_sha256"]:
        raise ValueError("Prepared arrays checksum differs")
    with np.load(folder / "data.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return arrays, meta


def candidate_ndcg(predictions, labels, groups):
    """Conditional candidate-normalized metric, explicitly not system NDCG."""
    offsets = np.r_[0, np.cumsum(groups)]
    width = int(max(groups))
    score = np.full((len(groups), width), -np.inf)
    gain = np.zeros_like(score)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        score[index, :stop - start] = predictions[start:stop]
        gain[index, :stop - start] = np.exp2(labels[start:stop]) - 1
    discount = 1 / np.log2(np.arange(min(10, width)) + 2)
    ranked = np.argsort(-score, axis=1, kind="stable")[:, :len(discount)]
    ideal = (np.sort(gain, axis=1)[:, ::-1][:, :len(discount)] * discount).sum(axis=1)
    dcg = (np.take_along_axis(gain, ranked, axis=1) * discount).sum(axis=1)
    usable = ideal > 0
    return float((dcg[usable] / ideal[usable]).mean())


def graph_learning_curve(split, queries, config, threads):
    """Replay the fixed 20-epoch C4; score the full cutoff catalog at checkpoints."""
    from threadpoolctl import threadpool_limits

    if config.lightgcn_epochs > 20:
        raise ValueError("Graph diagnostic is limited to the historical 20 epochs")
    curve = []

    def checkpoint(stats, model):
        if stats.epoch not in (1, 5, 10, 15, 20):
            return False
        final = model.final_embeddings
        users = np.asarray([model.user_ids[q.user_id] for q in queries])
        item_vectors = final[len(model.user_ids):]
        scores = final[users] @ item_vectors.T
        for index, query in enumerate(queries):
            visited = [model.item_ids[row.restaurant_id] for row in query.history]
            scores[index, visited] = -np.inf
        # item_keys is ascending: stable score ties match the retriever's ID order.
        positions = np.argsort(-scores, axis=1, kind="stable")[:, :100]
        recalls, ndcgs = [], []
        for index, query in enumerate(queries):
            truth = {key: grade for key, grade in query.relevance_by_item.items() if grade > 0}
            if not truth:
                continue
            selected = model.item_keys[positions[index]]
            selected = selected[np.isfinite(scores[index, positions[index]])]
            recalls.append(len(set(selected) & set(truth)) / len(truth))
            ideal = sum((2 ** grade - 1) / np.log2(rank + 2)
                        for rank, grade in enumerate(sorted(truth.values(), reverse=True)[:10]))
            dcg = sum((2 ** truth.get(int(key), 0) - 1) / np.log2(rank + 2)
                      for rank, key in enumerate(selected[:10]))
            ndcgs.append(dcg / ideal)
        curve.append({"epoch": stats.epoch, "bpr_loss": stats.bpr_loss,
                      "validation_recall_at_100": float(np.mean(recalls)),
                      "validation_full_truth_ndcg_at_10": float(np.mean(ndcgs)),
                      "queries": len(recalls)})
        print(json.dumps(curve[-1]), flush=True)
        return False

    with threadpool_limits(limits=threads):
        model = LightGCN(config.lightgcn_config).fit(split.train, callback=checkpoint)
    return {"scope": "fixed C4 architecture, seed and 20 epochs; validation only",
            "config": config.lightgcn_config.to_dict(), "curve": curve,
            "training_loss": [{"epoch": entry.epoch, "bpr_loss": entry.bpr_loss}
                              for entry in model.history],
            "test_prediction_calls": 0,
            "limitations": ["One seed, reused validation, five checkpoints; no independent final evidence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trees", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_trees <= 300:
        raise ValueError("Diagnostic budget must be 1..300 trees")
    started = time.perf_counter()
    manifest = read_json(args.run / "manifest.json")
    metrics = read_json(args.run / "metrics.json")
    if sha256(args.snapshot) != manifest["snapshot"]["artifact_sha256"]:
        raise ValueError("Snapshot differs from historical run")
    config = ExperimentConfig(**manifest["config"])
    rows = load_snapshot(args.snapshot)
    split = build_global_temporal_split(rows, train_fraction=config.train_fraction,
                                        validation_fraction=config.validation_fraction)
    if split.summary() != manifest["split"]:
        raise ValueError("Recomputed split differs")
    queries, _ = build_window_queries(split.train, split.validation, config=config,
                                     phase="validation", cutoff=split.train_cutoff)
    if args.graph_only:
        result = graph_learning_curve(split, queries, config, args.threads)
        result.update(snapshot_sha256=sha256(args.snapshot), source_sha256=sha256(Path(__file__)),
                      elapsed_seconds=time.perf_counter() - started)
        expected = metrics["validation"]["c4_lightgcn"]["recall_at_100"]
        result["historical_recall_at_100"] = expected
        result["terminal_recall_matches_historical"] = bool(
            np.isclose(result["curve"][-1]["validation_recall_at_100"], expected, atol=1e-12, rtol=0))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return
    entries = {entry["kind"]: entry for entry in manifest["prepared_data"]["entries"]}
    root = args.run.parents[1]
    train, train_meta = prepared(root, entries["train"])
    val, val_meta = prepared(root, entries["validation"])
    data = dict(val_meta["data"])
    data.pop("graph")
    data["catalog"] = set(data["catalog"])
    for name in ("item_popularity", "item_regions", "item_names"):
        data[name] = {int(key): value for key, value in data[name].items()}
    candidates = WindowCandidates(**val, **data)
    kwargs = ranker_eval_kwargs(candidates, queries, cutoff=10)
    xval, yval, gval = kwargs["eval_set"]
    if sum(train["groups"]) != len(train["labels"]):
        raise ValueError("Training group dimensions differ")
    if not np.isfinite(train["features"]).all() or not np.isfinite(xval).all():
        raise ValueError("Non-finite features")
    # Fixed historical architecture. The curve is diagnostic, not a new search.
    params = metrics["selection"]["final_ranker_params"]
    ranker = LightGBMLambdaRanker(
        random_seed=config.random_seed, ranking_k=10, feature_names=config.feature_names,
        n_estimators=args.max_trees, learning_rate=config.learning_rate,
        num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
        reg_lambda=config.reg_lambda, n_jobs=args.threads,
    )
    ranker.fit(train["features"], train["labels"], train["groups"].tolist(), **kwargs)
    sampled = sorted({1, 3, 4, 10, 20, 40, 76, 100, 150, 200, args.max_trees})
    sampled = [tree for tree in sampled if tree <= args.max_trees]
    curve = []
    for trees in sampled:
        train_score = ranker._model.predict(train["features"], num_iteration=trees)
        val_score = ranker._model.predict(xval, num_iteration=trees)
        curve.append({
            "trees": trees,
            "train_retrieved_groups_candidate_ndcg_at_10": candidate_ndcg(
                train_score, train["labels"], train["groups"]),
            "validation_retrieved_groups_candidate_ndcg_at_10": candidate_ndcg(
                val_score, yval, gval),
            "validation_full_truth_ndcg_at_10": ranker.validation_curve[trees - 1],
        })
        print(json.dumps(curve[-1]), flush=True)
    result = {
        "scope": "fixed historical architecture, T1 training, validation only; exploratory diagnostic",
        "run": str(args.run), "snapshot_sha256": sha256(args.snapshot),
        "source_sha256": sha256(Path(__file__)),
        "prepared_identity": {"train": train_meta["identity"], "validation": val_meta["identity"]},
        "params": {"num_leaves": params["num_leaves"], "min_child_samples": params["min_child_samples"],
                   "learning_rate": config.learning_rate, "seed": config.random_seed,
                   "max_trees": args.max_trees, "threads": args.threads},
        "train_groups": len(train["groups"]), "train_rows": len(train["labels"]),
        "train_label_counts": {str(label): int((train["labels"] == label).sum()) for label in (0, 1, 2)},
        "validation_queries": kwargs["eval_query_count"],
        "validation_candidate_rows": len(yval),
        "historical_validation_c5_ndcg_at_10": metrics["validation"]["r0_candidate_order"]["ndcg_at_10"],
        "historical_selected_validation_ndcg_at_10": metrics["validation"]["r1_lambdarank"]["ndcg_at_10"],
        "curve": curve, "validation_full_truth_curve": ranker.validation_curve,
        "test_prediction_calls": 0, "test_model_selection": False,
        "limitations": ["One architecture and seed, reused validation; not independent final evidence",
                        "Train metric includes only retrieved-positive groups and candidate IDCG",
                        "Historical prepared arrays were audited, not rebuilt under current cache identity",
                        "Diagnostic model is not the T2 refitted historical final model"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
