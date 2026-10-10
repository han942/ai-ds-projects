"""Validation-only RLMRec-Con / frozen E5 adaptation on the existing baseline.

python -m rating_recsys.experiments.rlmrec_cli
No E5 inference, LLM calls, or test evaluation is performed. Historical ranker
training candidates are regenerated from strictly earlier graph/profile data.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import evaluate_rankings
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import (
    STAGE1, CheckpointedLightGCN, _generator, _ranker, _ranker_grid,
    build_context, checkpoint_start, observations, paired_bootstrap,
    positive_eval_set, rank_window, ranked_ids, select_ranker,
    window_candidates, window_training_arrays,
)
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import (
    code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path,
)
from rating_recsys.retrieval.rlmrec import RLMRecConfig, RLMRecLightGCN
from rating_recsys.retrieval.hybrid import rrf

VERSION = "rlmrec-con-e5-review-mean-v1"


def write_json(path, value):
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staging.replace(path)


def log(value):
    print(value, file=sys.stderr, flush=True)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FrozenReviewProfiles:
    """Block mean -> review mean -> normalized entity mean, using past rows only.

    Users use query-role vectors; restaurants use passage-role vectors. All
    ratings are retained. The caps match the preceding Transformer experiment.
    Padding never contributes to a mean; no validation-only user bank is used.
    """

    def __init__(self, store, *, max_user_reviews=20, max_item_reviews=30):
        if min(max_user_reviews, max_item_reviews) < 1:
            raise ValueError("Review caps must be positive")
        self.review_index = dict(store.review_index)
        self.max_user_reviews, self.max_item_reviews = max_user_reviews, max_item_reviews
        self.source_identity = {key: store.manifest[key] for key in
                                ("input_manifest_sha256", "cache_identity", "block_config")}
        self.review_vectors = {}
        vectors = store.vectors.detach().cpu().numpy()
        for side, indices in (("user", store.query_blocks), ("item", store.document_blocks)):
            indices = indices.detach().cpu().numpy()
            counts = np.count_nonzero(indices, axis=1)
            # Padding may contain any value in a foreign cache; explicitly mask.
            pooled = np.zeros((len(indices), vectors.shape[1]), dtype=np.float32)
            for column in indices.T:
                present = column != 0
                pooled[present] += vectors[column[present]]
            pooled /= np.maximum(counts[:, None], 1)
            self.review_vectors[side] = pooled
        self.identity = {
            "version": VERSION, "source": self.source_identity,
            "user_role": "query", "item_role": "passage",
            "aggregation": "nonpadding block mean -> equal review mean -> entity L2 normalization",
            "review_vector_normalized": False, "ratings": "all",
            "max_user_reviews": max_user_reviews, "max_item_reviews": max_item_reviews,
        }

    def build(self, rows, cutoff, *, shuffle_seed=None):
        history = tuple(sorted(rows, key=lambda r: (r.event_date, r.review_id)))
        if any(row.event_date > cutoff for row in history):
            raise ValueError("Profile input includes a future review")
        selected = {"user": defaultdict(lambda: deque(maxlen=self.max_user_reviews)),
                    "item": defaultdict(lambda: deque(maxlen=self.max_item_reviews))}
        for row in history:
            index = self.review_index.get(row.review_id)
            if index is None:
                continue
            for side, key in (("user", row.user_id), ("item", row.restaurant_id)):
                if np.any(self.review_vectors[side][index]):
                    selected[side][key].append((row.review_id, index))
        banks, audits = {}, {}
        for side in ("user", "item"):
            bank, references = {}, {}
            for key in sorted(selected[side]):
                records = list(selected[side][key])
                mean = self.review_vectors[side][[index for _, index in records]].mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    bank[key] = np.asarray(mean / norm, dtype=np.float32)
                    references[key] = [rid for rid, _ in records]
            original = profile_hash(bank)
            permutation = list(range(len(bank)))
            if shuffle_seed is not None:
                rng = np.random.default_rng(np.random.SeedSequence([shuffle_seed, int(side == "item")]))
                permutation = rng.permutation(len(bank)).tolist()
                keys = sorted(bank)
                values = [bank[key] for key in keys]
                bank = {key: values[index] for key, index in zip(keys, permutation, strict=True)}
            banks[side] = bank
            audits[side] = {
                "profiles": len(bank), "reviews_selected": sum(map(len, references.values())),
                "review_ids_by_entity": references, "unshuffled_vectors_sha256": original,
                "vectors_sha256": profile_hash(bank), "shuffle_permutation": permutation if shuffle_seed is not None else None,
            }
        return banks["user"], banks["item"], {
            "identity": self.identity, "cutoff_inclusive": cutoff.isoformat(),
            "rows": len(history), "maximum_event_date": max((r.event_date for r in history), default=None).isoformat() if history else None,
            "row_review_ids_sha256": digest_json([r.review_id for r in history]),
            "shuffle_seed": shuffle_seed, **audits,
        }


def profile_hash(bank):
    digest = hashlib.sha256()
    for key in sorted(bank):
        digest.update(str(key).encode() + b"\0")
        digest.update(np.ascontiguousarray(bank[key], dtype="<f4").tobytes())
    return digest.hexdigest()


class TemporalRLMGraphs(CheckpointedLightGCN):
    def __init__(self, config, *, months, profiles, alignment, directory, log=log):
        super().__init__(config, months=months, log=log)
        self.profiles, self.alignment, self.directory = profiles, alignment, directory
        directory.mkdir(parents=True, exist_ok=True)

    def model_for(self, day, ordered_reference, reference_dates):
        block = checkpoint_start(day, self.months)
        key = block.isoformat()
        count = bisect.bisect_left(reference_dates, block)
        if key in self.summary and self.summary[key]["input_rows"] != count:
            raise RuntimeError("Historical graph input changed")
        if block == self._block:
            return self._model
        history = ordered_reference[:count]
        users, items, audit = self.profiles.build(history, block - timedelta(days=1))
        write_json(self.directory / f"{key}.profiles.json", audit)
        started = time.perf_counter()
        model = RLMRecLightGCN(self.config, self.alignment, users, items)
        # The same tiny-graph guard as fit_lightgcn; other errors must surface.
        edges = {(r.user_id, r.restaurant_id) for r in history}
        item_count = len({i for _, i in edges})
        per_user = defaultdict(set)
        for user, item in edges:
            per_user[user].add(item)
        if len(edges) < 2 or item_count < 2:
            model, info = None, {"skipped": "fewer than two graph edges/restaurants", "edges": len(edges)}
        elif all(len(seen) == item_count for seen in per_user.values()):
            model, info = None, {"skipped": "No user has an unvisited restaurant to sample", "edges": len(edges)}
        else:
            model.fit(history)
            model.final_embeddings
            model.save(self.directory / f"{key}.npz")
            info = {"edges": model.edge_count, "users": len(model.user_ids),
                    "restaurants": len(model.item_ids), "profile_coverage": model.profile_coverage,
                    "final_bpr_loss": model.history[-1].bpr_loss,
                    "alignment": asdict(model.alignment_history[-1])}
        self._block, self._model = block, model
        self.summary.setdefault(key, {"checkpoint": key, "queries": 0, "input_rows": count,
                                     "profiles_sha256": sha_file(self.directory / f"{key}.profiles.json"),
                                     "fit_seconds": time.perf_counter() - started, **info})
        write_json(self.directory / "summary.json", self.rows())
        self.log(f"RLMRec historical graph {key}: {len(edges)} edges; {time.perf_counter()-started:.1f}s")
        return model


def save_ranks(path, queries, arms):
    with path.open("w") as output:
        for query in queries:
            output.write(json.dumps({"query_id": query.query_id, "user_id": query.user_id,
                                     "relevance_by_item": query.relevance_by_item,
                                     **{name: rankings[query.query_id] for name, rankings in arms.items()}}, ensure_ascii=False) + "\n")


def validate_ranks(queries, arms, catalog, k):
    for rankings in arms.values():
        if set(rankings) != {q.query_id for q in queries}:
            raise ValueError("Ranking query IDs differ")
        for q in queries:
            ids = rankings[q.query_id]
            if (len(ids) > k or len(ids) != len(set(ids)) or not set(ids) <= catalog
                    or set(ids) & {r.restaurant_id for r in q.history}):
                raise ValueError("Invalid catalog/history/ranking")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=PROJECT_ROOT / "artifacts/snapshots/e7896add5b4b5939.jsonl")
    p.add_argument("--baseline-run", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add")
    p.add_argument("--prepared-validation", type=Path, default=PROJECT_ROOT / "artifacts/prepared/e7896add5b4b5939/validation-1d1692022f92bc38/manifest.json")
    p.add_argument("--frozen-checkpoint", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/review_transformer/20261009T023235302075Z-e7896add/review_transformer_validation.pt")
    p.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/rlmrec")
    p.add_argument("--weights", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--alignment-batch-size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    # Near-tied float32 scores can change order across BLAS reduction schemes.
    # Use the same deterministic reduction for controls, treatments and reload.
    with threadpool_limits(limits=1, user_api="blas"):
        return run(args, argv)


def run(args, argv=None):
    if not args.weights or any(w <= 0 or not np.isfinite(w) for w in args.weights) or len(set(args.weights)) != len(args.weights):
        raise ValueError("Provide distinct positive alignment weights")
    started = time.perf_counter()
    prior = json.loads((args.baseline_run / "manifest.json").read_text())
    exp = ExperimentConfig(**prior["config"])
    if exp.ranker_training_mode != "window":
        raise ValueError("This experiment requires the window baseline")
    rows = load_snapshot(args.snapshot)
    if dataset_digest(rows) != prior["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Baseline snapshot differs")
    texts, text_meta = load_review_texts(review_texts_path(args.snapshot), rows)
    split = build_global_temporal_split(rows, train_fraction=exp.train_fraction, validation_fraction=exp.validation_fraction)
    queries, query_summary = build_window_queries(split.train, split.validation, config=exp,
                                                 phase="validation", cutoff=split.train_cutoff)
    prepared = json.loads(args.prepared_validation.read_text())
    if prepared["identity"]["snapshot_id"] != dataset_digest(rows):
        raise ValueError("Prepared snapshot differs")
    identity = prepared["identity"]
    expected = {"lightgcn": exp.lightgcn_config.to_dict(), "candidate_k": exp.candidate_k,
                "rrf_constant": exp.rrf_constant, "features": list(exp.feature_names),
                "cutoffs": {"train_through": split.train_cutoff.isoformat(),
                            "validation_through": split.validation_cutoff.isoformat()}}
    if any(identity[key] != value for key, value in expected.items()):
        raise ValueError("Prepared baseline protocol/config differs")
    old = {stage: {qid: tuple(ids) for qid, ids in ranks.items()} for stage, ranks in prepared["data"]["ordered"].items()}
    old_r1 = {r["query_id"]: tuple(r["baseline"]) for r in map(json.loads, (args.baseline_run / "recommendations_validation.jsonl").read_text().splitlines())}
    catalog = {r.restaurant_id for r in split.train}
    baseline_arms = {"c1": old["c1_item_item"], "c4": old["c4_lightgcn"], "c5": old[STAGE1], "r1": old_r1}
    validate_ranks(queries, baseline_arms, catalog, exp.candidate_k)
    run = args.artifacts_dir / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + dataset_digest(rows)[:8])
    run.mkdir(parents=True)
    state = {"status": "preparing", "version": VERSION, "phase": "validation-only", "test_evaluated": False,
             "evidence_status": "exploratory validation used for weight and ranker selection; no independent holdout",
             "experiment": exp.to_dict(), "snapshot_id": dataset_digest(rows), "review_texts": text_meta,
             "cutoff": split.train_cutoff.isoformat(), "query_summary": query_summary,
             "weights": args.weights, "temperature": args.temperature, "alignment_batch_size": args.alignment_batch_size,
             "threads": args.threads, "seed": exp.random_seed, "api_requests": 0, "new_embedding_inference": 0,
             "blas_threads": 1, "threadpools": threadpool_info(),
             "selection": "nonzero weight with maximum C5 NDCG@10 at fixed epoch 20; ties Recall@100 then weight order",
             "command": [sys.executable, "-m", "rating_recsys.experiments.rlmrec_cli", *(argv if argv is not None else sys.argv[1:])],
             "sources": {"baseline_run": str(args.baseline_run), "prepared_validation": str(args.prepared_validation),
                         "prepared_sha256": sha_file(args.prepared_validation), "baseline_rankings_sha256": sha_file(args.baseline_run / "recommendations_validation.jsonl"),
                         "frozen_checkpoint": str(args.frozen_checkpoint), "frozen_checkpoint_sha256": sha_file(args.frozen_checkpoint)},
             "variants": []}
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    (run / "source.diff").write_text(source.pop("git_diff"))
    state["code"] = source
    for name in ("experiments/rlmrec_cli.py", "retrieval/rlmrec.py"):
        destination = run / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / "src/rating_recsys" / name, destination)
    write_json(run / "progress.json", state)
    log(f"[RLMRec-Con E5 adaptation] {run}")
    try:
        saved = torch.load(args.frozen_checkpoint, map_location="cpu", weights_only=False)
        store = saved["store"]
        if not set(store.review_index) <= {r.review_id for r in split.train}:
            raise ValueError("Frozen store contains non-training reviews")
        for record in store.manifest["records"]:
            text = texts[record["review_id"]]
            if any(text[b["start"]:b["end"]] != b["text"] for b in record["blocks"]):
                raise ValueError("Frozen text spans differ from the current review snapshot")
        profiles = FrozenReviewProfiles(store)
        users, items, profile_audit = profiles.build(split.train, split.train_cutoff)
        write_json(run / "profiles_validation.json", profile_audit)
        state["profiles"] = {"identity": profiles.identity, "users": len(users), "items": len(items),
                             "audit_sha256": sha_file(run / "profiles_validation.json")}
        del saved, store, texts
        gc.collect()
        evaluate = lambda ranks: evaluate_rankings(observations(queries, ranks), cutoffs=(10, 100), catalog_ids=catalog)
        state["baseline_validation"] = {name: evaluate(ranks) for name, ranks in baseline_arms.items()}
        exclude = {q.user_id: [r.restaurant_id for r in q.history] for q in queries}
        user_ids = [q.user_id for q in queries]
        best_record, best_fused = None, None

        def fused_ranks(graph_ranks):
            return {q.query_id: rrf((old["c1_item_item"][q.query_id], graph_ranks[q.query_id]),
                                    k=exp.candidate_k, constant=exp.rrf_constant) for q in queries}

        def fit_variant(weight, name, user_bank, item_bank, *, shuffled=False):
            record = {"name": name, "weight": weight, "shuffled": shuffled, "epochs": []}
            state["variants"].append(record)
            state["status"] = f"training-{name}"
            write_json(run / "progress.json", state)
            cfg = RLMRecConfig(weight, args.temperature, args.alignment_batch_size, exp.random_seed, args.threads)
            model = RLMRecLightGCN(exp.lightgcn_config, cfg, user_bank, item_bank)
            def callback(stats, current):
                graph = current.recommend(user_ids, exclude, exp.candidate_k)
                graph = {q.query_id: graph[q.user_id] for q in queries}
                fused = fused_ranks(graph)
                metrics, fusion_metrics = evaluate(graph), evaluate(fused)
                record["epochs"].append({**asdict(stats), "alignment": asdict(current.alignment_history[-1]),
                                         "graph": metrics, "fusion": fusion_metrics})
                write_json(run / "progress.json", state)
                log(f"{name} epoch {stats.epoch}/{exp.lightgcn_epochs}: BPR={stats.bpr_loss:.4f}; "
                    f"C5 Recall@100={fusion_metrics['recall_at_100']:.4%}; NDCG@10={fusion_metrics['ndcg_at_10']:.6f}")
            variant_started = time.perf_counter()
            model.fit(split.train, callback=callback)
            graph = model.recommend(user_ids, exclude, exp.candidate_k)
            graph = {q.query_id: graph[q.user_id] for q in queries}
            fused = fused_ranks(graph)
            validate_ranks(queries, {"graph": graph, "fusion": fused}, catalog, exp.candidate_k)
            record.update(graph=evaluate(graph), fusion=evaluate(fused), seconds=time.perf_counter()-variant_started,
                          profile_coverage=model.profile_coverage, projector_parameters=model.projector_parameter_count)
            model.save(run / f"{name}.npz")
            save_ranks(run / f"{name}.rankings.jsonl", queries, {"graph": graph, "fusion": fused})
            write_json(run / "progress.json", state)
            return record, graph, fused

        control, graph, fused = fit_variant(0, "lambda_0", users, items)
        state["baseline_reproduction"] = {"c4_exact": graph == baseline_arms["c4"], "c5_exact": fused == baseline_arms["c5"]}
        if not all(state["baseline_reproduction"].values()):
            raise ValueError("Weight-zero control does not reproduce the baseline rankings exactly")
        for weight in args.weights:
            record, _, fused = fit_variant(weight, f"lambda_{weight:g}", users, items)
            key = lambda r: (r["fusion"]["ndcg_at_10"], r["fusion"]["recall_at_100"])
            if best_record is None or key(record) > key(best_record):
                best_record, best_fused = record, fused
        state["selected_weight"] = best_record["weight"]
        shuffled_users, shuffled_items, shuffle_audit = profiles.build(split.train, split.train_cutoff, shuffle_seed=1701)
        write_json(run / "profiles_shuffled.json", shuffle_audit)
        shuffle_record, _, shuffle_fused = fit_variant(best_record["weight"], "shuffled_selected_weight", shuffled_users, shuffled_items, shuffled=True)
        state["candidate_bootstrap"] = {
            "aligned_vs_baseline_at10": paired_bootstrap(queries, best_fused, baseline_arms["c5"], cutoff=10, samples=exp.bootstrap_samples, seed=exp.random_seed),
            "aligned_vs_baseline_at100": paired_bootstrap(queries, best_fused, baseline_arms["c5"], cutoff=100, samples=exp.bootstrap_samples, seed=exp.random_seed),
            "aligned_vs_shuffled_at10": paired_bootstrap(queries, best_fused, shuffle_fused, cutoff=10, samples=exp.bootstrap_samples, seed=exp.random_seed),
        }
        state["status"] = "generating-historical-ranker-candidates"
        write_json(run / "progress.json", state)
        alignment = RLMRecConfig(best_record["weight"], args.temperature, args.alignment_batch_size, exp.random_seed, args.threads)
        graphs = TemporalRLMGraphs(exp.lightgcn_config, months=exp.lightgcn_checkpoint_months,
                                  profiles=profiles, alignment=alignment, directory=run / "historical_graphs")
        train = window_training_arrays(split.train, _generator(exp), graphs, config=exp,
                                       through=split.train_cutoff, log=log)
        state["training"] = train.summary
        state["historical_graphs"] = graphs.rows()
        write_json(run / "historical_graphs/summary.json", graphs.rows())
        np.savez_compressed(run / "ranker_training.npz", features=train.features, labels=train.labels, groups=train.groups)
        model = RLMRecLightGCN.load(run / f"lambda_{best_record['weight']:g}.npz")
        validation = window_candidates(queries, build_context(split.train), _generator(exp), graph=model,
                                       feature_names=exp.feature_names, rating_shrinkage_strength=exp.rating_shrinkage_strength)
        if validation.ordered[STAGE1] != best_fused:
            raise ValueError("Full pipeline fusion differs from the selection candidates")
        np.savez_compressed(run / "ranker_validation.npz", features=validation.features, labels=validation.labels,
                            groups=validation.group_sizes, restaurant_ids=validation.row_restaurant_ids)
        eval_set = positive_eval_set(validation)
        state["status"], state["ranker_grid"] = "training-ranker", []
        best_ranker, best_ranks, best_params, best_key = None, None, None, None
        for params in _ranker_grid(exp):
            ranker = _ranker(exp, params)
            rank_started = time.perf_counter()
            ranker.fit(train.features, train.labels, train.groups,
                       eval_set=eval_set if params["early_stopping"] else None,
                       early_stopping_rounds=exp.early_stopping_rounds if params["early_stopping"] else None)
            rankings = ranked_ids(rank_window(ranker, queries, validation))
            metrics = evaluate(rankings)
            record = {**params, **metrics, "best_iteration": ranker.best_iteration or params["n_estimators"],
                      "fit_seconds": time.perf_counter()-rank_started, "validation_curve": ranker.validation_curve}
            state["ranker_grid"].append(record)
            key = (metrics["ndcg_at_10"], metrics["recall_at_10"])
            if best_key is None or key > best_key:
                best_ranker, best_ranks, best_params, best_key = ranker, rankings, record, key
            write_json(run / "progress.json", state)
            log(f"LambdaRank {params['name']}: NDCG@10={metrics['ndcg_at_10']:.6f}; best_iteration={ranker.best_iteration}")
        if select_ranker(state["ranker_grid"], exp)["name"] != best_params["name"]:
            raise ValueError("Ranker selection policy differs")
        best_ranker.save(run / "rlmrec_lambdarank_validation.txt")
        import lightgbm as lgb

        reloaded = lgb.Booster(model_file=str(run / "rlmrec_lambdarank_validation.txt"))
        if ranked_ids(rank_window(reloaded, queries, validation)) != best_ranks:
            raise ValueError("Saved ranker does not reproduce the selected rankings")
        state["ranker_reload_exact"] = True
        validate_ranks(queries, {"rlmrec_r1": best_ranks}, catalog, exp.candidate_k)
        state["chosen_ranker"] = best_params
        state["rlmrec_r1_validation"] = evaluate(best_ranks)
        state["full_system_bootstrap"] = paired_bootstrap(queries, best_ranks, old_r1, cutoff=10,
                                                         samples=exp.bootstrap_samples, seed=exp.random_seed)
        save_ranks(run / "recommendations_validation.jsonl", queries,
                   {**baseline_arms, "rlmrec_c5": best_fused, "rlmrec_r1": best_ranks, "shuffled_c5": shuffle_fused})
        state.update(status="complete", seconds=time.perf_counter()-started)
        write_json(run / "progress.json", state)
        write_json(run / "metrics.json", state)
        print(json.dumps({"run": str(run), "selected_weight": best_record["weight"],
                          "baseline": state["baseline_validation"]["r1"], "rlmrec": state["rlmrec_r1_validation"]}, ensure_ascii=False))
        return state
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", seconds=time.perf_counter()-started)
        write_json(run / "progress.json", state)
        raise


def console_main():
    main()


if __name__ == "__main__":
    main()
