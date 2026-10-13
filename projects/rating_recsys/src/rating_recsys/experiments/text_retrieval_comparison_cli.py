"""Validation-only text item graph versus learned review-level late retrieval.

Uses existing local E5 blocks, never a ranker or future review input. Fixed
settings and all seeds/controls are reported; there is no epoch/grid selection.
The two families differ in training scope, so their head-to-head result is a
system comparison. Within-family controls measure narrower effects.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
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
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.prepared import _file_digest
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.review_multivector_cli import (
    find_prepared, load_prepared, protect_ranker, validate_rankings,
)
from rating_recsys.experiments.snapshot import (
    canonical_json, code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path,
)
from rating_recsys.retrieval.lightgcn import LightGCN, _sample_negatives
from rating_recsys.retrieval.review_multivector import (
    ReviewMultiVectorConfig, ReviewVectorBank, array_hash, build_review_bank,
    review_vectors_from_store, unit,
)

VERSION = "text-retrieval-families-v1"


def ranks_from_scores(scores, user_ids, item_ids, queries, k=100):
    scores = np.asarray(scores)
    if scores.shape != (len(user_ids), len(item_ids)) or not np.isfinite(scores).all():
        raise ValueError("Invalid full-catalog score matrix")
    if len(set(user_ids)) != len(user_ids) or np.any(np.diff(item_ids) <= 0):
        raise ValueError("Nonunique users or unsorted catalog")
    users = {int(uid): n for n, uid in enumerate(user_ids)}
    items = {int(iid): n for n, iid in enumerate(item_ids)}
    output = {}
    for q in queries:
        row = scores[users[q.user_id]].copy()
        seen = [items[r.restaurant_id] for r in q.history if r.restaurant_id in items]
        row[seen] = -np.inf
        order = np.argsort(-row, kind="stable")[:k]
        output[q.query_id] = tuple(int(item_ids[n]) for n in order if np.isfinite(row[n]))
    return output


def per_user_values(queries, ranks, cutoff, metric):
    return np.asarray([
        query_scores(RankingObservation(q.query_id, q.user_id, q.relevance_by_item,
                      tuple(ranks[q.query_id])), cutoff)[metric]
        for q in queries if q.relevant
    ], dtype=np.float64)


def mean_seed_comparison(queries, before, after, *, cutoff=100, metric="recall", samples=2000, seed=42):
    """Average deltas across seeds for each user, then bootstrap users once."""
    if len(before) != len(after) or not before:
        raise ValueError("Paired nonempty seed lists required")
    differences = np.stack([per_user_values(queries, b, cutoff, metric) -
                            per_user_values(queries, a, cutoff, metric)
                            for a, b in zip(before, after)])
    average = differences.mean(0)
    if not len(average):
        raise ValueError("No relevant queries")
    rng = np.random.default_rng(seed)
    boot = np.asarray([average[rng.integers(len(average), size=len(average))].mean()
                       for _ in range(samples)])
    return {"mean_difference": float(average.mean()),
            "bootstrap_95pct": np.quantile(boot, [.025, .975]).tolist(),
            "per_seed_differences": differences.mean(1).tolist(),
            "paired_users": len(average), "seeds": len(before), "samples": samples,
            "bootstrap_seed": seed, "scope": "users; conditional on fixed training seeds"}


def sampling_audit(rows, config, *, review_epochs=None):
    """Audit the shared RNG stream through both models' epoch budgets."""
    edges = sorted({(r.user_id, r.restaurant_id) for r in rows})
    users = {key: n for n, key in enumerate(sorted({u for u, _ in edges}))}
    items = {key: n for n, key in enumerate(sorted({i for _, i in edges}))}
    eu = np.asarray([users[u] for u, _ in edges], dtype=np.int64)
    ei = np.asarray([items[i] for _, i in edges], dtype=np.int64)
    positive = np.unique(eu * len(items) + ei)
    use = np.bincount(eu, minlength=len(users))[eu] < len(items)
    eu, ei = eu[use], ei[use]
    rng = np.random.default_rng(config.seed)
    initial = rng.normal(0, config.init_std, size=(len(users)+len(items), config.dimension)).astype(np.float32)
    audits, graph_audits = [], []
    epochs = max(config.epochs, review_epochs if review_epochs is not None else config.epochs)
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(eu))
        negatives = _sample_negatives(rng, eu, len(items), positive)
        audits.append({"epoch": epoch, "order_sha256": array_hash(order),
                       "negative_items_sha256": array_hash(negatives),
                       "sequence_sha256": hashlib.sha256(order.tobytes()+negatives.tobytes()).hexdigest()})
        if epoch > config.epochs:
            continue
        triples_digest = hashlib.sha256()
        batches = 0
        for start in range(0, len(order), config.batch_size):
            ix = order[start:start+config.batch_size]
            for value in (eu[ix], len(users)+ei[ix], len(users)+negatives[ix]):
                triples_digest.update(array_hash(value).encode())
            batches += 1
        graph_audits.append({"epoch": epoch, "batches": batches, "examples": len(eu),
                             "ordered_bpr_triples_sha256": triples_digest.hexdigest()})
    return {"initial_ego_sha256": array_hash(initial), "edges": len(eu),
            "epochs": audits, "graph_batches": graph_audits}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=PROJECT_ROOT/"artifacts/snapshots/e7896add5b4b5939.jsonl")
    p.add_argument("--baseline-run", type=Path, default=PROJECT_ROOT/"artifacts/runs/20261004T125314601899Z-e7896add")
    p.add_argument("--frozen-checkpoint", type=Path, default=PROJECT_ROOT/"artifacts/comparisons/review_transformer/20261009T155837960566Z-e7896add/review_transformer_validation.pt")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT/"artifacts/comparisons/text_retrieval_families")
    p.add_argument("--resume-dir", type=Path)
    p.add_argument("--reuse-graph-run", type=Path,
                   help="Reuse completed same-settings graph/C4 checkpoints and retrain only review arms")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--graph-epochs", type=int, default=20)
    p.add_argument("--review-epochs", type=int, default=12)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--balance-text-presence", action=argparse.BooleanOptionalAction,
                   default=True, help="Use text in a BPR pair only when both items have usable reviews")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (len(set(args.seeds)) != len(args.seeds) or not args.seeds or
            min(args.graph_epochs, args.review_epochs, args.threads) < 1):
        raise ValueError("Invalid seeds/epochs/threads")
    started = time.perf_counter()
    baseline = read_json(args.baseline_run/"manifest.json")
    cfg = dict(baseline["config"])
    for field in ("num_leaves_grid", "min_child_samples_grid"):
        cfg[field] = tuple(cfg[field])
    exp = ExperimentConfig(**cfg)
    if exp.ranker_training_mode != "window":
        raise ValueError("Window protocol required")
    rows = load_snapshot(args.snapshot)
    digest = dataset_digest(rows)
    if digest != baseline["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Baseline snapshot differs")
    split = build_global_temporal_split(rows, train_fraction=exp.train_fraction,
                                       validation_fraction=exp.validation_fraction)
    queries, excluded = build_window_queries(split.train, split.validation, config=exp,
                                             phase="validation", cutoff=split.train_cutoff)
    prepared_path = find_prepared(PROJECT_ROOT/"artifacts/prepared", digest, split, exp)
    prepared, c5 = load_prepared(prepared_path, digest, split, exp, queries)
    fixed_c4 = {qid: tuple(ids) for qid, ids in prepared["data"]["ordered"]["c4_lightgcn"].items()}
    catalog = sorted({r.restaurant_id for r in split.train})
    if set(prepared["data"]["catalog"]) != set(catalog):
        raise ValueError("Past catalog differs")
    protected = protect_ranker(args.baseline_run)
    sources = {"snapshot": str(args.snapshot.resolve()), "snapshot_sha256": _file_digest(args.snapshot),
               "prepared": str(prepared_path.resolve()), "prepared_sha256": _file_digest(prepared_path),
               "frozen_checkpoint": str(args.frozen_checkpoint.resolve()),
               "frozen_checkpoint_sha256": _file_digest(args.frozen_checkpoint)}
    settings = {"seeds": args.seeds, "graph_epochs": args.graph_epochs,
                "review_epochs": args.review_epochs, "threads": args.threads,
                "max_user_reviews": 8, "max_item_reviews": 12,
                "semantic_neighbors": 10, "semantic_weight": .1, "semantic_edge_weights": "binary",
                "projection_dim": 32, "matching_k": 3, "initial_text_coefficient": .1,
                "balance_text_presence": args.balance_text_presence}
    if args.resume_dir:
        run = args.resume_dir
        state = read_json(run/"progress.json")
        if state["sources"] != sources or state["settings"] != settings:
            raise ValueError("Resume sources/settings differ")
        if state["ranker_hashes_before"] != protected:
            raise ValueError("Protected ranker differs")
    else:
        run = args.output_dir/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+digest[:8])
        run.mkdir(parents=True, exist_ok=False)
        state = {"version": VERSION, "status": "preparing", "sources": sources, "settings": settings,
                 "snapshot_id": digest, "experiment": exp.to_dict(), "train_rows": len(split.train),
                 "cutoff": split.train_cutoff.isoformat(), "validation_through": split.validation_cutoff.isoformat(),
                 "queries": len(queries), "evaluated_queries": sum(bool(q.relevant) for q in queries),
                 "new_users_excluded": excluded, "catalog": len(catalog), "arms": {},
                 "ranker_hashes_before": protected, "ranker_trained": False, "ranker_predicted": False,
                 "test_evaluated": False, "embedding_inference_calls": 0, "api_calls": 0,
                 "selection": "all fixed final-epoch conditions; no setting/epoch selection",
                 "training_objective": "static BPR over all past unique visits; unvisited negatives",
                 "limitations": ["Repeated validation exploration; no independent holdout",
                                  "New users excluded; catalog-unseen truth retained",
                                  "Static visit-positive training differs from future satisfaction labels",
                                  "Fixed E5 graph adaptation; no learned LATTICE structure or image modality",
                                  "Review-level projection; no ColBERT token encoder fine-tuning",
                                  "Graph backbone trained; review branch trains on frozen same-seed C4 backbone",
                                  "Exact full-catalog scan; no approximate-index/production latency claim"]}
        write_json(run/"progress.json", state)
    print(f"[retrieval-only comparison] {run}", flush=True)
    bank_path = run/"review_bank.npz"
    if args.reuse_graph_run:
        previous = read_json(args.reuse_graph_run/"metrics.json")
        expected_settings = {k: v for k, v in settings.items() if k != "balance_text_presence"}
        previous_settings = {k: v for k, v in previous["settings"].items() if k != "balance_text_presence"}
        if previous["status"] != "complete" or previous["sources"] != sources or previous_settings != expected_settings:
            raise ValueError("Reused graph experiment differs in sources/settings or is incomplete")
        if _file_digest(args.reuse_graph_run/"review_bank.npz") != previous["bank_file_sha256"]:
            raise ValueError("Reused review bank was changed")
        if not bank_path.exists():
            shutil.copy2(args.reuse_graph_run/"review_bank.npz", bank_path)
        state.update(reused_graph_run=str(args.reuse_graph_run.resolve()),
                     reused_graph_metrics_sha256=_file_digest(args.reuse_graph_run/"metrics.json"),
                     bank_file_sha256=previous["bank_file_sha256"], bank_audit=previous["bank_audit"],
                     frozen_store=previous["frozen_store"], review_texts=previous["review_texts"],
                     review_training_policy="both semantic scores disabled unless user, positive and negative have text after own-review exclusion")
        write_json(run/"progress.json", state)
    if bank_path.exists():
        if _file_digest(bank_path) != state["bank_file_sha256"]:
            raise ValueError("Bank file checksum mismatch")
        with np.load(bank_path, allow_pickle=False) as saved:
            bank = ReviewVectorBank(**{key: saved[key] for key in saved.files}, audit=state["bank_audit"])
        if any(array_hash(getattr(bank, k)) != v for k, v in bank.audit["arrays_sha256"].items()):
            raise ValueError("Bank array checksum mismatch")
    else:
        import torch
        torch.set_num_threads(args.threads)
        texts, text_meta = load_review_texts(review_texts_path(args.snapshot), rows)
        saved = torch.load(args.frozen_checkpoint, map_location="cpu", weights_only=False)
        store = saved["store"]
        records = store.manifest["records"]
        if set(store.review_index) != {r["review_id"] for r in records} or not set(store.review_index) <= {r.review_id for r in split.train}:
            raise ValueError("Frozen store includes inconsistent/nonpast reviews")
        if hashlib.sha256(__import__("json").dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != store.manifest["input_manifest_sha256"]:
            raise ValueError("Frozen manifest hash mismatch")
        for record in records:
            text = texts[record["review_id"]]
            if not text or any(text[b["start"]:b["end"]] != b["text"] for b in record["blocks"]):
                raise ValueError("Frozen blocks differ from sidecar")
        qv, iv = review_vectors_from_store(store)
        bank = build_review_bank(split.train, split.train_cutoff, store.review_index, qv, iv,
                                 query_users=sorted({r.user_id for r in split.train}),
                                 config=ReviewMultiVectorConfig(max_user_reviews=8, max_item_reviews=12))
        state.update(bank_audit=bank.audit, review_texts=text_meta,
                     frozen_store={k: store.manifest[k] for k in ("input_manifest_sha256", "cache_identity", "block_config", "reviews")})
        del saved, store, texts, records, qv, iv
        gc.collect()
        bank.save(bank_path)
        state["bank_file_sha256"] = _file_digest(bank_path)
        write_json(run/"progress.json", state)
    if args.prepare_only:
        state["status"] = "prepared"
        write_json(run/"progress.json", state)
        print("Prepared frozen review bank; no training yet", flush=True)
        return run
    return train(args, exp, split, queries, bank, c5, fixed_c4, run, state, protected, started)


def train(args, exp, split, queries, bank, c5, fixed_c4, run, state, protected, started):
    import torch
    from rating_recsys.retrieval.text_item_graph import TextItemGraph
    from rating_recsys.retrieval.learned_review_retrieval import LearnedReviewRetriever, LearnedReviewRetrievalConfig
    torch.set_num_threads(args.threads)
    user_ids = np.asarray([q.user_id for q in queries], dtype=np.int64)
    item_ids = bank.item_ids
    popularity = Counter(r.restaurant_id for r in split.train)
    mask = bank.item_review_ids >= 0
    item_means = unit((bank.item_vectors*mask[..., None]).sum(1)/np.maximum(mask.sum(1)[:, None], 1))
    all_ranks = {"c5": c5, "fixed_c4": fixed_c4}
    evaluate = lambda ranks: evaluate_rankings(
        [RankingObservation(q.query_id, q.user_id, q.relevance_by_item, tuple(ranks[q.query_id])) for q in queries],
        cutoffs=(10, 20, 50, 100), catalog_ids=item_ids)
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    (run/"source.diff").write_text(source.pop("git_diff"))
    state["code"] = source
    for name in ("retrieval/text_item_graph.py", "retrieval/learned_review_retrieval.py", "retrieval/lightgcn.py",
                 "retrieval/review_multivector.py", "experiments/text_retrieval_comparison_cli.py"):
        target = run/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT/"src/rating_recsys"/name, target)
    state.update(status="training", baseline_metrics={"c5": evaluate(c5), "fixed_c4": evaluate(fixed_c4)})
    write_json(run/"progress.json", state)

    def finish_arm(name, model, scores, metadata, fit_seconds, curve, model_path):
        ranks = ranks_from_scores(scores, user_ids, item_ids, queries)
        validate_rankings(queries, {name: ranks}, set(item_ids), 100)
        np.savez_compressed(run/f"{name}.scores.npz", scores=scores, user_ids=user_ids, item_ids=item_ids)
        metrics = evaluate(ranks)
        write_jsonl(run/f"{name}.rankings.jsonl", ({"query_id": q.query_id, "user_id": q.user_id,
                    "relevance_by_item": q.relevance_by_item, name: ranks[q.query_id]} for q in queries))
        all_ranks[name] = ranks
        state["arms"][name] = {"metrics": metrics, "metadata": metadata,
                               "fit_seconds": fit_seconds, "curve": curve,
                               "checkpoint": model_path.name, "checkpoint_sha256": _file_digest(model_path),
                               "scores_sha256": _file_digest(run/f"{name}.scores.npz")}
        write_json(run/"progress.json", state)
        print(f"{name}: Recall100={metrics['recall_at_100']:.4%}, NDCG10={metrics['ndcg_at_10']:.6f}; fit={fit_seconds:.1f}s", flush=True)
        return ranks

    with threadpool_limits(limits=args.threads, user_api="blas"):
        for seed in args.seeds:
            cf = replace(exp.lightgcn_config, seed=seed, epochs=args.graph_epochs)
            expected = sampling_audit(split.train, cf, review_epochs=args.review_epochs)
            base = None
            for condition in ("baseline", "actual", "rewired"):
                name = f"graph_{condition}_seed{seed}"
                if args.reuse_graph_run:
                    previous = read_json(args.reuse_graph_run/"metrics.json")
                    old = previous["arms"][name]
                    old_checkpoint = args.reuse_graph_run/old["checkpoint"]
                    old_scores = args.reuse_graph_run/f"{name}.scores.npz"
                    if (_file_digest(old_checkpoint) != old["checkpoint_sha256"] or
                            _file_digest(old_scores) != old["scores_sha256"]):
                        raise ValueError("Reused graph checkpoint or scores changed")
                    model_path = run/old_checkpoint.name
                    shutil.copy2(old_checkpoint, model_path)
                    with np.load(old_scores, allow_pickle=False) as saved:
                        scores = saved["scores"]
                        if not np.array_equal(saved["user_ids"], user_ids) or not np.array_equal(saved["item_ids"], item_ids):
                            raise ValueError("Reused score axes differ")
                    metadata = dict(old["metadata"])
                    metadata["reused_checkpoint_from"] = str(old_checkpoint.resolve())
                    ranks = finish_arm(name, None, scores, metadata, old["fit_seconds"], old["curve"], model_path)
                    if condition == "baseline":
                        base = LightGCN(cf)
                        with np.load(model_path, allow_pickle=False) as saved:
                            base.user_ids = {int(k): n for n, k in enumerate(saved["user_ids"])}
                            base.item_ids = {int(k): n for n, k in enumerate(saved["item_ids"])}
                            base.item_keys = saved["item_ids"]
                            base._ego = saved["ego"]
                            base._final = np.concatenate([saved["user_vectors"], saved["item_vectors"]])
                    if seed == exp.random_seed and condition == "baseline":
                        if ranks != fixed_c4:
                            raise ValueError("Reused seed42 baseline differs from prepared C4")
                        state["baseline_reproduced_exactly"] = True
                    del scores
                    continue
                model = LightGCN(cf) if condition == "baseline" else TextItemGraph(
                    cf, item_ids, item_means, neighbors=10, weight=.1,
                    shuffle_seed=seed+1000 if condition == "rewired" else None, graph_mode="binary")
                curve = []
                def callback(stats, fitted):
                    curve.append(asdict(stats))
                    state.update(active_arm=name, active_epoch=stats.epoch)
                    write_json(run/"progress.json", state)
                    if stats.epoch == 1 or stats.epoch % 5 == 0:
                        print(f"{name}: epoch {stats.epoch}/{args.graph_epochs}; BPR={stats.bpr_loss:.5f}", flush=True)
                    return False
                clock = time.perf_counter()
                model.fit(split.train, callback=callback)
                fit_seconds = time.perf_counter()-clock
                positions = [model.user_ids[int(uid)] for uid in user_ids]
                scores = model.user_embeddings[positions] @ model.item_embeddings.T
                model_path = run/f"{name}.npz"
                metadata = {"config": cf.to_dict(), "sampling_audit": expected, "parameter_count": model._ego.size}
                if condition == "baseline":
                    np.savez_compressed(model_path, user_ids=np.asarray(list(model.user_ids)), item_ids=model.item_keys,
                                        ego=model._ego, user_vectors=model.user_embeddings, item_vectors=model.item_embeddings)
                    base = model
                else:
                    if (model.initialization_sha256 != expected["initial_ego_sha256"] or
                            model.sampling_history != expected["graph_batches"]):
                        raise ValueError("Graph initialization/sampling differs from common baseline")
                    model.save(model_path)
                    metadata["graph_audit"] = model.graph_audit
                    metadata["observed_initialization_sha256"] = model.initialization_sha256
                    metadata["observed_sampling_history"] = model.sampling_history
                ranks = finish_arm(name, model, scores, metadata, fit_seconds, curve, model_path)
                if seed == exp.random_seed and condition == "baseline" and args.graph_epochs == exp.lightgcn_epochs:
                    if ranks != fixed_c4:
                        raise ValueError("Seed42 baseline does not exactly reproduce prepared C4")
                    state["baseline_reproduced_exactly"] = True
                del scores
                if condition != "baseline":
                    del model
                gc.collect()
            assert base is not None
            for mode, text in (("mean", "actual"), ("late", "actual"), ("late", "shuffled")):
                name = f"review_{mode}_{text}_seed{seed}"
                config = LearnedReviewRetrievalConfig(mode=mode, text_condition=text, epochs=args.review_epochs,
                                                     seed=seed, threads=args.threads,
                                                     balance_text_presence=args.balance_text_presence)
                model = LearnedReviewRetriever(config)
                curve = []
                def callback(stats, fitted):
                    curve.append(asdict(stats))
                    state.update(active_arm=name, active_epoch=stats.epoch)
                    write_json(run/"progress.json", state)
                    print(f"{name}: epoch {stats.epoch}/{args.review_epochs}; BPR={stats.bpr_loss:.5f}, text_coef={stats.coefficient:.4f}", flush=True)
                    return False
                clock = time.perf_counter()
                model.fit(split.train, bank=bank, cf_user_ids=np.asarray(list(base.user_ids)),
                          cf_item_ids=base.item_keys, cf_user_vectors=base.user_embeddings,
                          cf_item_vectors=base.item_embeddings, cutoff=split.train_cutoff, callback=callback)
                fit_seconds = time.perf_counter()-clock
                actual_sampling = model.metadata["sampling_audit"]
                if actual_sampling != expected["epochs"][:args.review_epochs]:
                    raise ValueError("Review BPR sampling differs from common baseline")
                clock = time.perf_counter()
                scores = model.score_all(user_ids=user_ids,
                    progress=lambda done, total: print(f"{name}: full-catalog scoring {done}/{total}", flush=True) if done == total or done % 256 == 0 else None)
                model.metadata["validation_scoring_seconds"] = time.perf_counter()-clock
                model_path = run/f"{name}.pt"
                model.save(model_path, include_banks=False)
                finish_arm(name, model, scores, model.metadata, fit_seconds, curve, model_path)
                del model, scores
                gc.collect()
            del base
            gc.collect()

    state["mean_seed_metrics"] = {}
    families = ("graph_baseline", "graph_actual", "graph_rewired", "review_mean_actual", "review_late_actual", "review_late_shuffled")
    for family in families:
        metrics = [state["arms"][f"{family}_seed{seed}"]["metrics"] for seed in args.seeds]
        state["mean_seed_metrics"][family] = {key: float(np.mean([m[key] for m in metrics]))
            for key in ("recall_at_20", "recall_at_50", "recall_at_100", "ndcg_at_10")}
    comparisons = (("graph_actual", "graph_baseline"), ("graph_actual", "graph_rewired"),
                   ("review_late_actual", "graph_baseline"), ("review_late_actual", "review_mean_actual"),
                   ("review_late_actual", "review_late_shuffled"), ("review_late_actual", "graph_actual"))
    state["paired_comparisons"] = {}
    for after, before in comparisons:
        key = f"{after}_minus_{before}"
        a = [all_ranks[f"{before}_seed{s}"] for s in args.seeds]
        b = [all_ranks[f"{after}_seed{s}"] for s in args.seeds]
        state["paired_comparisons"][key] = {
            "recall_at_100": mean_seed_comparison(queries, a, b),
            "ndcg_at_10": mean_seed_comparison(queries, a, b, cutoff=10, metric="ndcg")}
    for family in ("graph_actual", "review_late_actual"):
        state["paired_comparisons"][f"{family}_minus_c5"] = {
            "recall_at_100": mean_seed_comparison(queries, [c5]*len(args.seeds),
                [all_ranks[f"{family}_seed{s}"] for s in args.seeds])}
    for seed in args.seeds:
        arms = {"c5": c5, **{family: all_ranks[f"{family}_seed{seed}"] for family in families}}
        segments, changes = analyze_segments(queries, arms, popularity, baseline="graph_baseline")
        write_json(run/f"segments_seed{seed}.json", segments)
        write_jsonl(run/f"positive_changes_seed{seed}.jsonl", changes)
    write_jsonl(run/"rankings_validation.jsonl", ({"query_id": q.query_id, "user_id": q.user_id,
        "relevance_by_item": q.relevance_by_item, **{arm: ranks[q.query_id] for arm, ranks in all_ranks.items()}}
        for q in queries))
    state["ranker_hashes_after"] = protect_ranker(args.baseline_run)
    if state["ranker_hashes_after"] != protected:
        raise ValueError("Ranker changed during retrieval comparison")
    state.update(status="complete", seconds_this_process=time.perf_counter()-started,
                 peak_rss_kib=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss)
    write_json(run/"metrics.json", state)
    write_json(run/"progress.json", state)
    print(f"Complete: {run}", flush=True)
    return run


if __name__ == "__main__":
    main()
