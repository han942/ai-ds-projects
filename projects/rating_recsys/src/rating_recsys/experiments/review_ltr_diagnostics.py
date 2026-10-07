"""Audit saved LTR rankings without training, prediction or embedding generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from dataclasses import fields
from pathlib import Path
from statistics import mean, median

import numpy as np

from rating_recsys.evaluation.metrics import RankingObservation, query_scores
from rating_recsys.experiments.artifacts import write_json


def _ideal(grades, cutoff):
    return sum((2 ** grade - 1) / math.log2(rank + 2)
               for rank, grade in enumerate(sorted(grades, reverse=True)[:cutoff]))


def _paired(left, right, *, samples, seed):
    delta = np.asarray(left) - np.asarray(right)
    rng = np.random.default_rng(seed)
    boot = [float(delta[rng.integers(len(delta), size=len(delta))].mean())
            for _ in range(samples)]
    return {"delta": float(delta.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "wins": int((delta > 0).sum()), "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum())}


def diagnose_rankings(records, *, cutoff=10, samples=2000, seed=42):
    if cutoff < 1 or samples < 1:
        raise ValueError("cutoff and samples must be positive")
    records = list(records)
    if len({row["query_id"] for row in records}) != len(records):
        raise ValueError("Duplicate query IDs")
    arms = ("c5_candidates", "baseline", "review_ltr", "oracle")
    scores = {arm: [] for arm in arms}
    conditional = {arm: [] for arm in arms[:-1]}
    ratios, candidate_rows, observed_rows, positive_rows, low_rows = [], 0, 0, 0, 0
    retrieved, excluded, differing_idcg = 0, 0, 0
    for row in records:
        candidates = tuple(row["c5_candidates"])
        pool = set(candidates)
        if len(pool) != len(candidates):
            raise ValueError("Duplicate candidates")
        for arm in ("baseline", "review_ltr"):
            ids = row[arm]
            if len(ids) != len(pool) or set(ids) != pool:
                raise ValueError("Ranker changed the candidate pool")
        grades = {int(key): value for key, value in row["relevance_by_item"].items()}
        if any(value not in (0, 1, 2) for value in grades.values()):
            raise ValueError("Expected common relevance grades 0/1/2")
        candidate_rows += len(pool)
        observed_rows += len(pool & grades.keys())
        positive_rows += sum(grades.get(item, 0) > 0 for item in pool)
        low_rows += sum(item in grades and grades[item] == 0 for item in pool)
        relevant = {item: value for item, value in grades.items() if value > 0}
        if not relevant:
            excluded += 1
            continue
        local = {item: value for item, value in relevant.items() if item in pool}
        retrieved += bool(local)
        oracle = tuple(sorted(candidates, key=lambda item: -grades.get(item, 0)))
        for arm in arms:
            ids = oracle if arm == "oracle" else tuple(row[arm])
            observation = RankingObservation(row["query_id"], row["user_id"], grades, ids)
            scores[arm].append(query_scores(observation, cutoff))
            if local and arm != "oracle":
                conditional[arm].append(query_scores(
                    RankingObservation(row["query_id"], row["user_id"], local, ids), cutoff)["ndcg"])
        if local:
            global_idcg = _ideal(relevant.values(), cutoff)
            local_idcg = _ideal(local.values(), cutoff)
            ratios.append(local_idcg / global_idcg)
            differing_idcg += not math.isclose(local_idcg, global_idcg, abs_tol=1e-12)
    n = len(scores["baseline"])
    if not n:
        raise ValueError("No relevant queries")
    metrics = {arm: {metric: mean(value[metric] for value in values)
                     for metric in ("ndcg", "recall", "precision", "map", "mrr")}
               for arm, values in scores.items()}
    comparisons = {}
    for left, right in (("review_ltr", "baseline"), ("baseline", "c5_candidates"),
                        ("review_ltr", "c5_candidates")):
        comparisons[f"{left}_minus_{right}"] = {
            metric: _paired([s[metric] for s in scores[left]], [s[metric] for s in scores[right]],
                            samples=samples, seed=seed)
            for metric in ("ndcg", "recall")
        }
    return {
        "queries": len(records), "evaluated_queries": n, "excluded_no_relevant": excluded,
        "candidate_rows": candidate_rows, "observed_candidate_rows": observed_rows,
        "unobserved_candidate_rows": candidate_rows - observed_rows,
        "retrieved_positive_rows": positive_rows, "observed_low_grade_rows": low_rows,
        "recoverable_queries": retrieved, "unrecoverable_queries": n - retrieved,
        "unrecoverable_query_fraction": (n - retrieved) / n,
        "metrics": metrics, "paired_comparisons": comparisons,
        "early_stopping_diagnostic": {
            "queries": retrieved, "different_idcg_queries": differing_idcg,
            "median_candidate_to_global_idcg": median(ratios) if ratios else None,
            "candidate_normalized_ndcg": {arm: mean(values) if values else None
                                          for arm, values in conditional.items()},
            "interpretation": "Missing-positive queries add constant zeros; differing IDCG reweights users.",
        },
        "caveats": [
            "Oracle is a hindsight ceiling, not a deployable ranker.",
            "Unobserved candidates are not known dislikes; exposure logs are unavailable.",
            "Post-hoc bootstrap does not correct model selection or repeated holdout use.",
            "Candidate-normalized scores use saved order; exact-score ties may differ from callback ties.",
        ],
    }


def diagnose_review_signal(manifest, records, *, snapshot, review_cache, tokenizer_path):
    """Read pinned cached vectors only; never instantiate an embedding backend."""
    from tokenizers import Tokenizer

    from rating_recsys.datasets.split import build_global_temporal_split
    from rating_recsys.experiments.snapshot import dataset_digest, load_review_texts, load_snapshot, review_texts_path
    from rating_recsys.retrieval.review_embeddings import ProfileFormatter, ReviewEmbeddingConfig, _unit
    from rating_recsys.retrieval.review_profiles import build_profiles

    embedding = ReviewEmbeddingConfig(**{field.name: manifest["embedding"][field.name]
                                        for field in fields(ReviewEmbeddingConfig) if field.init})
    if embedding.backend != "local" or embedding.aggregation != "concat":
        raise ValueError("Review signal audit currently requires cached local concat profiles")
    tokenizer_bytes = tokenizer_path.read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != manifest["preprocessing"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum differs from the evaluated run")
    rows = load_snapshot(snapshot)
    if dataset_digest(rows) != manifest["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Snapshot differs from the evaluated run")
    texts, texts_meta = load_review_texts(review_texts_path(snapshot), rows)
    if texts_meta["artifact_sha256"] != manifest["review_texts"]["artifact_sha256"]:
        raise ValueError("Review text snapshot differs from the evaluated run")
    # Keep the executed split fixed, rather than taking today's CLI defaults.
    config = manifest["config"]
    split = build_global_temporal_split(rows, train_fraction=config["train_fraction"],
                                       validation_fraction=config["validation_fraction"])
    if any(row["cutoff"] != split.train_cutoff.isoformat() for row in records):
        raise ValueError("Signal audit accepts validation rankings at T1 only")
    users, items = build_profiles(split.train, texts, [row["user_id"] for row in records], embedding)
    candidate_ids = {item for row in records for item in row["c5_candidates"]}
    items = {key: value for key, value in items.items() if key in candidate_ids}
    formatter = ProfileFormatter(embedding, tokenizer_path.parent,
                                 tokenizer=Tokenizer.from_str(tokenizer_bytes.decode()))
    truncated = {
        "user": sum(len(formatter.tokenizer.encode(f"{embedding.query_prefix}: {text}").ids)
                    > embedding.max_document_tokens for text in users.values()),
        "item": sum(len(formatter.tokenizer.encode(f"{embedding.document_prefix}: {text}").ids)
                    > embedding.max_document_tokens for text in items.values()),
    }
    user_docs, item_docs = formatter.prepare(users, items)
    identity = {
        "model": embedding.model, "model_revision": embedding.model_revision,
        "tokenizer_revision": embedding.tokenizer_revision,
        "profile_version": embedding.profile_version, "dimensions": embedding.dimensions,
        "max_tokens": embedding.max_document_tokens, "dtype": "float32", "device": "cpu",
    }
    vectors = {}
    with sqlite3.connect(f"{review_cache.resolve().as_uri()}?mode=ro", uri=True) as db:
        for doc in set(user_docs.values()) | set(item_docs.values()):
            key = hashlib.sha256(json.dumps([identity, doc], ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
            record = db.execute("SELECT vector,model,dimensions FROM vectors WHERE key=?", (key,)).fetchone()
            if record is None or record[1:] != (embedding.model, embedding.dimensions):
                raise ValueError("Cached validation vector missing or mismatched")
            vector = np.asarray(_unit(json.loads(record[0])), dtype=np.float32)
            if (vector.shape != (embedding.dimensions,) or not np.isfinite(vector).all()
                    or np.linalg.norm(vector) < 1e-6):
                raise ValueError("Invalid cached vector")
            vectors[doc] = vector
    aucs, differences, positives, others = [], [], [], []
    pairs, total_rows = 0, 0
    for row in records:
        total_rows += len(row["c5_candidates"])
        if row["user_id"] not in user_docs:
            continue
        user = vectors[user_docs[row["user_id"]]]
        grades = {int(key): value for key, value in row["relevance_by_item"].items()}
        positive, other = [], []
        for item in row["c5_candidates"]:
            if item not in item_docs:
                continue
            cosine = float(np.clip(user @ vectors[item_docs[item]], -1, 1))
            pairs += 1
            (positive if grades.get(item, 0) > 0 else other).append(cosine)
        positives.extend(positive)
        others.extend(other)
        if positive and other:
            p, o = np.asarray(positive)[:, None], np.asarray(other)[None, :]
            aucs.append(float(((p > o) + 0.5 * (p == o)).mean()))
            differences.append(mean(positive) - mean(other))
    return {
        "scope": "validation only; cached vectors; float32 normalized dot product as actual features",
        "queries": len(records), "signal_queries": len(aucs), "candidate_rows": total_rows,
        "pair_rows": pairs, "pair_coverage": pairs / total_rows,
        "positive_pair_rows": len(positives), "other_pair_rows": len(others),
        "within_query_pairwise_auc_macro": mean(aucs) if aucs else None,
        "within_query_pairwise_auc_median": median(aucs) if aucs else None,
        "mean_query_positive_minus_other_cosine": mean(differences) if differences else None,
        "median_positive_cosine": median(positives) if positives else None,
        "median_other_cosine": median(others) if others else None,
        "user_profiles": len(user_docs), "candidate_item_profiles": len(item_docs),
        "truncated_user_profiles": truncated["user"], "truncated_item_profiles": truncated["item"],
        "encoder_loaded": False, "cache_misses": 0,
        "caveats": ["Other candidates are unobserved or grade zero, not proven dislikes.",
                    "Macro AUC averages only users with both positive and other vector pairs.",
                    "Signal separation is descriptive, not a causal feature ablation or significance test."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true",
                        help="Descriptive audit only; never use existing test to select another model")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--review-cache", type=Path, help="Optional read-only validation cosine audit")
    parser.add_argument("--snapshot", type=Path, help="Required with --review-cache")
    parser.add_argument("--tokenizer", type=Path, help="Required with --review-cache; pinned tokenizer.json")
    args = parser.parse_args(argv)
    if args.review_cache and (args.snapshot is None or args.tokenizer is None):
        parser.error("--review-cache requires --snapshot and --tokenizer")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    expected = json.loads((args.run_dir / "metrics.json").read_text())
    if json.loads((args.run_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("Only audit completed runs")
    output = {"run_id": manifest["run_id"], "cutoff": manifest["config"]["ranking_k"],
              "samples": args.samples, "phases": {},
              "new_training": 0, "new_predictions": 0, "embedding_requests": 0}
    phases = ("validation", "test") if args.include_test else ("validation",)
    for phase in phases:
        path = args.run_dir / f"recommendations_{phase}.jsonl"
        data = path.read_bytes()
        records = [json.loads(line) for line in data.splitlines()]
        result = diagnose_rankings(records,
                                   cutoff=output["cutoff"], samples=args.samples,
                                   seed=manifest["config"]["random_seed"])
        ordered = {row["query_id"]: row["c5_candidates"] for row in records}
        order_hash = hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()
        if phase == "test":
            if order_hash != manifest["shared_test_candidates_sha256"]:
                raise ValueError("Saved test C5 order differs from original manifest")
            for metric, value in result["metrics"]["c5_candidates"].items():
                if not math.isclose(value, expected["r0"][f"{metric}_at_{output['cutoff']}"],
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Saved test C5 {metric} does not reproduce")
        entries = manifest.get("prepared_data", {}).get("entries", [])
        for entry in entries:
            if entry["kind"] == phase:
                from rating_recsys.experiments.snapshot import canonical_json

                metadata_path = args.run_dir.parent.parent.parent / entry["path"] / "manifest.json"
                metadata = json.loads(metadata_path.read_text())
                checksum = metadata.pop("metadata_sha256")
                if hashlib.sha256(canonical_json(metadata).encode()).hexdigest() != checksum:
                    raise ValueError("Prepared candidate metadata checksum differs")
                if ordered != metadata["data"]["ordered"]["c5_c1_lightgcn_rrf"]:
                    raise ValueError(f"Saved {phase} C5 order differs from prepared candidates")
        for arm in ("baseline", "review_ltr"):
            for metric, value in result["metrics"][arm].items():
                if not math.isclose(value, expected[phase][arm][f"{metric}_at_{output['cutoff']}"],
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Saved {phase} {arm} {metric} does not reproduce")
        output["phases"][phase] = {"source_sha256": hashlib.sha256(data).hexdigest(),
                                   "c5_order_sha256": order_hash, **result}
        if phase == "validation" and args.review_cache:
            output["review_signal"] = diagnose_review_signal(
                manifest, records, snapshot=args.snapshot,
                review_cache=args.review_cache, tokenizer_path=args.tokenizer)
    write_json(args.output, output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
