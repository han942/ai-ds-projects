"""Compare baseline LambdaRank with frozen review features on identical C5.

python -m rating_recsys.experiments.review_ltr --snapshot <file> --dry-run
python -m rating_recsys.experiments.review_ltr --snapshot <file> --embed-only
python -m rating_recsys.experiments.review_ltr --snapshot <file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.artifacts import write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_training_windows, build_window_queries
from rating_recsys.experiments.snapshot import (
    code_manifest, environment_manifest, freeze_snapshot, load_snapshot,
    load_review_texts, review_texts_path,
)
from rating_recsys.retrieval.review_embeddings import (
    EmbeddingAPIError, create_embedding_cache, ProfileFormatter,
    E5_MODEL, LIQUID_MODEL, NEMOTRON_MODEL, model_embedding_config, profile_api_inputs,
)
from rating_recsys.ranking.review_features import REVIEW_FEATURE_NAMES, ReviewFeatureBuilder


def profile_preflight(rows, texts, config, embedding, formatter, cache, *, log=lambda _: None):
    """Include every train/refit window, not only the two evaluation cutoffs."""
    split = build_global_temporal_split(rows, train_fraction=config.train_fraction,
                                        validation_fraction=config.validation_fraction)
    docs, audit = {}, []
    # T1 partial window and T2 refit share a start/history. The union is sufficient.
    windows = build_training_windows(rows, config=config, through=split.validation_cutoff)
    phases = []
    for phase, history, targets, cutoff in (
        ("validation", split.train, split.validation, split.train_cutoff),
        ("test", split.train + split.validation, split.test, split.validation_cutoff),
    ):
        qs, _ = build_window_queries(history, targets, config=config, phase=phase, cutoff=cutoff)
        phases.append((phase, history, qs))
    # Cache the evaluation profiles first, then historical training profiles.
    # This changes call order only, never the inputs, labels or evaluation.
    phases.extend((f"training:{start}", tuple(r for r in rows if r.event_date < start), qs)
                  for start, qs in windows)
    for phase, history, qs in phases:
        _, _, _, inputs = profile_api_inputs(history, texts, [q.user_id for q in qs], embedding, formatter)
        docs.update(dict.fromkeys(inputs))
        audit.append({"phase": phase, "cutoff": qs[0].cutoff.isoformat(),
                      "queries": len(qs), "unique_inputs": len(set(inputs))})
        log(f"[profiles] {phase}: {len(set(inputs))} inputs")
    if embedding.api_role_mode:
        missing_by_role = {
            role: cache.missing_count([doc for doc in docs if cache._api_input(doc)[0] == role])
            for role in ("query", "passage")
        }
    else:
        missing_by_role = {"mixed": cache.missing_count(list(docs))}
    misses = sum(missing_by_role.values())
    return list(docs), {
        "embedding": embedding.to_dict(), "windows": audit, "unique_inputs": len(docs),
        "unique_cache_misses": misses, "missing_by_role": missing_by_role,
        "estimated_requests": (sum(math.ceil(count / embedding.batch_size) for count in missing_by_role.values())
                               if embedding.backend == "openrouter" else 0),
        "estimated_local_batches": math.ceil(misses / embedding.batch_size) if embedding.backend == "local" else 0,
        "preprocessing": formatter.metadata(),
        "inputs_sha256": hashlib.sha256(json.dumps(sorted(docs), ensure_ascii=False).encode()).hexdigest(),
    }


def _ranker(config, params, names):
    from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
    return LightGBMLambdaRanker(
        random_seed=config.random_seed, ranking_k=config.ranking_k, feature_names=names,
        n_estimators=int(params["n_estimators"]), learning_rate=config.learning_rate,
        num_leaves=int(params["num_leaves"]), min_child_samples=int(params["min_child_samples"]),
        reg_lambda=config.reg_lambda, n_jobs=config.n_jobs, label_gain=config.label_gain,
    )


def run_comparison(rows, texts, texts_meta, *, config, embedding, formatter, cache,
                   artifacts_root, project_root, log=lambda _: None):
    from rating_recsys.evaluation.metrics import evaluate_rankings
    from rating_recsys.experiments.pipeline import (
        STAGE1, CheckpointedLightGCN, _generator, _ranker_grid,
        observations, paired_bootstrap, positive_eval_set, rank_window, ranked_ids,
        select_ranker, window_training_arrays,
    )
    from rating_recsys.experiments.prepared import PreparedData

    if config.ranker_training_mode != "window":
        raise ValueError("Review LTR requires cutoff-safe window training")
    started = time.perf_counter()
    snapshot_path, snapshot = freeze_snapshot(rows, artifacts_root / "snapshots")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + snapshot["dataset_snapshot_id"][:8]
    folder = artifacts_root / "comparisons" / "review_ltr" / run_id
    folder.mkdir(parents=True)
    source = code_manifest(project_root, allow_dirty=True)
    (folder / "source.diff").write_text(source.pop("git_diff", ""), encoding="utf-8")
    write_json(folder / "status.json", {"status": "running"})
    split = build_global_temporal_split(rows, train_fraction=config.train_fraction,
                                        validation_fraction=config.validation_fraction)
    validation_q, _ = build_window_queries(split.train, split.validation, config=config,
                                            phase="validation", cutoff=split.train_cutoff)
    test_history = split.train + split.validation
    test_q, _ = build_window_queries(test_history, split.test, config=config,
                                     phase="test", cutoff=split.validation_cutoff)
    builder = ReviewFeatureBuilder(texts, embedding, formatter, cache, cache_only=True)
    graphs = CheckpointedLightGCN(config.lightgcn_config, months=config.lightgcn_checkpoint_months, log=log)
    prepared = PreparedData(artifacts_root, snapshot["dataset_snapshot_id"], split, config, log=log)
    train = window_training_arrays(split.train, _generator(config), graphs, config=config,
                                   through=split.train_cutoff, log=log, feature_transform=builder)
    validation, _ = prepared.window("validation", split.train, validation_q)
    validation = replace(validation, features=builder(split.train, validation_q, validation))
    widths = {"baseline": len(config.feature_names),
              "review_ltr": len(config.feature_names) + len(REVIEW_FEATURE_NAMES)}
    names = {"baseline": config.feature_names,
             "review_ltr": config.feature_names + REVIEW_FEATURE_NAMES}
    selection, validation_metrics, final_models = {}, {}, {}

    def metrics(queries, candidates, ranked):
        return evaluate_rankings(observations(queries, ranked_ids(ranked)),
                                 cutoffs=config.ranking_cutoffs, catalog_ids=candidates.catalog)

    def save_rankings(phase, queries, candidates, rankings):
        ids = {arm: ranked_ids(value) for arm, value in rankings.items()}
        write_jsonl(folder / f"recommendations_{phase}.jsonl", (
            {"query_id": q.query_id, "user_id": q.user_id, "cutoff": q.cutoff.isoformat(),
             "relevance_by_item": q.relevance_by_item,
             "c5_candidates": candidates.ordered[STAGE1][q.query_id],
             **{arm: value[q.query_id] for arm, value in ids.items()}}
            for q in queries
        ))

    validation_ranked = {}
    for arm, width in widths.items():
        candidates = replace(validation, features=validation.features[:, :width])
        grid = []
        for params in _ranker_grid(config):
            model = _ranker(config, params, names[arm])
            model.fit(train.features[:, :width], train.labels, train.groups,
                      eval_set=positive_eval_set(candidates),
                      early_stopping_rounds=config.early_stopping_rounds if params["early_stopping"] else None)
            ranked = rank_window(model, validation_q, candidates)
            scores = metrics(validation_q, candidates, ranked)
            trees = (model.best_iteration or int(params["n_estimators"])) if params["early_stopping"] else int(params["n_estimators"])
            grid.append({**params, **scores, "best_iteration": trees})
            model.save(folder / f"validation_{arm}_{params['name']}.txt")
            log(f"[{arm}] {params['name']}: val NDCG@{config.ranking_k}={scores[f'ndcg_at_{config.ranking_k}']:.6f}, trees={trees}")
        chosen = select_ranker(grid, config)
        selection[arm] = {"grid": grid, "chosen": chosen}
        # Predict with selected saved model; no repeated test/model selection.
        from lightgbm import Booster
        selected = Booster(model_file=str(folder / f"validation_{arm}_{chosen['name']}.txt"))
        validation_ranked[arm] = rank_window(selected, validation_q, candidates)
        validation_metrics[arm] = metrics(validation_q, candidates, validation_ranked[arm])
    save_rankings("validation", validation_q, validation, validation_ranked)
    write_json(folder / "selection.json", selection)
    del train
    # Rebuild the complete T2 windows. Never append the overlapping partial T1 block.
    refit = window_training_arrays(test_history, _generator(config), graphs, config=config,
                                   through=split.validation_cutoff, phase="refit", log=log,
                                   feature_transform=builder)
    for arm, width in widths.items():
        chosen = selection[arm]["chosen"]
        params = {key: chosen[key] for key in ("num_leaves", "min_child_samples")}
        params["n_estimators"] = chosen["best_iteration"]
        model = _ranker(config, params, names[arm])
        model.fit(refit.features[:, :width], refit.labels, refit.groups)
        model.save(folder / f"model_{arm}.txt")
        final_models[arm] = model
    test, _ = prepared.window("test", test_history, test_q)
    test = replace(test, features=builder(test_history, test_q, test))
    rankings = {arm: rank_window(model, test_q, replace(test, features=test.features[:, :widths[arm]]))
                for arm, model in final_models.items()}
    test_metrics = {arm: metrics(test_q, test, ranking) for arm, ranking in rankings.items()}
    c5 = evaluate_rankings(observations(test_q, test.ordered[STAGE1]),
                          cutoffs=config.candidate_cutoffs, catalog_ids=test.catalog)
    r0 = evaluate_rankings(observations(test_q, test.ordered[STAGE1]),
                          cutoffs=config.ranking_cutoffs, catalog_ids=test.catalog)
    bootstrap = paired_bootstrap(test_q, ranked_ids(rankings["review_ltr"]),
                                 ranked_ids(rankings["baseline"]), cutoff=config.ranking_k,
                                 samples=config.bootstrap_samples, seed=config.random_seed)
    save_rankings("test", test_q, test, rankings)
    result = {"validation": validation_metrics, "test": test_metrics, "c5": c5, "r0": r0,
              "bootstrap_review_minus_baseline": bootstrap, "selection": selection,
              "feature_importance": {arm: model.feature_importance() for arm, model in final_models.items()},
              "final_training": refit.summary}
    manifest = {
        "run_id": run_id, "config": config.to_dict(), "embedding": embedding.to_dict(),
        "feature_names": names, "snapshot": {**snapshot, "path": str(snapshot_path)},
        "review_texts": texts_meta, "split": split.summary(), "code": source,
        "environment": environment_manifest(), "profile_audit": builder.audit,
        "preprocessing": formatter.metadata(), "embedding_usage": cache.usage,
        "prepared_data": prepared.manifest,
        "embedding_cache_totals": cache.totals(), "seconds": time.perf_counter() - started,
        "candidate_policy": "unchanged C5; no review retrieval or target injection",
        "shared_test_candidates_sha256": hashlib.sha256(json.dumps(test.ordered[STAGE1], sort_keys=True).encode()).hexdigest(),
        "test_evaluations_per_arm": 1,
    }
    write_json(folder / "metrics.json", result)
    write_json(folder / "manifest.json", manifest)
    k = config.ranking_k
    report = ["# 리뷰 임베딩 LTR 비교", "", "동일 C5 후보에서 기존 LambdaRank와 리뷰 피처 추가 모델을 비교했다.", "",
              f"| 모델 | Validation NDCG@{k} | Test NDCG@{k} | Test Recall@{k} |", "|---|---:|---:|---:|"]
    for arm in widths:
        report.append(f"| {arm} | {validation_metrics[arm][f'ndcg_at_{k}']:.8f} | {test_metrics[arm][f'ndcg_at_{k}']:.8f} | {test_metrics[arm][f'recall_at_{k}']:.6%} |")
    report += ["", f"공유 C5 Recall@{config.candidate_k}: {c5[f'recall_at_{config.candidate_k}']:.6%}.", "",
               "Bootstrap 리뷰 LTR − 기존 LTR:", "", "```json", json.dumps(bootstrap, ensure_ascii=False, indent=2), "```",
               "", f"피처: cosine·사용자/식당/쌍 프로필 존재 여부·선택 리뷰 수. 최근 긍정 리뷰 문서와 frozen {embedding.model} ({embedding.dimensions}차원)을 사용했다.",
               "리뷰 수는 토큰 잘림 전 선택된 리뷰 수이며 충분성 threshold를 뜻하지 않는다.",
               "기간별 cutoff·모델·소스·피처 schema·비용은 manifest.json, 사용자별 후보와 순위는 recommendations_*.jsonl에 저장했다."]
    (folder / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(folder / "status.json", {"status": "complete"})
    return folder, result


def main(argv=None):
    parser = argparse.ArgumentParser(description="C5를 고정하고 frozen 리뷰 임베딩을 LambdaRank 피처로 비교")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--review-texts", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--embedding-model", choices=(E5_MODEL, LIQUID_MODEL, NEMOTRON_MODEL), default=E5_MODEL)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--embedding-concurrency", type=int)
    parser.add_argument("--embedding-request-timeout", type=float)
    parser.add_argument("--local-model-cache", type=Path)
    parser.add_argument("--cpu-threads", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--embed-only", action="store_true")
    args = parser.parse_args(argv)
    rows = load_snapshot(args.snapshot)
    texts, texts_meta = load_review_texts(args.review_texts or review_texts_path(args.snapshot), rows)
    config = ExperimentConfig(ranker_training_mode="window", satisfaction_mode="history-aware")
    cache_name = {E5_MODEL: "e5_review_embedding_cache.sqlite",
                  NEMOTRON_MODEL: "nemotron_review_embedding_cache.sqlite",
                  LIQUID_MODEL: "review_embedding_cache.sqlite"}[args.embedding_model]
    cache_path = args.embedding_cache or args.artifacts_dir / cache_name
    embedding = model_embedding_config(args.embedding_model, str(cache_path))
    overrides = {name: value for name, value in (
        ("batch_size", args.embedding_batch_size), ("concurrency", args.embedding_concurrency),
        ("request_timeout_seconds", args.embedding_request_timeout),
        ("cpu_threads", args.cpu_threads),
    ) if value is not None}
    if embedding.backend == "local":
        overrides["local_model_cache"] = str(args.local_model_cache or args.artifacts_dir / "local_models")
    embedding = replace(embedding, **overrides)
    formatter = ProfileFormatter(embedding, args.artifacts_dir / "tokenizers")
    log = lambda s: print(s, file=sys.stderr, flush=True)
    with create_embedding_cache(cache_path, embedding, progress=log) as cache:
        if args.dry_run or args.embed_only:
            docs, preflight = profile_preflight(rows, texts, config, embedding, formatter, cache, log=log)
            preflight["api_requests_sent"] = 0
            diagnostic = {E5_MODEL: "review_ltr_e5_preflight.json",
                          NEMOTRON_MODEL: "review_ltr_nemotron_preflight.json",
                          LIQUID_MODEL: "review_ltr_preflight.json"}[args.embedding_model]
            path = args.artifacts_dir / "diagnostics" / diagnostic
            write_json(path, preflight)
            if args.embed_only:
                try:
                    cache.prefill(docs)
                    preflight["embedding_status"] = "complete"
                except KeyboardInterrupt:
                    preflight["embedding_status"] = "paused"
                except EmbeddingAPIError as exc:
                    preflight["embedding_status"] = "blocked_api"
                    preflight["error"] = {"http_status": exc.status, "daily_limit": exc.daily_limit}
                except RuntimeError as exc:
                    preflight["embedding_status"] = "blocked_local" if embedding.backend == "local" else "blocked_request"
                    preflight["error"] = {"message": str(exc)}
                preflight["usage"] = cache.usage
                preflight["api_requests_sent"] = cache.usage["api_requests"]
                preflight["remaining_cache_misses"] = cache.missing_count(docs)
                write_json(path, preflight)
            print(json.dumps({k: v for k, v in preflight.items() if k not in ("windows", "preprocessing")}, ensure_ascii=False, indent=2))
            if preflight.get("embedding_status") == "paused":
                return 130
            return 2 if preflight.get("embedding_status") in {"blocked_api", "blocked_request", "blocked_local"} else 0
        # Fail before expensive graph fitting when vectors have not been restored.
        _, preflight = profile_preflight(rows, texts, config, embedding, formatter, cache, log=log)
        if preflight["unique_cache_misses"]:
            parser.error(f"{preflight['unique_cache_misses']} review embeddings are missing; restore the cache or run --embed-only")
        folder, result = run_comparison(rows, texts, texts_meta, config=config, embedding=embedding,
                                        formatter=formatter, cache=cache, artifacts_root=args.artifacts_dir,
                                        project_root=PROJECT_ROOT, log=log)
        print(json.dumps({"report": str(folder / "report.md"), "test": result["test"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
