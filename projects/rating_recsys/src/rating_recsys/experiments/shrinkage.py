"""One V2 ablation: raw rating means versus shrinkage, with shared C5 retrieval.

Run with ``python -m rating_recsys.experiments.shrinkage --snapshot <file>``.
The existing prefix/relevance baseline, seed and ranker grid are held fixed.
Only user_average_rating and item_average_rating change; no V1 code is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import evaluate_rankings
from rating_recsys.experiments.artifacts import write_json, write_jsonl
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.prepared import PreparedData, add_cache_arguments
from rating_recsys.experiments.pipeline import (
    STAGE1, CheckpointedLightGCN, ExperimentResult, _ranker, _ranker_grid, observations,
    paired_bootstrap, positive_eval_set, rank_window, ranked_ids, select_ranker,
)
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import code_manifest, environment_manifest, freeze_snapshot, load_snapshot


def shrink_feature_matrix(features, feature_names, prior, strength):
    """Apply (n * mean + strength * prior)/(n + strength) to two columns.

    In C5 feature rows popularity_score is the exact past review count, even
    outside C0's Top-k. user_history_length is the user's past review count.
    ``prior`` is either the window's scalar mean or a past-only prior per row.
    The remaining columns, label groups and candidate order stay identical.
    """
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("shrinkage strength must be finite and nonnegative")
    result = features.copy()
    if not strength:
        return result
    prior = np.asarray(prior, dtype=np.float64)
    if prior.ndim > 1 or (prior.ndim == 1 and len(prior) != len(features)):
        raise ValueError("rating prior must be a scalar or one value per feature row")
    for mean_name, count_name in (
        ("user_average_rating", "user_history_length"),
        ("item_average_rating", "popularity_score"),
    ):
        mean_index, count_index = feature_names.index(mean_name), feature_names.index(count_name)
        counts = features[:, count_index].astype(np.float64)
        result[:, mean_index] = (features[:, mean_index] * counts + strength * prior) / (counts + strength)
    return result


def _tune(name, config, arrays, candidates, queries, strength, run_dir, emit):
    arm_config = replace(config, rating_shrinkage_strength=strength)
    priors = np.repeat(arrays.rating_priors, arrays.groups)
    features = shrink_feature_matrix(arrays.features, config.feature_names, priors, strength)
    arm_candidates = replace(candidates, features=shrink_feature_matrix(
        candidates.features, config.feature_names, candidates.rating_prior, strength,
    ))
    eval_set = positive_eval_set(arm_candidates)
    grid, models = [], {}
    for params in _ranker_grid(config):
        started = time.perf_counter()
        model = _ranker(arm_config, params)
        model.fit(features, arrays.labels, arrays.groups, eval_set=eval_set,
                  early_stopping_rounds=config.early_stopping_rounds if params["early_stopping"] else None)
        ranked = rank_window(model, queries, arm_candidates)
        scores = evaluate_rankings(observations(queries, ranked_ids(ranked)),
                                   cutoffs=config.ranking_cutoffs, catalog_ids=candidates.catalog)
        iterations = (model.best_iteration or int(params["n_estimators"])) if params["early_stopping"] else int(params["n_estimators"])
        grid.append({**params, **scores, "best_iteration": iterations,
                     "fit_seconds": round(time.perf_counter() - started, 3),
                     "validation_curve": model.validation_curve})
        models[str(params["name"])] = model
        emit(f"{name} {params['name']}: trees={iterations}, validation NDCG@{config.ranking_k}={scores[f'ndcg_at_{config.ranking_k}']:.6f}")
    chosen = select_ranker(grid, config)
    model = models[str(chosen["name"])]
    model.save(run_dir / f"validation_{name}.txt")
    ranked = rank_window(model, queries, arm_candidates)
    params = {key: chosen[key] for key in ("num_leaves", "min_child_samples")}
    params["n_estimators"] = int(chosen["best_iteration"])
    return {"config": arm_config.to_dict(), "grid": grid, "chosen": chosen["name"],
            "final_params": params}, ranked


def _metrics(queries, candidates, ranked, config):
    return evaluate_rankings(
        observations(queries, ranked_ids(ranked)), cutoffs=config.ranking_cutoffs,
        catalog_ids=candidates.catalog, item_popularity=candidates.item_popularity,
        item_regions=candidates.item_regions,
    )


def _save_queries(run_dir, phase, queries, candidates, ranked):
    write_jsonl(run_dir / f"recommendations_{phase}.jsonl", (
        {"query_id": q.query_id, "user_id": q.user_id, "cutoff": q.cutoff.isoformat(),
         "history_restaurant_ids": [v.restaurant_id for v in q.history],
         "ratings_by_item": {v.restaurant_id: v.rating for v in q.window},
         "relevance_by_item": q.relevance_by_item,
         "c5_candidates": candidates.ordered[STAGE1][q.query_id],
         **{name: [{"restaurant_id": i, "score": score} for i, score in rows[q.query_id]]
            for name, rows in ranked.items()}}
        for q in queries
    ))


def run_shrinkage_comparison(interactions, *, project_root, artifacts_root, strength=10.0,
                             config=None, log=None, use_cache=True, rebuild_cache=False):
    config = config or ExperimentConfig()
    if config.ranker_training_mode != "prefix" or config.ranker_label_mode != "relevance" or config.rating_shrinkage_strength:
        raise ValueError("This ablation holds the existing prefix/relevance baseline fixed")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("The one shrinkage treatment must have a finite positive strength")
    emit = log or (lambda s: print(s, file=sys.stderr, flush=True))
    rows = tuple(interactions)
    source = code_manifest(project_root, allow_dirty=True)
    snapshot_path, snapshot = freeze_snapshot(rows, artifacts_root / "snapshots")
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S%fZ") + f"-{snapshot['dataset_snapshot_id'][:8]}"
    run_dir = artifacts_root / "comparisons" / "shrinkage" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    diff = source.pop("git_diff", "")
    (run_dir / "source.diff").write_text(diff, encoding="utf-8")
    runner_source = Path(__file__).read_text(encoding="utf-8")
    (run_dir / "runner_source.py").write_text(runner_source, encoding="utf-8")
    source.update(git_diff_artifact="source.diff", runner_source_artifact="runner_source.py",
                  runner_source_sha256=hashlib.sha256(runner_source.encode()).hexdigest())
    timings = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now
        emit(f"[{name}] {timings[name]:.1f}s")

    split = build_global_temporal_split(rows, train_fraction=config.train_fraction,
                                        validation_fraction=config.validation_fraction)
    t1, t2 = split.train_cutoff, split.validation_cutoff
    validation_queries, validation_new = build_window_queries(
        split.train, split.validation, config=config, phase="validation", cutoff=t1,
    )
    test_history = split.train + split.validation
    test_queries, test_new = build_window_queries(
        test_history, split.test, config=config, phase="test", cutoff=t2,
    )
    graph_config = config.lightgcn_config
    prepared = PreparedData(artifacts_root, snapshot["dataset_snapshot_id"], split, config,
                            enabled=use_cache, rebuild=rebuild_cache, log=emit)
    graphs = CheckpointedLightGCN(graph_config, months=config.lightgcn_checkpoint_months, log=emit)
    train = prepared.training("train", split.train, t1, graphs)
    train_summary = train.summary
    validation, validation_graph_info = prepared.window("validation", split.train, validation_queries)
    lap("01_shared_validation_retrieval_and_training_rows")
    arms = {"baseline": 0.0, "shrinkage": strength}
    selection, validation_ranked = {}, {}
    for name, arm_strength in arms.items():
        selection[name], validation_ranked[name] = _tune(
            name, config, train, validation, validation_queries, arm_strength, run_dir, emit,
        )
    validation_metrics = {name: _metrics(validation_queries, validation, ranked, config)
                          for name, ranked in validation_ranked.items()}
    _save_queries(run_dir, "validation", validation_queries, validation, validation_ranked)
    del validation, validation_ranked
    gc.collect()
    lap("02_select_both_rankers_on_validation")
    refit = prepared.training("refit", test_history, t2, graphs)
    refit_summary = refit.summary
    features = np.concatenate([train.features, refit.features])
    labels = np.concatenate([train.labels, refit.labels])
    groups = train.groups + refit.groups
    priors = np.repeat(np.concatenate([train.rating_priors, refit.rating_priors]), groups)
    del train, refit
    gc.collect()
    models, importance = {}, {}
    for name, arm_strength in arms.items():
        arm_features = shrink_feature_matrix(features, config.feature_names, priors, arm_strength)
        model = _ranker(replace(config, rating_shrinkage_strength=arm_strength), selection[name]["final_params"])
        model.fit(arm_features, labels, groups)
        model.save(run_dir / f"model_{name}.txt")
        models[name], importance[name] = model, model.feature_importance()
        del arm_features
    del features, labels, groups, priors
    gc.collect()
    lap("03_shared_refit_rows_and_final_rankers")
    test, test_graph_info = prepared.window("test", test_history, test_queries)
    test_ranked = {}
    for name, arm_strength in arms.items():
        arm_candidates = replace(test, features=shrink_feature_matrix(
            test.features, config.feature_names, test.rating_prior, arm_strength,
        ))
        test_ranked[name] = rank_window(models[name], test_queries, arm_candidates)
    test_metrics = {name: _metrics(test_queries, test, ranked, config) for name, ranked in test_ranked.items()}
    shared = {
        "c5": evaluate_rankings(observations(test_queries, test.ordered[STAGE1]),
                                 cutoffs=config.candidate_cutoffs, catalog_ids=test.catalog),
        "r0": evaluate_rankings(observations(test_queries, test.ordered[STAGE1]),
                                 cutoffs=config.ranking_cutoffs, catalog_ids=test.catalog),
    }
    bootstrap = {f"at_{k}": paired_bootstrap(
        test_queries, ranked_ids(test_ranked["shrinkage"]), ranked_ids(test_ranked["baseline"]),
        cutoff=k, samples=config.bootstrap_samples, seed=config.random_seed,
    ) for k in config.ranking_cutoffs}
    _save_queries(run_dir, "test", test_queries, test, test_ranked)
    candidate_digest = hashlib.sha256(json.dumps(test.ordered[STAGE1], sort_keys=True).encode()).hexdigest()
    lap("04_single_test_evaluation_per_arm")
    metrics = {"validation": validation_metrics, "test": test_metrics, "shared": shared,
               "bootstrap_shrinkage_minus_baseline": bootstrap, "selection": selection,
               "training": {"train_prefix": train_summary, "refit_validation_window_prefix": refit_summary},
               "feature_importance": importance,
               "lightgcn": {"validation": validation_graph_info, "test": test_graph_info,
                            "training_checkpoints": graphs.rows()}}
    manifest = {
        "run_id": run_id, "created_at": created.isoformat(), "config": config.to_dict(),
        "prepared_data": prepared.manifest,
        "shrinkage_strength": strength, "shrinkage_strength_selection": "fixed at 10 before evaluation; no strength grid" if strength == 10 else "fixed before evaluation; no strength grid",
        "snapshot": {**snapshot, "path": str(snapshot_path)}, "split": split.summary(),
        "code": source, "environment": {**environment_manifest(), "thread_limits": {
            key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        }}, "timings_seconds": timings,
        "scope": "V2 only; replace two mean-rating features; shared C5 extraction",
        "windows": {"validation_queries": len(validation_queries), "test_queries": len(test_queries),
                    "validation_new_users": validation_new, "test_new_users": test_new},
        "leakage_checks": {"prior": "query-ordered past interactions for prefix; history through window cutoff for evaluation",
                           "training_targets_through": t1.isoformat(), "refit_targets_through": t2.isoformat(),
                           "shared_test_candidates_sha256": candidate_digest, "test_evaluations_per_arm": 1,
                           "candidate_injection": False, "v1_code_used": False},
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "report.md").write_text(render_report(manifest, metrics), encoding="utf-8")
    return ExperimentResult(run_id=run_id, run_dir=run_dir, metrics=metrics, manifest=manifest)


def render_report(manifest, metrics):
    k, ck = manifest["config"]["ranking_k"], manifest["config"]["candidate_k"]
    delta = metrics["bootstrap_shrinkage_minus_baseline"][f"at_{k}"].get(f"ndcg_at_{k}")
    if delta:
        low, high = delta["ci95"]
        verdict = "NDCG 개선" if low > 0 else "NDCG 악화" if high < 0 else "NDCG 차이를 확정할 근거 부족"
        comparison = f"NDCG@{k} Δ={delta['delta']:+.6f}, paired bootstrap 95% CI [{low:+.6f}, {high:+.6f}]. **{verdict}**."
    else:
        comparison = "평가 가능한 positive query가 없어 NDCG 차이와 신뢰구간을 계산하지 않았다."
    lines = [f"# V2 평점 평균 shrinkage 비교 · {manifest['run_id']}", "",
             f"- Snapshot `{manifest['snapshot']['dataset_snapshot_id'][:16]}` · {manifest['snapshot']['interactions']:,} interactions",
             f"- 사용자 {manifest['snapshot']['users']:,}명 · 식당 {manifest['snapshot']['restaurants']:,}개 · test 평가 query {metrics['test']['baseline']['evaluated_queries']:,}개",
             f"- Train ≤ {manifest['split']['cutoffs']['train_through']}, validation ≤ {manifest['split']['cutoffs']['validation_through']}, test는 그 이후",
             f"- 실행 단계 합계 {sum(manifest['timings_seconds'].values()) / 60:.1f}분 · validation/test 기간에 처음 등장한 사용자는 정확도 평가에서 제외",
             f"- 조건: C5 Top-{ck}, prefix/relevance, seed {manifest['config']['random_seed']}, 동일 ranker grid와 동점 처리",
             f"- 처리: 사용자·식당 평균 두 feature에만 `(평점 합 + λ × 과거 전체 평균)/(평가 수 + λ)`, λ={manifest['shrinkage_strength']:g}",
             "- λ는 결과를 보기 전에 고정했다. 모델 종류·텍스트·label·window는 유지했다. V1은 사용하지 않음.",
             "- 후보 추출·학습 group·label은 공유하고 두 ranker의 설정과 트리 수는 같은 규칙으로 validation에서 각각 선택.", "",
             "## 결과", "", f"공통 C5 Recall@{ck}: {metrics['shared']['c5'][f'recall_at_{ck}']:.4%}", "",
             "| 지표 | Baseline | Shrinkage | 차이 (shrinkage − baseline) |", "|---|---:|---:|---:|"]
    for cutoff in sorted(metrics["bootstrap_shrinkage_minus_baseline"], key=lambda v: int(v.split("_")[-1])):
        metric_k = int(cutoff.split("_")[-1])
        for metric in ("ndcg", "recall", "precision", "map", "mrr", "catalog_coverage", "novelty", "intra_list_region_diversity"):
            key = f"{metric}_at_{metric_k}"
            if key not in metrics["test"]["baseline"]: continue
            b, s = metrics["test"]["baseline"][key], metrics["test"]["shrinkage"][key]
            lines.append(f"| {metric}@{metric_k} | {b:.6f} | {s:.6f} | {s-b:+.6f} |")
    lines += ["", comparison, "",
              "## Validation 선택", "", "| 조건 | leaves | min_child_samples | trees | Val NDCG |", "|---|---:|---:|---:|---:|"]
    for name in ("baseline", "shrinkage"):
        selected = metrics["selection"][name]
        p = selected["final_params"]
        lines.append(f"| {name} | {p['num_leaves']} | {p['min_child_samples']} | {p['n_estimators']} | {metrics['validation'][name][f'ndcg_at_{k}']:.6f} |")
    lines += ["", "## 해석과 한계", "",
              "- 기본값은 λ=0으로 유지한다. 이 한 번의 비교로 shrinkage를 기본 모델에 채택하지 않는다.",
              "- 이 실험은 평점 평균 feature 두 개의 shrinkage만 측정한다. 사용자별 label 보정이나 MF bias 모델의 효과로 확대 해석하지 않는다.",
              "- 후보 Recall은 같은 추출 결과이므로 동일하다. LTR 이후의 순위 성능으로 판단한다.",
              "- Seed 1회, 보정 강도 1개. 이미 확인한 test window의 탐색적 비교이며 최종 일반화 확인에는 새 미래 holdout이 필요하다.",
              "- 과거 2026-09-30 run과 달리 현재 동점 처리와 공통 validation metric을 양쪽에 적용해 baseline을 다시 학습했다.", "",
              "## 재현", "", "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python -m rating_recsys.experiments.shrinkage \\",
              f"  --snapshot artifacts/snapshots/{Path(manifest['snapshot']['path']).name} \\",
              f"  --strength {manifest['shrinkage_strength']:g}", "```", "",
              "원본: manifest.json, metrics.json, recommendations_validation/test.jsonl(공유 C5 후보·정답·양쪽 순위), model_baseline/shrinkage.txt, source.diff, runner_source.py."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=10.0)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    add_cache_arguments(parser)
    args = parser.parse_args(argv)
    result = run_shrinkage_comparison(load_snapshot(args.snapshot), project_root=PROJECT_ROOT,
                                      artifacts_root=args.artifacts_dir, strength=args.strength,
                                      use_cache=not args.no_cache, rebuild_cache=args.rebuild_cache)
    print(json.dumps({"run_id": result.run_id, "report": str(result.run_dir / "report.md"),
                      "test": result.metrics["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
