"""Compare one candidate model with Stage 1 (C5) on the pipeline's split.

The model is described by a :class:`~rating_recsys.experiments.candidate_models.CandidateModel`.
Split, evaluation queries and C0-C5 are the same as ``pipeline.run_experiment``
(C4 LightGCN uses the config's fixed settings and the window history).

1. Validation. Each grid config is trained on train (<= T1); models read only
   interactions (and, for text models, reviews) up to T1. The validation
   window is scored every ``eval_every`` epochs, which gives the learning
   curve. A config's epoch count is its best validation Recall@candidate_k;
   training stops after ``patience`` evaluations without improvement. The best
   config is chosen, then the policy with the highest validation
   Recall@candidate_k: the model alone, RRF with C1, or RRF with C1 and C4.
2. Test. The chosen config is refit on train + validation window (<= T2) for
   the chosen epoch count and the test window is scored once, with paired
   bootstrap intervals against C5.
3. The Stage 2 ranker is not retrained. ``baseline_run`` compares the Top-K
   with R1 of a pipeline run on the same snapshot and split.

Output: ``artifacts/comparisons/<model>/<run_id>/`` with ``report.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import evaluate_rankings
from rating_recsys.experiments.artifacts import read_json, read_jsonl, write_json, write_jsonl
from rating_recsys.experiments.candidate_models import CandidateModel
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import WindowQuery
from rating_recsys.experiments.pipeline import (
    STAGE1,
    WindowCandidates,
    _generator,
    _window_summary,
    build_context,
    fit_lightgcn,
    observations,
    paired_bootstrap,
    window_candidates,
)
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import (
    code_manifest,
    environment_manifest,
    freeze_snapshot,
)
from rating_recsys.retrieval.hybrid import rrf


BASE_STAGES = (
    STAGE1,
    "c1_item_item",
    "c4_lightgcn",
    "c0_popularity",
    "c2_region_popularity",
    "c3_rrf_union",
)
BASE_LABELS = {
    STAGE1: "C5 C1+LightGCN RRF (현재 기준선)",
    "c1_item_item": "C1 item-item",
    "c4_lightgcn": "C4 LightGCN (config 고정)",
    "c0_popularity": "참고 · C0 전체 인기",
    "c2_region_popularity": "참고 · C2 지역 인기",
    "c3_rrf_union": "참고 · C3 quota RRF",
    "r1_lambdarank": "R1 LambdaRank (기준 run)",
}


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, object]
    manifest: dict[str, object]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def fusions(model: CandidateModel) -> dict[str, tuple[str, ...]]:
    """RRF policies tried with the new source: with C1, and with C1 + C4."""

    return {
        f"rrf_c1_{model.name}": ("c1_item_item", model.name),
        f"rrf_c1_c4_{model.name}": ("c1_item_item", "c4_lightgcn", model.name),
    }


def policies(model: CandidateModel) -> tuple[str, ...]:
    return (model.name, *fusions(model))


def stage_labels(model: CandidateModel) -> dict[str, str]:
    return {
        **BASE_LABELS,
        model.name: f"{model.letter} {model.title} 단독",
        f"rrf_c1_{model.name}": f"RRF C1+{model.letter}",
        f"rrf_c1_c4_{model.name}": f"RRF C1+C4+{model.letter}",
    }


def candidate_cutoffs(config: ExperimentConfig) -> tuple[int, ...]:
    return tuple(sorted({config.ranking_k, *config.candidate_cutoffs}))


def source_rankings(
    model, queries: Sequence[WindowQuery], k: int
) -> dict[str, tuple[int, ...]]:
    """Top-``k`` per query from ``model.recommend``, never a history restaurant."""

    exclude = {q.user_id: [item.restaurant_id for item in q.history] for q in queries}
    ranked = model.recommend([q.user_id for q in queries], exclude, k)
    return {q.query_id: ranked[q.user_id] for q in queries}


def policy_rankings(
    model: CandidateModel,
    queries: Sequence[WindowQuery],
    candidates: WindowCandidates,
    ranked: dict[str, tuple[int, ...]],
    config: ExperimentConfig,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """C0-C5, the new source alone and every fusion, keyed by stage then query."""

    sources = {**candidates.ordered, model.name: ranked}
    stages = {name: dict(sources[name]) for name in (*BASE_STAGES, model.name)}
    for name, members in fusions(model).items():
        stages[name] = {
            q.query_id: rrf(
                [sources[member][q.query_id] for member in members],
                k=config.candidate_k,
                constant=config.rrf_constant,
            )
            for q in queries
        }
    return stages


def evaluate_stages(
    queries: Sequence[WindowQuery],
    stages: dict[str, dict[str, tuple[int, ...]]],
    candidates: WindowCandidates,
    config: ExperimentConfig,
) -> dict[str, dict[str, object]]:
    return {
        name: evaluate_rankings(
            observations(queries, ordered),
            cutoffs=candidate_cutoffs(config),
            catalog_ids=candidates.catalog,
            item_popularity=candidates.item_popularity,
            item_regions=candidates.item_regions,
        )
        for name, ordered in stages.items()
    }


def positive_overlap(
    queries: Sequence[WindowQuery],
    left: dict[str, tuple[int, ...]],
    right: dict[str, tuple[int, ...]],
    k: int,
) -> dict[str, int]:
    """Relevant window visits found in the Top-k of each list, both or neither."""

    counts = {"relevant_positives": 0, "both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for query in queries:
        in_left, in_right = set(left[query.query_id][:k]), set(right[query.query_id][:k])
        for restaurant_id, relevance in query.relevance_by_item.items():
            if relevance <= 0:
                continue
            counts["relevant_positives"] += 1
            a, b = restaurant_id in in_left, restaurant_id in in_right
            key = "both" if a and b else "left_only" if a else "right_only" if b else "neither"
            counts[key] += 1
    return counts


def compare_with_run(
    run_dir: Path,
    snapshot: dict[str, object],
    split_summary: dict[str, object],
    queries: tuple[WindowQuery, ...],
    lists: dict[str, dict[str, tuple[int, ...]]],
    candidates: WindowCandidates,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Top-``ranking_k`` of ``lists`` vs the R1 Top-K of a pipeline run on the same data."""

    manifest = read_json(run_dir / "manifest.json")
    if manifest["snapshot"]["dataset_snapshot_id"] != snapshot["dataset_snapshot_id"]:
        raise ValueError(f"{run_dir} uses a different snapshot")
    if manifest["split"]["cutoffs"] != split_summary["cutoffs"]:
        raise ValueError(f"{run_dir} uses different split cutoffs")
    r1 = {
        row["query_id"]: tuple(item["restaurant_id"] for item in row["recommendations"])
        for row in read_jsonl(run_dir / "recommendations_test.jsonl")
    }
    if set(r1) != {q.query_id for q in queries}:
        raise ValueError(f"{run_dir} has different test queries")
    k = config.ranking_k
    compared = {"r1_lambdarank": r1, **lists}
    return {
        "run_id": manifest["run_id"],
        "cutoff": k,
        "metrics": {
            name: evaluate_rankings(
                observations(queries, ordered),
                cutoffs=config.ranking_cutoffs,
                catalog_ids=candidates.catalog,
                item_popularity=candidates.item_popularity,
                item_regions=candidates.item_regions,
            )
            for name, ordered in compared.items()
        },
        "bootstrap_vs_r1": {
            name: paired_bootstrap(
                queries, ordered, r1, cutoff=k,
                samples=config.bootstrap_samples, seed=config.random_seed,
            )
            for name, ordered in lists.items()
        },
    }


def train_with_curve(
    model: CandidateModel,
    model_config,
    train: Sequence[Interaction],
    texts: Mapping[int, str | None] | None,
    queries: Sequence[WindowQuery],
    candidates: WindowCandidates,
    config: ExperimentConfig,
    *,
    eval_every: int,
    patience: int,
    log: Callable[[str], None],
) -> tuple[dict[str, object], dict[str, tuple[int, ...]]]:
    """Fit on ``train``, scoring the validation window every ``eval_every`` epochs.

    Returns the grid row (curve, best epoch) and the rankings of the best
    epoch. Ties keep the earlier epoch.
    """

    k = config.candidate_k
    cutoffs = candidate_cutoffs(config)
    curve: list[dict[str, object]] = []
    best: dict[str, object] = {"epoch": 0, f"recall_at_{k}": -1.0}
    best_rankings: dict[str, tuple[int, ...]] = {}
    since_best = 0
    started = time.perf_counter()

    def callback(stats, fitted) -> bool:
        nonlocal best, best_rankings, since_best
        point: dict[str, object] = {
            "epoch": stats.epoch,
            **model.loss_fields(stats),
            "epoch_seconds": stats.seconds,
        }
        if stats.epoch % eval_every and stats.epoch != model_config.epochs:
            curve.append(point)
            return False
        ranked = source_rankings(fitted, queries, k)
        scores = evaluate_rankings(
            observations(queries, ranked),
            cutoffs=cutoffs,
            catalog_ids=candidates.catalog,
            item_popularity=candidates.item_popularity,
        )
        point.update(
            {
                key: value
                for key, value in scores.items()
                if key.split("_at_")[0] in ("recall", "ndcg", "catalog_coverage", "novelty")
            }
        )
        extra = model.diagnostics(fitted, model_config, queries)
        point.update(extra)
        curve.append(point)
        if float(point[f"recall_at_{k}"]) > float(best[f"recall_at_{k}"]):
            best, best_rankings, since_best = dict(point), ranked, 0
        else:
            since_best += 1
        log(
            f"{model_config.name} epoch {stats.epoch}: loss={point['loss']:.4f} "
            f"val recall@{k}={point[f'recall_at_{k}']:.4f} "
            f"(best {best[f'recall_at_{k}']:.4f} @ {best['epoch']}, {stats.seconds:.0f}s)"
            + "".join(
                f" {key}={value:.4f}" for key, value in extra.items() if isinstance(value, float)
            )
        )
        return since_best >= patience

    fitted = model.fit(model_config, train, texts, callback)
    row = {
        "name": model_config.name,
        "config": model_config.to_dict(),
        "best_epoch": best["epoch"],
        "stopped_epoch": curve[-1]["epoch"] if curve else 0,
        "early_stopped": bool(curve) and curve[-1]["epoch"] < model_config.epochs,
        "fit_seconds": round(time.perf_counter() - started, 3),
        **model.model_summary(fitted),
        "best": {key: value for key, value in best.items() if key != "epoch_seconds"},
        "curve": curve,
    }
    return row, best_rankings


def select_model_config(rows: list[dict[str, object]], config: ExperimentConfig) -> int:
    """Index of the highest best validation Recall@candidate_k; ties by NDCG@ranking_k, then order."""

    k, rk = config.candidate_k, config.ranking_k
    return max(
        range(len(rows)),
        key=lambda index: (
            rows[index]["best"][f"recall_at_{k}"],
            rows[index]["best"][f"ndcg_at_{rk}"],
            -index,
        ),
    )


def select_policy(
    model: CandidateModel,
    stage_metrics: dict[str, dict[str, object]],
    config: ExperimentConfig,
) -> str:
    """Highest validation Recall@candidate_k among the model's policies; ties by NDCG, then order."""

    k = config.candidate_k
    options = policies(model)
    return max(
        enumerate(options),
        key=lambda pair: (
            stage_metrics[pair[1]][f"recall_at_{k}"],
            stage_metrics[pair[1]][f"ndcg_at_{k}"],
            -pair[0],
        ),
    )[1]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_candidate_comparison(
    model: CandidateModel,
    interactions: Iterable[Interaction],
    *,
    project_root: Path,
    artifacts_root: Path,
    texts: Mapping[int, str | None] | None = None,
    texts_meta: dict[str, object] | None = None,
    config: ExperimentConfig | None = None,
    grid: Sequence[object] | None = None,
    eval_every: int | None = None,
    patience: int | None = None,
    baseline_run: Path | None = None,
    label: str | None = None,
    command: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
    plot: bool = True,
) -> ComparisonResult:
    config = config or ExperimentConfig()
    grid = tuple(grid or model.default_grid(config.random_seed))
    eval_every = model.default_eval_every if eval_every is None else eval_every
    patience = model.default_patience if patience is None else patience
    if not grid:
        raise ValueError(f"{model.title} grid must not be empty")
    if eval_every < 1 or patience < 1:
        raise ValueError("eval_every and patience must be positive")
    emit = log or (lambda message: print(message, flush=True))
    all_interactions = tuple(interactions)
    if not all_interactions:
        raise ValueError("Cannot run an experiment without interactions")
    if model.needs_review_texts:
        if texts is None:
            raise ValueError(f"{model.title} needs review texts")
        missing = {row.review_id for row in all_interactions} - set(texts)
        if missing:
            raise ValueError(f"No review text row for {len(missing)} interactions")
    else:
        texts = None
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = round(now - clock, 3)
        clock = now
        emit(f"[{name}] {timings[name]:.1f}s")

    source = code_manifest(project_root, allow_dirty=True)
    snapshot_path, snapshot = freeze_snapshot(all_interactions, artifacts_root / "snapshots")
    try:
        snapshot_location = str(snapshot_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        snapshot_location = str(snapshot_path)
    created_at = datetime.now(timezone.utc)
    run_id = (
        created_at.strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{str(snapshot['dataset_snapshot_id'])[:8]}"
    )
    run_dir = artifacts_root / "comparisons" / model.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    dirty_diff = str(source.pop("git_diff", ""))
    if dirty_diff:
        (run_dir / "source.diff").write_text(dirty_diff, encoding="utf-8")
        source["git_diff_artifact"] = "source.diff"

    # ---- Split and evaluation queries (identical to the pipeline) ----------
    split = build_global_temporal_split(
        all_interactions,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    t1, t2 = split.train_cutoff, split.validation_cutoff
    validation_queries, validation_new = build_window_queries(
        split.train, split.validation, config=config, phase="validation", cutoff=t1
    )
    test_history = split.train + split.validation
    test_queries, test_new = build_window_queries(
        test_history, split.test, config=config, phase="test", cutoff=t2
    )
    generator = _generator(config)
    lap("01_split_and_window_queries")

    # ---- Validation: C0-C5 with the pipeline's fixed LightGCN --------------
    graph, _ = fit_lightgcn(split.train, config.lightgcn_config)
    validation_candidates = window_candidates(
        validation_queries, build_context(split.train), generator,
        graph=graph, feature_names=None,
    )
    del graph
    lap("02_validation_c0_to_c5")

    # ---- Validation: learning curves, config and policy choice -------------
    grid_rows: list[dict[str, object]] = []
    best_rankings: list[dict[str, tuple[int, ...]]] = []
    for model_config in grid:
        row, ranked = train_with_curve(
            model, model_config, split.train, texts, validation_queries,
            validation_candidates, config,
            eval_every=eval_every, patience=patience, log=emit,
        )
        grid_rows.append(row)
        best_rankings.append(ranked)
    chosen_index = select_model_config(grid_rows, config)
    chosen = grid_rows[chosen_index]
    chosen_config = replace(grid[chosen_index], epochs=int(chosen["best_epoch"]))
    validation_stages = policy_rankings(
        model, validation_queries, validation_candidates, best_rankings[chosen_index], config
    )
    validation_metrics = evaluate_stages(
        validation_queries, validation_stages, validation_candidates, config
    )
    chosen_policy = select_policy(model, validation_metrics, config)
    references = model.references(split.train, validation_queries, test_history, test_queries)
    del best_rankings
    lap("03_validation_curves_and_selection")

    # ---- Test: refit on train + validation window, evaluate once ------------
    refit_started = time.perf_counter()
    refit = model.fit(chosen_config, test_history, texts)
    refit_summary = {
        "config": chosen_config.to_dict(),
        **model.model_summary(refit),
        "fit_seconds": round(time.perf_counter() - refit_started, 3),
        "loss_curve": [
            {"epoch": stats.epoch, **model.loss_fields(stats)} for stats in refit.history
        ],
    }
    lap("04_refit_train_plus_validation")
    graph, _ = fit_lightgcn(test_history, config.lightgcn_config)
    test_candidates = window_candidates(
        test_queries, build_context(test_history), generator, graph=graph, feature_names=None
    )
    del graph
    test_ranked = source_rankings(refit, test_queries, config.candidate_k)
    test_stages = policy_rankings(model, test_queries, test_candidates, test_ranked, config)
    test_metrics = evaluate_stages(test_queries, test_stages, test_candidates, config)
    refit_summary["test_diagnostics"] = model.diagnostics(refit, chosen_config, test_queries)
    cutoffs = sorted({20, config.candidate_k} & set(range(1, config.candidate_k + 1)))
    comparisons: dict[str, object] = {
        "vs_stage1": {
            name: {
                f"at_{cutoff}": paired_bootstrap(
                    test_queries, test_stages[name], test_stages[STAGE1],
                    cutoff=cutoff, samples=config.bootstrap_samples,
                    seed=config.random_seed,
                )
                for cutoff in cutoffs
            }
            for name in policies(model)
        },
        "positive_overlap_vs_stage1": positive_overlap(
            test_queries, test_stages[model.name], test_stages[STAGE1], config.candidate_k
        ),
    }
    if baseline_run is not None:
        comparisons["vs_baseline_run"] = compare_with_run(
            baseline_run, snapshot, split.summary(), test_queries,
            {name: test_stages[name] for name in dict.fromkeys((model.name, chosen_policy))},
            test_candidates, config,
        )
    write_jsonl(
        run_dir / "candidates_test.jsonl",
        (
            {
                "query_id": q.query_id,
                "user_id": q.user_id,
                model.name: test_stages[model.name][q.query_id],
                chosen_policy: test_stages[chosen_policy][q.query_id],
                STAGE1: test_stages[STAGE1][q.query_id],
            }
            for q in test_queries
        ),
    )
    lap("05_single_test_evaluation")

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
        "test_comparisons": comparisons,
        "references": references,
        "selection": {
            "rule": {
                "epochs": (
                    f"{eval_every} epoch마다 validation Recall@{config.candidate_k}; "
                    f"{patience}회 연속 개선 없으면 중단, 최고 epoch 사용"
                ),
                "config": (
                    f"best validation Recall@{config.candidate_k} 최대 "
                    f"(동률이면 NDCG@{config.ranking_k}, 그다음 grid 순서)"
                ),
                "policy": (
                    f"{model.title} 단독과 RRF 결합 중 validation Recall@{config.candidate_k} 최대"
                ),
            },
            "grid": grid_rows,
            "chosen_config": chosen["name"],
            "chosen_epochs": chosen["best_epoch"],
            "chosen_policy": chosen_policy,
        },
        "refit": refit_summary,
    }
    environment = environment_manifest()
    wanted = {"lightgbm", "rating-recsys", *model.packages}
    manifest = {
        "run_id": run_id,
        "kind": "candidate-comparison",
        "label": label,
        "created_at": created_at.isoformat(),
        "command": list(command) if command is not None else None,
        "config": config.to_dict(),
        "model": {
            "name": model.name,
            "title": model.title,
            "letter": model.letter,
            "grid": [item.to_dict() for item in grid],
            "eval_every": eval_every,
            "patience": patience,
            "description": list(model.description),
            "loss_description": model.loss_description,
        },
        "stage_labels": stage_labels(model),
        "fusions": {name: list(members) for name, members in fusions(model).items()},
        "baseline_run": str(baseline_run) if baseline_run else None,
        "snapshot": {**snapshot, "path": snapshot_location},
        "review_texts": texts_meta if model.needs_review_texts else None,
        "code": source,
        "environment": {
            "python": environment["python"],
            "platform": environment["platform"],
            "packages": [
                p for p in environment["packages"] if p.split("==")[0].lower() in wanted
            ],
        },
        "split": split.summary(),
        "windows": {
            "validation": _window_summary(validation_queries, validation_new, t1),
            "test": _window_summary(test_queries, test_new, t2),
        },
        "leakage_checks": {
            "validation_model_inputs_through": t1.isoformat(),
            "test_model_inputs_through": t2.isoformat(),
            "validation_context_through": t1.isoformat(),
            "test_context_through": t2.isoformat(),
            **model.leakage_checks,
            "test_evaluations": 1,
        },
        "timings_seconds": timings,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    from rating_recsys.evaluation.comparison_report import write_learning_curve, write_report

    if plot:
        write_learning_curve(run_dir, manifest, metrics)
    write_report(run_dir)
    return ComparisonResult(run_id=run_id, run_dir=run_dir, metrics=metrics, manifest=manifest)
