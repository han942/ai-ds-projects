"""Run the two-stage recommender experiment and write artifacts/runs/<run_id>/.

Examples
--------
Default conditions on a frozen snapshot::

    rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl

Change conditions (every value that differs from the default is listed in
the report)::

    rating-recsys-experiment --snapshot <file> --candidate-k 200 --label k200

Without ``--snapshot`` the current DB is read and frozen to
``artifacts/snapshots/<snapshot_id>.jsonl`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.experiments.config import ExperimentConfig


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(item) for item in text.split(",") if item.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    default = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    run = parser.add_argument_group("입력과 출력")
    run.add_argument("--snapshot", type=Path, help="고정된 dataset.jsonl. 생략하면 DB를 읽는다")
    run.add_argument("--label", help="보고서와 MLflow run 이름에 붙일 짧은 이름")
    run.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    run.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="완료 후 artifacts/mlflow.db에 기록 (--no-mlflow로 끔)",
    )

    data = parser.add_argument_group("분할과 정답")
    data.add_argument("--train-fraction", type=float, default=default.train_fraction)
    data.add_argument(
        "--validation-fraction", type=float, default=default.validation_fraction
    )
    data.add_argument(
        "--relevance-high", type=float, default=default.relevance_high_threshold,
        help="이 평점 이상은 relevance 2",
    )
    data.add_argument(
        "--relevance-low", type=float, default=default.relevance_low_threshold,
        help="이 평점 이상은 relevance 1, 미만은 정답이 아님",
    )

    model = parser.add_argument_group("후보 생성과 LTR")
    model.add_argument("--candidate-k", type=int, default=default.candidate_k)
    model.add_argument("--ranking-k", type=int, default=default.ranking_k)
    model.add_argument("--rrf-constant", type=int, default=default.rrf_constant)
    model.add_argument(
        "--region-mode", choices=("with_region", "without_region"),
        default=default.region_mode,
    )
    model.add_argument("--seed", type=int, default=default.random_seed)
    model.add_argument(
        "--legacy-c3-quota", type=float, default=default.legacy_c3_quota,
        help="참고용 C3(이전 기준선)의 quota. Stage 1 결과에는 영향 없음",
    )

    graph = parser.add_argument_group("C4 LightGCN (고정 설정)")
    graph.add_argument("--lightgcn-layers", type=int, default=default.lightgcn_layers)
    graph.add_argument("--lightgcn-dimension", type=int, default=default.lightgcn_dimension)
    graph.add_argument("--lightgcn-epochs", type=int, default=default.lightgcn_epochs)
    graph.add_argument(
        "--lightgcn-regularization", type=float, default=default.lightgcn_regularization
    )
    graph.add_argument(
        "--lightgcn-learning-rate", type=float, default=default.lightgcn_learning_rate
    )
    graph.add_argument("--lightgcn-batch-size", type=int, default=default.lightgcn_batch_size)
    graph.add_argument(
        "--lightgcn-checkpoint-months", type=int,
        default=default.lightgcn_checkpoint_months,
        help="Ranker 학습 query용 LightGCN을 다시 학습하는 간격(개월, 12의 약수)",
    )

    tuning = parser.add_argument_group("Validation 선택 grid (LambdaRank)")
    tuning.add_argument("--num-leaves-grid", type=_ints, default=default.num_leaves_grid)
    tuning.add_argument(
        "--min-child-samples-grid", type=_ints, default=default.min_child_samples_grid
    )
    tuning.add_argument("--learning-rate", type=float, default=default.learning_rate)
    tuning.add_argument("--max-estimators", type=int, default=default.max_estimators)
    tuning.add_argument(
        "--early-stopping-rounds", type=int, default=default.early_stopping_rounds
    )
    tuning.add_argument(
        "--bootstrap-samples", type=int, default=default.bootstrap_samples
    )
    tuning.add_argument("--n-jobs", type=int, default=default.n_jobs)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        candidate_k=args.candidate_k,
        ranking_k=args.ranking_k,
        rrf_constant=args.rrf_constant,
        random_seed=args.seed,
        region_mode=args.region_mode,
        relevance_high_threshold=args.relevance_high,
        relevance_low_threshold=args.relevance_low,
        legacy_c3_quota=args.legacy_c3_quota,
        lightgcn_layers=args.lightgcn_layers,
        lightgcn_dimension=args.lightgcn_dimension,
        lightgcn_epochs=args.lightgcn_epochs,
        lightgcn_regularization=args.lightgcn_regularization,
        lightgcn_learning_rate=args.lightgcn_learning_rate,
        lightgcn_batch_size=args.lightgcn_batch_size,
        lightgcn_checkpoint_months=args.lightgcn_checkpoint_months,
        num_leaves_grid=tuple(args.num_leaves_grid),
        min_child_samples_grid=tuple(args.min_child_samples_grid),
        learning_rate=args.learning_rate,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        bootstrap_samples=args.bootstrap_samples,
        n_jobs=args.n_jobs,
    )


def _load_interactions(snapshot: Path | None):
    from rating_recsys.experiments.snapshot import load_snapshot

    if snapshot is not None:
        return load_snapshot(snapshot)
    from rating_recsys.config import get_settings
    from rating_recsys.datasets.repository import InteractionRepository
    from rating_recsys.db.connection import connect

    settings = get_settings(require_database=True, require_user_hash_salt=False)
    with connect(settings.database_url) as connection:
        return InteractionRepository(connection).fetch_first_interactions()


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = build_parser().parse_args(argv)
    # Validate every condition before reading data or training.
    config = config_from_args(args)
    from rating_recsys.experiments.artifacts import write_json
    from rating_recsys.experiments.pipeline import run_experiment

    result = run_experiment(
        _load_interactions(args.snapshot),
        project_root=PROJECT_ROOT,
        artifacts_root=args.artifacts_dir,
        config=config,
        label=args.label,
    )
    summary: dict[str, object] = {
        "run_id": result.run_id,
        "report": str(result.run_dir / "report.md"),
    }
    if args.mlflow:
        from rating_recsys.observability.tracking import log_run

        tracking = log_run(result.run_dir, artifacts_root=args.artifacts_dir)
        result.manifest["mlflow"] = tracking
        write_json(result.run_dir / "manifest.json", result.manifest)
        summary["mlflow_run_id"] = tracking["run_id"]
    test = result.metrics["test"]
    k, ck = config.ranking_k, config.candidate_k
    summary["test"] = {
        f"c5_recall_at_{ck}": test["c5_c1_lightgcn_rrf"][f"recall_at_{ck}"],
        f"c3_recall_at_{ck}": test["c3_rrf_union"][f"recall_at_{ck}"],
        f"r0_ndcg_at_{k}": test["r0_candidate_order"][f"ndcg_at_{k}"],
        f"r1_ndcg_at_{k}": test["r1_lambdarank"][f"ndcg_at_{k}"],
        f"r1_recall_at_{k}": test["r1_lambdarank"][f"recall_at_{k}"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {summary['report']}", file=sys.stderr)
    return summary


def console_main() -> int:
    """Console-script entry point. ``sys.exit(main())`` would treat the summary
    dict as an error message and exit with status 1."""

    main()
    return 0


if __name__ == "__main__":
    console_main()
