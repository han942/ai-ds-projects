"""Compare one candidate model with Stage 1 (C5) on the pipeline's split.

Writes ``artifacts/comparisons/<model>/<run_id>/`` (report.md,
learning_curve.png, manifest.json, metrics.json, candidates_test.jsonl).
Models are listed in ``experiments.candidate_models.CANDIDATE_MODELS``; each
adds its own grid flags (``rating-recsys-compare <model> --help``).

Examples
--------
LightGCN grid (layers x L2), compared with a pipeline run's R1 Top-K::

    rating-recsys-compare lightgcn \\
      --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \\
      --baseline-run artifacts/runs/<run_id>

DeepCoNN with one config. Review text is read from ``<snapshot>.reviews.jsonl``
next to the snapshot; when missing it is fetched once from the DB (read only)
for exactly the snapshot's review ids::

    rating-recsys-compare deepconn --snapshot <file> --variants bpr:linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.experiments.candidate_models import CANDIDATE_MODELS, CandidateModel
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.cli import add_satisfaction_arguments, satisfaction_config_from_args
from rating_recsys.experiments.prepared import add_cache_arguments


def build_parser(model: CandidateModel) -> argparse.ArgumentParser:
    default = ExperimentConfig()
    parser = argparse.ArgumentParser(
        prog=f"rating-recsys-compare {model.name}",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run = parser.add_argument_group("입력과 출력")
    run.add_argument("--snapshot", type=Path, required=True, help="고정된 snapshot jsonl")
    if model.needs_review_texts:
        run.add_argument(
            "--review-texts", type=Path,
            help="리뷰 본문 파일. 기본값은 snapshot 옆의 <snapshot>.reviews.jsonl "
            "(없으면 DB에서 한 번 읽어 만든다)",
        )
    run.add_argument(
        "--baseline-run", type=Path,
        help="같은 snapshot의 rating-recsys-experiment run 폴더. 주면 R1 Top-K와 비교",
    )
    run.add_argument("--label", help="보고서에 붙일 짧은 이름")
    run.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    run.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True,
        help="learning_curve.png 생성 (matplotlib 필요)",
    )

    data = parser.add_argument_group("분할과 후보 (pipeline과 같은 의미)")
    data.add_argument("--train-fraction", type=float, default=default.train_fraction)
    data.add_argument("--validation-fraction", type=float, default=default.validation_fraction)
    data.add_argument("--candidate-k", type=int, default=default.candidate_k)
    data.add_argument("--ranking-k", type=int, default=default.ranking_k)
    data.add_argument("--rrf-constant", type=int, default=default.rrf_constant)
    data.add_argument("--bootstrap-samples", type=int, default=default.bootstrap_samples)
    data.add_argument("--seed", type=int, default=default.random_seed)
    add_satisfaction_arguments(data)

    training = parser.add_argument_group("학습 곡선과 조기 종료")
    training.add_argument("--max-epochs", type=int, default=model.default_max_epochs)
    training.add_argument(
        "--eval-every", type=int, default=model.default_eval_every,
        help="validation 평가 간격(epoch)",
    )
    training.add_argument(
        "--patience", type=int, default=model.default_patience,
        help="연속으로 개선이 없으면 중단할 평가 횟수",
    )
    model.add_arguments(parser.add_argument_group(f"{model.title} grid"))
    add_cache_arguments(parser)
    parser.set_defaults(**getattr(model, "experiment_defaults", {}))
    return parser


def configs_from_args(
    model: CandidateModel, args: argparse.Namespace
) -> tuple[ExperimentConfig, tuple[object, ...]]:
    config = ExperimentConfig(
        ranker_training_mode=getattr(model, "ranker_training_mode", "prefix"),
        **satisfaction_config_from_args(args),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        candidate_k=args.candidate_k,
        ranking_k=args.ranking_k,
        rrf_constant=args.rrf_constant,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
    )
    grid = model.grid_from_args(args)
    if not grid:
        raise ValueError(f"{model.title} grid must not be empty")
    return config, grid


def load_or_fetch_texts(path: Path, interactions) -> tuple[dict[int, str | None], dict[str, object]]:
    from rating_recsys.experiments.snapshot import load_review_texts, write_review_texts

    if not path.exists():
        from rating_recsys.config import get_settings
        from rating_recsys.datasets.repository import InteractionRepository
        from rating_recsys.db.connection import connect

        settings = get_settings(require_database=True, require_user_hash_salt=False)
        with connect(settings.database_url) as connection:
            texts = InteractionRepository(connection).fetch_review_texts(
                [item.review_id for item in interactions]
            )
        write_review_texts(texts, path)
    return load_review_texts(path, interactions)


def _model_from_argv(argv: list[str]) -> CandidateModel:
    top = argparse.ArgumentParser(
        prog="rating-recsys-compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    top.add_argument("model", choices=sorted(CANDIDATE_MODELS))
    return CANDIDATE_MODELS[top.parse_args(argv[:1]).model]


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    argv = list(sys.argv[1:] if argv is None else argv)
    model = _model_from_argv(argv)
    args = build_parser(model).parse_args(argv[1:])
    config, grid = configs_from_args(model, args)
    if args.baseline_run is not None and not (args.baseline_run / "manifest.json").exists():
        raise SystemExit(f"--baseline-run {args.baseline_run} has no manifest.json")
    from rating_recsys.experiments.cli import _load_interactions
    from rating_recsys.experiments.comparison import run_candidate_comparison

    interactions = _load_interactions(args.snapshot)
    texts = texts_meta = None
    if model.needs_review_texts:
        from rating_recsys.experiments.snapshot import review_texts_path

        texts_path = args.review_texts or review_texts_path(args.snapshot)
        texts, texts_meta = load_or_fetch_texts(texts_path, interactions)
        try:
            texts_meta["path"] = str(texts_path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            texts_meta["path"] = str(texts_path)
    result = run_candidate_comparison(
        model,
        interactions,
        project_root=PROJECT_ROOT,
        artifacts_root=args.artifacts_dir,
        texts=texts,
        texts_meta=texts_meta,
        config=config,
        grid=grid,
        eval_every=args.eval_every,
        patience=args.patience,
        baseline_run=args.baseline_run,
        label=args.label,
        command=argv,
        log=lambda message: print(message, file=sys.stderr, flush=True),
        plot=args.plot,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_cache,
    )
    k = config.candidate_k
    selection = result.metrics["selection"]
    test = result.metrics["test"]
    summary = {
        "run_id": result.run_id,
        "report": str(result.run_dir / "report.md"),
        "chosen_config": selection["chosen_config"],
        "chosen_epochs": selection["chosen_epochs"],
        "chosen_policy": selection["chosen_policy"],
        "test": {
            f"c5_recall_at_{k}": test["c5_c1_lightgcn_rrf"][f"recall_at_{k}"],
            f"{model.name}_recall_at_{k}": test[model.name][f"recall_at_{k}"],
            f"policy_recall_at_{k}": test[selection["chosen_policy"]][f"recall_at_{k}"],
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def console_main() -> int:
    """Console-script entry point; ``main`` returns the summary for callers."""

    main()
    return 0


if __name__ == "__main__":
    console_main()
