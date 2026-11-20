"""Command-line interface for the news-simplification agent.

Examples
--------
    # list currently available :free models on OpenRouter
    python -m agent.cli list-free-models --limit 25

    # run C0 single-shot baseline on stratified splits (with checkpointing)
    python -m agent.cli c0-run --model qwen/qwen3-32b:free --limit 10

    # simplify a single article with the LangGraph agent
    python -m agent.cli simplify --text "인천 청라시티타워 '운명의 날'…내일 추진 여부 결정"
    python -m agent.cli simplify --file article.txt

    # judge a pair
    python -m agent.cli eval --original "..." --simplified "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import tracing
from .agent import NewsSimplifierAgent
from .config import load_settings
from .evaluation import Evaluator


def _print_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _cmd_seed_prompts(args: argparse.Namespace) -> int:
    from .prompts import seed_prompts

    names = seed_prompts(label=args.label)
    print(f"Seeded {len(names)} prompts to Langfuse: {', '.join(names)}")
    return 0


def _cmd_simplify(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.model:
        settings.model = args.model.strip()
    settings.require_llm(require_model=True)
    text = args.text
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    if not text:
        print("Provide --text or --file", file=sys.stderr)
        return 2

    agent = NewsSimplifierAgent(settings)
    result = agent.run(text)

    if args.json:
        _print_json(result.to_dict())
    else:
        print("\n=== 원문 ===\n" + result.original)
        print("\n=== 쉬운 버전 ===\n" + result.simplified)
        print(f"\n(iterations={result.iterations}, approved={result.approved})")

    if args.evaluate:
        ev = Evaluator(settings).evaluate(result.original, result.simplified)
        print("\n=== 평가 ===")
        _print_json(ev.to_dict())

    tracing.flush()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    settings = load_settings()
    settings.require_llm(require_model=False)
    if not settings.resolved_judge_model:
        raise RuntimeError(
            "No judge model configured. Set JUDGE_MODEL or OPENROUTER_MODEL in .env."
        )
    ev = Evaluator(settings).evaluate(
        args.original, args.simplified, reference=args.reference
    )
    _print_json(ev.to_dict())
    tracing.flush()
    return 0


def _cmd_upload_dataset(args: argparse.Namespace) -> int:
    from .dataset import upload_dataset

    n = upload_dataset(csv_path=args.csv, name=args.name, limit=args.limit)
    print(f"Uploaded {n} items to Langfuse dataset '{args.name}'.")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from .dataset import run_experiment

    settings = load_settings()
    settings.require_llm(require_model=True)
    summary = run_experiment(
        dataset_name=args.name,
        run_name=args.run_name,
        limit=args.limit,
        settings=settings,
        csv_path=args.csv,
    )
    _print_json(summary)
    tracing.flush()
    return 0


def _cmd_list_free_models(args: argparse.Namespace) -> int:
    from .c0 import list_openrouter_free_models

    settings = load_settings()
    models = list_openrouter_free_models(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    if args.filter:
        needle = args.filter.lower()
        models = [
            m
            for m in models
            if needle in m["id"].lower() or needle in str(m["name"]).lower()
        ]
    models = models[: args.limit]
    _print_json(models)
    return 0


def _cmd_c0_run(args: argparse.Namespace) -> int:
    from .c0 import run_c0_experiment

    settings = load_settings()
    summary = run_c0_experiment(
        settings=settings,
        model=args.model,
        csv_path=args.csv,
        split_file=args.split_file,
        split_names=tuple(args.splits),
        out_dir=args.out_dir,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        min_interval=args.min_interval,
        max_retries=args.max_retries,
    )
    _print_json(summary)
    tracing.flush()
    return 0 if summary["counts"]["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed-prompts", help="Push default prompts to Langfuse")
    p.add_argument("--label", default="production")
    p.set_defaults(func=_cmd_seed_prompts)

    p = sub.add_parser("simplify", help="Simplify one article")
    p.add_argument("--text", default="")
    p.add_argument("--file")
    p.add_argument("--model", help="Override model ID")
    p.add_argument("--json", action="store_true", help="Emit JSON")
    p.add_argument("--evaluate", action="store_true", help="Also run the judge")
    p.set_defaults(func=_cmd_simplify)

    p = sub.add_parser("eval", help="Judge an original/simplified pair")
    p.add_argument("--original", required=True)
    p.add_argument("--simplified", required=True)
    p.add_argument("--reference", help="Human reference for morpheme ROUGE-L")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("upload-dataset", help="Upload result.csv as a Langfuse dataset")
    p.add_argument("--csv", default="result.csv")
    p.add_argument("--name", default="news-simplification")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_upload_dataset)

    p = sub.add_parser("experiment", help="Run the agent over the dataset")
    p.add_argument("--name", default="news-simplification", help="Dataset name")
    p.add_argument("--run-name", default="agent-run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--csv", default="result.csv", help="Fallback CSV if Langfuse is off")
    p.set_defaults(func=_cmd_experiment)

    p = sub.add_parser(
        "list-free-models", help="List currently free models from OpenRouter"
    )
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--filter", help="Case-insensitive substring filter on model id/name")
    p.set_defaults(func=_cmd_list_free_models)

    p = sub.add_parser(
        "c0-run", help="Run checkpointed C0 single-shot baseline over stratified splits"
    )
    p.add_argument("--model", help="Override OPENROUTER_MODEL / OPENAI_MODEL")
    p.add_argument("--csv", type=Path, default=Path("result.csv"))
    p.add_argument(
        "--split-file",
        type=Path,
        default=Path("evaluation/splits/stratified_v1.json"),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["eval", "stress"],
        help="Split names to run (union deduplicated)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to evaluation/runs/c0_<model_slug>)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--min-interval", type=float, default=1.5)
    p.add_argument("--max-retries", type=int, default=5)
    p.set_defaults(func=_cmd_c0_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
