"""Run a review-only OpenRouter embedding candidate comparison.

`--dry-run` is entirely local: it reports cutoff-safe profile counts and cache
misses without calling the API. The full run uses the shared C5 comparison
runner and the same validation/test split as the baseline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.candidate_models import ReviewEmbeddingsCandidate
from rating_recsys.experiments.compare_cli import build_parser, configs_from_args, load_or_fetch_texts
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import review_texts_path
from rating_recsys.retrieval.review_embeddings import create_embedding_cache, ProfileFormatter, profile_api_inputs


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    model = ReviewEmbeddingsCandidate()
    parser = build_parser(model)
    parser.prog = "rating-recsys-review-embeddings"
    parser.set_defaults(plot=False)
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 프로필과 캐시 누락량만 집계")
    args = parser.parse_args(argv)
    if args.max_epochs != 1:
        parser.error("review embeddings require --max-epochs 1")
    config, grid = configs_from_args(model, args)
    from rating_recsys.experiments.cli import _load_interactions

    interactions = _load_interactions(args.snapshot)
    texts_path = args.review_texts or review_texts_path(args.snapshot)
    texts, texts_meta = load_or_fetch_texts(texts_path, interactions)
    try:
        texts_meta["path"] = str(texts_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        texts_meta["path"] = str(texts_path)
    if args.dry_run:
        split = build_global_temporal_split(
            interactions, train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
        )
        phases = (
            ("validation", split.train, split.validation, split.train_cutoff),
            ("test", split.train + split.validation, split.test, split.validation_cutoff),
        )
        output: dict[str, object] = {"model": grid[0].to_dict(), "phases": {}}
        cache_path = Path(grid[0].cache_path)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        with create_embedding_cache(cache_path, grid[0]) as cache:
            all_inputs = []
            for phase, history, window, cutoff in phases:
                queries, _ = build_window_queries(
                    history, window, config=config, phase=phase, cutoff=cutoff,
                )
                formatter = ProfileFormatter(grid[0], cache_path.parent / "tokenizers")
                users, items, liked, docs = profile_api_inputs(
                    history, texts, [q.user_id for q in queries], grid[0], formatter,
                )
                all_inputs.extend(docs)
                misses = cache.missing_count(docs)
                output["phases"][phase] = {
                    "cutoff": cutoff.isoformat(), "queries": len(queries),
                    "users_with_profiles": len(liked) if grid[0].user_profile == "liked_items" else len(users), "restaurants_with_profiles": len(items),
                    "documents": len(docs), "unique_cache_misses": misses,
                    "estimated_requests": math.ceil(misses / grid[0].batch_size),
                    "document_characters": sum(map(len, docs)),
                    "preprocessing": formatter.metadata(),
                }
            output["total_unique_cache_misses"] = cache.missing_count(all_inputs)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output

    from rating_recsys.experiments.comparison import run_candidate_comparison

    result = run_candidate_comparison(
        model, interactions, project_root=PROJECT_ROOT,
        artifacts_root=args.artifacts_dir, texts=texts, texts_meta=texts_meta,
        config=config, grid=grid, eval_every=args.eval_every, patience=args.patience,
        baseline_run=args.baseline_run, label=args.label,
        command=["rating-recsys-review-embeddings", *(argv if argv is not None else sys.argv[1:])],
        log=lambda message: print(message, file=sys.stderr, flush=True),
        plot=args.plot, use_cache=not args.no_cache, rebuild_cache=args.rebuild_cache,
    )
    k = config.candidate_k
    selection = result.metrics["selection"]
    test = result.metrics["test"]
    output = {
        "report": str(result.run_dir / "report.md"),
        "chosen_policy": selection["chosen_policy"],
        f"c5_recall_at_{k}": test["c5_c1_lightgcn_rrf"][f"recall_at_{k}"],
        f"embedding_recall_at_{k}": test[model.name][f"recall_at_{k}"],
        f"chosen_policy_recall_at_{k}": test[selection["chosen_policy"]][f"recall_at_{k}"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


def console_main() -> int:
    main()
    return 0


if __name__ == "__main__":
    console_main()
