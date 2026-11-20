"""Deterministic stratified train/dev/eval/stress splits for ``result.csv``.

Groups duplicate ``original`` rows so the same source text never appears twice
within a split or leaks between ``dev`` and ``eval``. Stratifies by source-text
character length tertiles (short / medium / long) crossed with number presence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evaluation.korean import numbers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ("original", "simplified_human", "simplified_by_model")
# Documented error/diagnostic examples from evaluation/REPORT.md.
STRESS_INDICES: tuple[int, ...] = (0, 1, 5, 24, 25, 65, 100, 167, 1285, 2008)


def _proportional_counts(stratum_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Allocate ``total`` items across strata using largest-remainder rounding."""
    pop_total = sum(stratum_sizes.values())
    if total < 0 or total > pop_total:
        raise ValueError(f"Requested {total} samples from pool of size {pop_total}.")
    if total == 0:
        return {k: 0 for k in stratum_sizes}

    exact = {k: (v * total) / pop_total for k, v in stratum_sizes.items()}
    counts = {k: min(stratum_sizes[k], int(math.floor(exact[k]))) for k in stratum_sizes}
    remainder = total - sum(counts.values())
    if remainder > 0:
        order = sorted(
            stratum_sizes.keys(),
            key=lambda k: (-(exact[k] - counts[k]), k),
        )
        for k in order:
            if remainder == 0:
                break
            if counts[k] < stratum_sizes[k]:
                counts[k] += 1
                remainder -= 1
    return counts


def _sample_stratified(
    pool_indices: list[int],
    strata: list[str],
    total: int,
    seed: int,
) -> list[int]:
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for idx in sorted(pool_indices):
        by_stratum[strata[idx]].append(idx)

    targets = _proportional_counts(
        {k: len(v) for k, v in by_stratum.items()}, total
    )
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for k in sorted(by_stratum.keys()):
        n_pick = targets[k]
        if n_pick > 0:
            choices = rng.choice(by_stratum[k], size=n_pick, replace=False).tolist()
            picked.extend(int(i) for i in choices)
    return sorted(picked)


def _count_strata(strata: list[str], indices: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for idx in indices:
        counts[strata[idx]] += 1
    return dict(sorted(counts.items()))


def build_splits(
    csv_path: Path = PROJECT_ROOT / "result.csv",
    dev_n: int = 100,
    eval_n: int = 150,
    dev_seed: int = 42,
    eval_seed: int = 0,
) -> dict[str, Any]:
    """Build deterministic ``eval``, ``dev``, and ``stress`` splits."""
    csv_path = Path(csv_path)
    source_csv_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    df = df[list(REQUIRED_COLUMNS)].astype(str)

    n_rows = len(df)
    stress_indices = [i for i in STRESS_INDICES if 0 <= i < n_rows]
    if len(stress_indices) != len(STRESS_INDICES):
        raise ValueError(
            f"CSV has {n_rows} rows, cannot index all STRESS_INDICES={STRESS_INDICES}"
        )

    originals = df["original"].tolist()
    by_original: dict[str, list[int]] = defaultdict(list)
    for idx, orig in enumerate(originals):
        by_original[orig].append(idx)
    duplicate_groups = [idxs for idxs in by_original.values() if len(idxs) > 1]
    unique_pool = sorted(idxs[0] for idxs in by_original.values())

    lengths = df["original"].str.len()
    tertile = pd.qcut(lengths, 3, labels=["short", "medium", "long"])
    tertile_bounds = {
        "short_max": int(lengths[tertile == "short"].max()),
        "medium_max": int(lengths[tertile == "medium"].max()),
        "long_min": int(lengths[tertile == "long"].min()),
    }

    strata = [
        f"{tertile.iloc[i]}_{'num' if numbers(orig) else 'nonum'}"
        for i, orig in enumerate(originals)
    ]

    eval_indices = _sample_stratified(unique_pool, strata, eval_n, eval_seed)
    excluded_originals = {originals[i] for i in eval_indices} | {
        originals[i] for i in stress_indices
    }
    excluded_indices = set(eval_indices) | set(stress_indices)
    dev_pool = [
        i
        for i in unique_pool
        if originals[i] not in excluded_originals and i not in excluded_indices
    ]
    dev_indices = _sample_stratified(dev_pool, strata, dev_n, dev_seed)

    return {
        "version": "stratified_v1",
        "source_csv": csv_path.name,
        "source_csv_sha256": source_csv_sha256,
        "total_rows": n_rows,
        "unique_sources": len(by_original),
        "duplicate_groups": duplicate_groups,
        "tertile_bounds": tertile_bounds,
        "seeds": {"eval": eval_seed, "dev": dev_seed},
        "counts": {
            "eval": len(eval_indices),
            "dev": len(dev_indices),
            "stress": len(stress_indices),
        },
        "strata_counts": {
            "population": _count_strata(strata, range(n_rows)),
            "unique_pool": _count_strata(strata, unique_pool),
            "eval": _count_strata(strata, eval_indices),
            "dev": _count_strata(strata, dev_indices),
            "stress": _count_strata(strata, stress_indices),
        },
        "overlap_eval_stress": sorted(set(eval_indices) & set(stress_indices)),
        "splits": {
            "eval": eval_indices,
            "dev": dev_indices,
            "stress": stress_indices,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "result.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "splits" / "stratified_v1.json",
    )
    parser.add_argument("--dev-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=150)
    parser.add_argument("--dev-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = build_splits(
        csv_path=args.csv,
        dev_n=args.dev_n,
        eval_n=args.eval_n,
        dev_seed=args.dev_seed,
        eval_seed=args.eval_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {args.out}: eval={payload['counts']['eval']}, "
        f"dev={payload['counts']['dev']}, stress={payload['counts']['stress']}, "
        f"tertiles={payload['tertile_bounds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
