"""Backfill the auxiliary pair diagnostic from saved target ranks, no training.

Run with: python -m rating_recsys.evaluation.pair_diagnostics --run-dir <run>
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from rating_recsys.evaluation import metrics as metric_module
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_observed_pairs
from rating_recsys.evaluation.report import observed_pair_lines
from rating_recsys.experiments.artifacts import read_json, read_jsonl, write_json


def backfill_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path)
    results = {}
    input_hashes = {}
    for phase in ("validation", "test"):
        queries_path = run_dir / f"queries_{phase}.jsonl"
        ranks_path = run_dir / f"target_diagnostics_{phase}.jsonl"
        for path in (queries_path, ranks_path):
            # Fail before any writes if saved artifacts are incomplete.
            input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        queries = read_jsonl(queries_path)
        ranks = read_jsonl(ranks_path)
        indexed = {row["query_id"]: row for row in ranks}
        if (len(indexed) != len(ranks) or len({q["query_id"] for q in queries}) != len(queries)
                or set(indexed) != {q["query_id"] for q in queries}):
            raise ValueError(f"{phase}: saved query/rank IDs do not match uniquely")
        observations = {"r0_candidate_order": [], "r1_lambdarank": []}
        for query in queries:
            query_id = query["query_id"]
            grades = {int(row["restaurant_id"]): row["relevance"] for row in query["window"]}
            targets = indexed[query_id]["targets"]
            target_ids = [int(row["restaurant_id"]) for row in targets]
            if len(set(target_ids)) != len(target_ids) or set(target_ids) != set(grades):
                raise ValueError(f"{query_id}: saved target IDs do not match grades uniquely")
            for target in targets:
                if (target["candidate_rank"] is None) != (target["final_rank"] is None):
                    raise ValueError(f"{query_id}: candidate/final pools differ")
            for stage, field in (("r0_candidate_order", "candidate_rank"),
                                 ("r1_lambdarank", "final_rank")):
                present = [row for row in targets if row[field] is not None]
                positions = [row[field] for row in present]
                if (len(set(positions)) != len(positions)
                        or any(type(rank) is not int or rank < 1 for rank in positions)):
                    raise ValueError(f"{query_id}: saved ranks must be positive and unique")
                ordered = tuple(int(row["restaurant_id"]) for row in sorted(present, key=lambda r: r[field]))
                # Other candidates have no actual grades and cannot affect pair order.
                observations[stage].append(RankingObservation(
                    query_id, query["user_id"], grades, ordered,
                ))
        results[phase] = {stage: evaluate_observed_pairs(rows) for stage, rows in observations.items()}
        existing = metrics[phase].get("observed_pair_diagnostics")
        if existing is not None and existing != results[phase]:
            raise ValueError(f"{phase}: saved diagnostic differs from artifact recomputation")

    summary = {
        "schema_version": "observed-graded-pair-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "diagnostic_only": True,
        "scope": "same observed different-grade pairs within complete C5 candidate pool",
        "aggregation": "mean of per-query accuracies; same-grade/unobserved/missing pairs excluded",
        "input_sha256": input_hashes,
        "implementation_sha256": {
            "metrics.py": hashlib.sha256(Path(metric_module.__file__).read_bytes()).hexdigest(),
            "pair_diagnostics.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "phases": results,
    }
    report_path = run_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    heading = "### 관측 만족도 쌍 순서 정확도 · 보조 진단"
    if heading not in report:
        section = "\n".join(observed_pair_lines({"observed_pair_diagnostics": results["test"]}))
        section += "\n\n기존 저장 순위로 사후 계산했으며 재학습·재튜닝하지 않았다. "
        section += "Validation/test 계산과 입력 hash는 `observed_pair_diagnostics.json`에 기록했다.\n"
        marker = "\n## 6. "
        if marker in report:
            report = report.replace(marker, section + marker, 1)
        else:
            report = report.rstrip() + "\n" + section
    for phase, values in results.items():
        metrics[phase]["observed_pair_diagnostics"] = values
    write_json(run_dir / "observed_pair_diagnostics.json", summary)
    write_json(metrics_path, metrics)
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = backfill_run(args.run_dir)
    for phase, diagnostics in summary["phases"].items():
        print(phase, diagnostics)


if __name__ == "__main__":
    main()
