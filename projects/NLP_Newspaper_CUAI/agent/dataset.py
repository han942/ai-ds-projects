"""Langfuse datasets & experiment runs, seeded from ``result.csv``.

``result.csv`` (from the fine-tuning project) has columns:
    original, simplified_human, simplified_by_model

We upload ``original`` as the input and ``simplified_human`` as the expected
output, then run the agent over the dataset as a Langfuse *experiment* so each
run is comparable in the UI. With Langfuse disabled, ``run_experiment`` still
executes locally and prints aggregate scores.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from . import tracing
from .agent import NewsSimplifierAgent
from .config import Settings
from .evaluation import Evaluator

DEFAULT_DATASET = "news-simplification"


def _read_csv(csv_path: str | Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def upload_dataset(
    csv_path: str | Path = "result.csv",
    name: str = DEFAULT_DATASET,
    limit: int | None = None,
) -> int:
    """Create/populate a Langfuse dataset from the CSV. Returns item count."""
    client = tracing.get_langfuse()
    if client is None:
        raise RuntimeError(
            "Langfuse is not configured; cannot upload a dataset. Set the "
            "LANGFUSE_* env vars first."
        )

    client.create_dataset(
        name=name,
        description="Hard Korean news -> simplified for upper-elementary/middle-school readers.",
    )
    rows = _read_csv(csv_path, limit)
    for row in rows:
        client.create_dataset_item(
            dataset_name=name,
            input=row.get("original", ""),
            expected_output=row.get("simplified_human", ""),
            metadata={"baseline_finetuned": row.get("simplified_by_model", "")},
        )
    client.flush()
    return len(rows)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    agg: dict[str, float] = {}
    for axis in ("meaning_preservation", "simplicity", "age_appropriateness",
                 "fluency", "overall"):
        vals = [r["judge"].get(axis) for r in records
                if isinstance(r["judge"].get(axis), (int, float))]
        if vals:
            agg[axis] = round(statistics.mean(vals), 3)
    rouges = [r["rouge_l"] for r in records if isinstance(r.get("rouge_l"), (int, float))]
    if rouges:
        agg["rouge_l"] = round(statistics.mean(rouges), 4)
    return agg


_JUDGE_AXES = ("meaning_preservation", "simplicity", "age_appropriateness",
               "fluency", "overall")


def run_experiment(
    dataset_name: str = DEFAULT_DATASET,
    run_name: str = "agent-run",
    limit: int | None = None,
    settings: Settings | None = None,
    csv_path: str | Path = "result.csv",
) -> dict[str, Any]:
    """Run the agent + evaluator over the dataset.

    When Langfuse is configured, this uses the v4 ``run_experiment`` API so the
    run is comparable in the UI. Otherwise it iterates the local CSV and returns
    aggregate scores.
    """
    settings = settings or Settings()
    agent = NewsSimplifierAgent(settings)
    evaluator = Evaluator(settings)
    client = tracing.get_langfuse()

    records: list[dict[str, Any]] = []

    def _evaluate(original: str, simplified: str, reference: str) -> dict[str, Any]:
        # log_scores=False: in an experiment the scores come back via evaluators.
        ev = evaluator.evaluate(original, simplified, reference=reference,
                                log_scores=False)
        records.append({"judge": ev.judge, "rouge_l": ev.rouge_l})
        return ev.to_dict()

    if client is not None:
        from langfuse.experiment import Evaluation

        def task(*, item: Any, **_: Any) -> str:
            original = item.input if isinstance(item.input, str) else str(item.input)
            return agent.run(original).simplified

        def judge_evaluator(*, input: Any, output: Any,
                            expected_output: Any = None, **_: Any) -> list:
            ev = _evaluate(str(input), str(output), expected_output or "")
            evals = []
            for axis in _JUDGE_AXES:
                v = ev["judge"].get(axis)
                if isinstance(v, (int, float)):
                    evals.append(Evaluation(name=f"judge_{axis}", value=float(v),
                                            comment=ev["judge"].get("rationale")))
            if isinstance(ev.get("rouge_l"), (int, float)):
                evals.append(Evaluation(name="rouge_l", value=float(ev["rouge_l"])))
            return evals

        dataset = client.get_dataset(dataset_name)
        result = dataset.run_experiment(
            name=dataset_name,
            run_name=run_name,
            task=task,
            evaluators=[judge_evaluator],
            max_concurrency=4,
        )
        client.flush()
        return {
            "run_name": run_name,
            "n": len(records),
            "aggregate": _aggregate(records),
            "langfuse": True,
            "experiment_url": getattr(result, "dataset_run_url", None),
        }

    # Local fallback (no Langfuse)
    for row in _read_csv(csv_path, limit):
        original = row.get("original", "")
        simplified = agent.run(original).simplified
        _evaluate(original, simplified, row.get("simplified_human", ""))

    return {
        "run_name": run_name,
        "n": len(records),
        "aggregate": _aggregate(records),
        "langfuse": False,
    }
