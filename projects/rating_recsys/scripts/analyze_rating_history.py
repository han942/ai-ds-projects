"""Train-only evidence for the minimum history used by satisfaction labels.

Run from the project directory with its Python environment. Writes aggregate
evidence, never future ratings or individual user histories. This is a Python
analysis companion; it needs no notebook kernel or database connection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_training_windows, global_interaction_key
from rating_recsys.experiments.snapshot import dataset_digest, load_snapshot


def analyze(snapshot: Path) -> dict:
    interactions = load_snapshot(snapshot)
    split = build_global_temporal_split(interactions)
    train = tuple(sorted(split.train, key=global_interaction_key))
    histories = defaultdict(list)
    for row in train:
        histories[row.user_id].append(row.rating)
    counts = np.array([len(values) for values in histories.values()])
    # Pooled within-user sample variance. Between-user mean differences are
    # deliberately excluded from the estimate of a user's mean uncertainty.
    squared = math.fsum(
        math.fsum((rating - np.mean(values)) ** 2 for rating in values)
        for values in histories.values() if len(values) > 1
    )
    degrees = sum(len(values) - 1 for values in histories.values() if len(values) > 1)
    pooled_sd = math.sqrt(squared / degrees)
    config = ExperimentConfig(ranker_training_mode="window")
    windows = build_training_windows(train, config=config, through=split.train_cutoff)
    queries = [query for _, items in windows for query in items]
    history_lengths = np.array([len(query.history) for query in queries])
    # Same users for every n; otherwise apparent stability could simply come
    # from comparing different populations. Later train ratings are used only
    # for this drift diagnostic, never for a query's labeling profile.
    common = [np.array(values) for values in histories.values() if len(values) >= 40]
    comparisons = []
    for n in (3, 5, 8, 10, 12, 15, 20):
        samples = [np.array(values[:n]) for values in histories.values() if len(values) >= n]
        half_widths = np.array([1.96 * np.std(values, ddof=1) / np.sqrt(n) for values in samples])
        changes = np.array([abs(np.mean(values[:n]) - np.mean(values[n:2*n])) for values in common])
        comparisons.append({
            "minimum_history": n,
            "eligible_train_users": len(samples),
            "eligible_train_user_fraction": len(samples) / len(histories),
            "pooled_approx_95pct_half_width": 1.96 * pooled_sd / math.sqrt(n),
            "per_user_approx_half_width_median": float(np.median(half_widths)),
            "per_user_approx_half_width_p90": float(np.percentile(half_widths, 90)),
            "zero_variance_first_n_fraction": float(np.mean([np.ptp(values) == 0 for values in samples])),
            "common_cohort_first_n_next_n_difference_median": float(np.median(changes)),
            "common_cohort_first_n_next_n_difference_p90": float(np.percentile(changes, 90)),
            "personalizable_train_window_queries": int(np.sum(history_lengths >= n)),
            "personalizable_train_window_fraction": float(np.mean(history_lengths >= n)),
        })
    assert len({(r.user_id, r.restaurant_id) for r in train}) == len(train)
    assert all(r.event_date <= split.train_cutoff for r in train)
    assert all(r.event_date <= q.cutoff for q in queries for r in q.history)
    return {
        "snapshot": str(snapshot), "dataset_snapshot_id": dataset_digest(interactions),
        "analysis_scope": "train through T1 only; no validation/test ratings used to choose the rule",
        "train_through": split.train_cutoff.isoformat(),
        "train_interactions": len(train), "train_users": len(histories),
        "history_count_percentiles": dict(zip(("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"),
                                             np.percentile(counts, (0, 25, 50, 75, 90, 95, 99, 100)).tolist())),
        "rating_counts": dict(Counter(row.rating for row in train)),
        "date_precision_counts": dict(Counter(row.reviewed_at_precision for row in train)),
        "pooled_within_user_sd": pooled_sd,
        "pooled_variance_degrees_of_freedom": degrees,
        "train_window_queries": len(queries),
        "common_drift_cohort_users": len(common), "common_drift_cohort_min_train_visits": 40,
        "comparisons": comparisons,
        "decision": {
            "minimum_history": 10,
            "reason": "Practical precision/coverage compromise: pooled half-width about 0.40 stars, "
                      "90th percentile per-user half-width about 0.57 stars, 42.8% train window coverage. "
                      "5/8 histories are noisier; 15 histories reduces coverage to 29.1%.",
            "rule": "strong threshold = clip(4 + 0.5*(past user mean - 4), 3.5, 4.5); "
                    "below 10 past visits use 4; weak threshold remains 3",
            "performance_tuned": False,
        },
        "limitations": [
            "1.96*s/sqrt(n) is an approximate precision diagnostic, not a guaranteed confidence interval; "
            "bounded ratings, constant samples, dependence, and preference drift limit its interpretation.",
            "The same-cohort diagnostic includes only 273 active train users with at least 40 visits; "
            "it does not establish stability for sparse users.",
            "Observed visits are selected behavior, not exposure or randomized satisfaction observations.",
            "Displayed usernames may merge people; event dates may be approximate.",
            "10 is a practical task definition, not a statistically optimal or performance-maximizing threshold.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/snapshots/e7896add5b4b5939.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/analyses/rating_history/analysis.json"))
    args = parser.parse_args()
    evidence = analyze(args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "decision": evidence["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
