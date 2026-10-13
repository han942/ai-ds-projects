"""Cutoff-only sparsity diagnostics on fixed, original recommendation ranks."""
from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

import numpy as np

from rating_recsys.evaluation.metrics import RankingObservation, query_scores

USER_BINS = ("1-4", "5-9", "10-19", "20+")
ITEM_BINS = ("0-unseen", *USER_BINS)


def count_bin(count: int) -> str:
    if count < 0:
        raise ValueError("History counts cannot be negative")
    return "0-unseen" if count == 0 else "1-4" if count < 5 else "5-9" if count < 10 else "10-19" if count < 20 else "20+"


def validate_saved_rows(queries, rows, arms: Sequence[str]):
    """Cross-check identities and complete graded truth; never infer missing keys."""
    expected = {q.query_id: q for q in queries}
    if len(expected) != len(queries) or len({q.user_id for q in queries}) != len(queries):
        raise ValueError("Segment evaluation requires one unique query per user")
    if len(rows) != len(expected) or len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("Saved rows have missing or duplicate queries")
    ranks = {arm: {} for arm in arms}
    for row in rows:
        q = expected.get(row["query_id"])
        if q is None or row["user_id"] != q.user_id:
            raise ValueError("Saved query/user identity differs")
        truth = {int(k): v for k, v in row["relevance_by_item"].items()}
        if truth != q.relevance_by_item:
            raise ValueError("Saved relevance differs from rebuilt full window truth")
        if set(row) != {"query_id", "user_id", "relevance_by_item", *arms}:
            raise ValueError("Saved arm fields differ from the declared arms")
        for arm in arms:
            ranks[arm][q.query_id] = tuple(row[arm])
    return ranks


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _paired(values, reference, indices):
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if not len(delta):
        return {"mean_difference": None, "bootstrap_95pct": None, "paired_users": 0}
    means = delta[indices].mean(axis=1)
    return {"mean_difference": float(delta.mean()), "bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
            "paired_users": len(delta)}


def analyze_segments(queries, rankings_by_arm: Mapping[str, Mapping[str, Sequence[int]]],
                     item_popularity: Mapping[int, int], *, baseline="c5", samples=2000, seed=42):
    """Return JSON-ready macro user metrics, item truth recall and exposures.

    Item recall averages users with positives in that bin. DCG contribution
    uses the *whole query's* ideal DCG@10 and averages all relevant users;
    contributions sum to overall NDCG@10. Lists are never filtered/re-ranked.
    Item paired CIs resample users, preserving all positives from each user.
    """
    queries = tuple(queries)
    if samples < 1 or baseline not in rankings_by_arm:
        raise ValueError("A baseline and positive bootstrap sample count are required")
    ids = {q.query_id for q in queries}
    if len(ids) != len(queries) or len({q.user_id for q in queries}) != len(queries):
        raise ValueError("Segment evaluation requires unique queries and users")
    catalog = set(item_popularity)
    for arm, rankings in rankings_by_arm.items():
        if set(rankings) != ids:
            raise ValueError(f"Query population differs in {arm}")
        for q in queries:
            ranks = rankings[q.query_id]
            if (len(ranks) > 100 or len(set(ranks)) != len(ranks) or set(ranks) - catalog
                    or set(ranks) & {r.restaurant_id for r in q.history}):
                raise ValueError(f"Invalid past-catalog candidates in {arm}")
    valid = [q for q in queries if any(v > 0 for v in q.relevance_by_item.values())]
    if any(not q.history for q in queries):
        raise ValueError("Zero-history users need a separate cold-start protocol")
    rng = np.random.default_rng(seed)
    resamples = {}
    def indices(key, size):
        if key not in resamples:
            resamples[key] = rng.integers(0, size, size=(samples, size), dtype=np.int32) if size else None
        return resamples[key]
    scores = {}
    for arm, ranks in rankings_by_arm.items():
        scores[arm] = {q.query_id: {
            "recall_at_100": query_scores(RankingObservation(q.query_id, q.user_id, q.relevance_by_item, tuple(ranks[q.query_id])), 100)["recall"],
            "ndcg_at_10": query_scores(RankingObservation(q.query_id, q.user_id, q.relevance_by_item, tuple(ranks[q.query_id])), 10)["ndcg"],
        } for q in valid}
    user_rows = []
    for label in USER_BINS:
        all_group = [q for q in queries if count_bin(len(q.history)) == label]
        group = [q for q in valid if count_bin(len(q.history)) == label]
        row = {"history_bin": label, "queries": len(all_group), "evaluated_users": len(group), "arms": {}}
        ix = indices(("user", label), len(group))
        for arm in rankings_by_arm:
            metrics = {}
            for metric in ("recall_at_100", "ndcg_at_10"):
                values = [scores[arm][q.query_id][metric] for q in group]
                base = [scores[baseline][q.query_id][metric] for q in group]
                metrics[metric] = _mean(values)
                metrics[metric + "_change_vs_baseline"] = _paired(values, base, ix)
            row["arms"][arm] = metrics
        user_rows.append(row)
    item_rows, changes = [], []
    for label in ITEM_BINS:
        truth = {q.query_id: {i: v for i, v in q.relevance_by_item.items()
                 if v > 0 and count_bin(item_popularity.get(i, 0)) == label} for q in valid}
        eligible = [q for q in valid if truth[q.query_id]]
        row = {"visit_bin": label, "catalog_items": sum(count_bin(n) == label for n in item_popularity.values()),
               "relevant_user_item_pairs": sum(len(v) for v in truth.values()),
               "distinct_relevant_items": len({i for v in truth.values() for i in v}),
               "users_with_bin_positives": len(eligible), "contribution_users": len(valid), "arms": {}}
        per_arm = {}
        for arm, ranks in rankings_by_arm.items():
            recalls = {100: [], 10: []}
            contributions = []
            hits = Counter(); exposure = Counter(); exposed_items = {100: set(), 10: set()}
            for q in queries:
                for k in (100, 10):
                    items = {i for i in ranks[q.query_id][:k] if count_bin(item_popularity.get(i, 0)) == label}
                    exposure[k] += len(items); exposed_items[k].update(items)
            for q in valid:
                positives = truth[q.query_id]
                for k in (100, 10):
                    hits[k] += len(set(ranks[q.query_id][:k]) & positives.keys())
                    if positives:
                        recalls[k].append(len(set(ranks[q.query_id][:k]) & positives.keys()) / len(positives))
                ideal = sum((2**v - 1)/math.log2(j+2) for j,v in enumerate(sorted(
                    (v for v in q.relevance_by_item.values() if v > 0), reverse=True)[:10]))
                dcg = sum((2**positives.get(i, 0) - 1)/math.log2(j+2) for j,i in enumerate(ranks[q.query_id][:10]))
                contributions.append(dcg/ideal)
                if arm != baseline:
                    before, after = set(rankings_by_arm[baseline][q.query_id][:100]), set(ranks[q.query_id][:100])
                    gained, lost = positives.keys() & (after-before), positives.keys() & (before-after)
                    if gained or lost:
                        changes.append({"arm": arm, "query_id": q.query_id, "user_id": q.user_id,
                                        "visit_bin": label, "gained_relevant_items": sorted(gained), "lost_relevant_items": sorted(lost)})
            per_arm[arm] = (recalls, contributions)
            summary = {"dcg_contribution_at_10_over_full_idcg": _mean(contributions)}
            for k in (100, 10):
                summary[f"macro_recall_at_{k}"] = _mean(recalls[k])
                summary[f"micro_hits_at_{k}"] = hits[k]
                total = row["relevant_user_item_pairs"]
                summary[f"micro_recall_at_{k}"] = hits[k]/total if total else None
                summary[f"exposures_at_{k}_all_queries"] = exposure[k]
                summary[f"distinct_exposed_items_at_{k}"] = len(exposed_items[k])
            summary["gained_pairs_vs_baseline"] = sum(len(c["gained_relevant_items"]) for c in changes if c["arm"] == arm and c["visit_bin"] == label)
            summary["lost_pairs_vs_baseline"] = sum(len(c["lost_relevant_items"]) for c in changes if c["arm"] == arm and c["visit_bin"] == label)
            row["arms"][arm] = summary
        for arm in rankings_by_arm:
            for k in (100,10):
                row["arms"][arm][f"macro_recall_at_{k}_change_vs_baseline"] = _paired(
                    per_arm[arm][0][k], per_arm[baseline][0][k], indices(("item-recall", label), len(eligible)))
            row["arms"][arm]["dcg_contribution_change_vs_baseline"] = _paired(
                per_arm[arm][1], per_arm[baseline][1], indices(("item-contribution",), len(valid)))
        item_rows.append(row)
    result = {"baseline": baseline, "arms": list(rankings_by_arm), "queries": len(queries), "evaluated_users": len(valid),
              "bootstrap": {"unit": "user", "samples": samples, "seed": seed, "multiple_comparison_adjustment": False},
              "evidence_status": "exploratory repeated-validation subgroup analysis; no independent holdout or multiplicity correction",
              "definitions": {"user_history": "first user-restaurant interactions through query cutoff",
                  "item_visits": "first user-restaurant interactions through query cutoff; zero retained as unseen truth",
                  "item_recall": "macro over users having positives in that bin; original ranks and bin positive denominator",
                  "dcg_contribution": "original rank DCG@10 for bin / full-query IDCG@10, averaged over all relevant users",
                  "exposure": "all queries, including queries with no relevant positives"},
              "user_history_segments": user_rows, "item_visit_segments": item_rows}
    return result, changes
