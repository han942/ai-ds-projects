"""Independently audit a completed history-aware Window baseline's outputs.

This verifier uses saved snapshot, query and recommendation records, not the
production label or ranking-metric functions. It recomputes labels, cutoff
history, graded NDCG, macro Recall and absolute rating diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(run_dir: Path, project: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = manifest["config"]
    assert config["ranker_training_mode"] == "window"
    assert config["ranker_label_mode"] == "relevance"
    assert config["satisfaction_mode"] == "history-aware"
    snapshot = Path(manifest["snapshot"]["path"])
    if not snapshot.is_absolute():
        snapshot = project / snapshot
    records = read_lines(snapshot)
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest["snapshot"]["artifact_sha256"]
    assert len(records) == manifest["snapshot"]["interactions"]
    assert len({(r["user_id"], r["restaurant_id"]) for r in records}) == len(records)
    cutoffs = manifest["split"]["cutoffs"]
    result = {"run_id": manifest["run_id"], "independent_recalculation": True, "phases": {}}
    for phase in ("validation", "test"):
        cutoff = cutoffs["train_through" if phase == "validation" else "validation_through"]
        past, future = defaultdict(list), defaultdict(list)
        for row in records:
            if row["event_date"] <= cutoff:
                past[row["user_id"]].append(row)
            elif phase == "test" or row["event_date"] <= cutoffs["validation_through"]:
                future[row["user_id"]].append(row)
        saved = read_lines(run_dir / f"queries_{phase}.jsonl")
        expected_users = {user for user in future if user in past}
        assert {q["user_id"] for q in saved} == expected_users
        recommendations = {q["query_id"]: q["recommendations"] for q in read_lines(run_dir / f"recommendations_{phase}.jsonl")}
        target_ranks = {q["query_id"]: {v["restaurant_id"]: v for v in q["targets"]}
                        for q in read_lines(run_dir / f"target_diagnostics_{phase}.jsonl")}
        ndcgs, recalls, candidate_recalls, high_recalls = [], [], [], []
        high_visits = low_visits = low_hits = personalized = label_changes = 0
        grade_counts = {str(grade): 0 for grade in (0, 1, 2)}
        k, ck = config["ranking_k"], config["candidate_k"]
        for query in saved:
            user = query["user_id"]
            assert query["cutoff"] == cutoff
            assert {(r["restaurant_id"], r["event_date"], r["rating"]) for r in query["history"]} == {
                (r["restaurant_id"], r["event_date"], r["rating"]) for r in past[user]}
            assert {(r["restaurant_id"], r["event_date"], r["rating"]) for r in query["window"]} == {
                (r["restaurant_id"], r["event_date"], r["rating"]) for r in future[user]}
            assert all(r["event_date"] <= cutoff for r in query["history"])
            assert all(r["event_date"] > cutoff for r in query["window"])
            count = len(past[user])
            average = math.fsum(r["rating"] for r in past[user]) / count
            personal = count >= config["satisfaction_min_history"]
            boundary = config["relevance_high_threshold"]
            if personal:
                shift = config["satisfaction_mean_weight"] * (average - boundary)
                boundary += max(-config["satisfaction_max_shift"], min(config["satisfaction_max_shift"], shift))
            profile = query["satisfaction"]
            assert profile["history_count"] == count and profile["personalized"] == personal
            assert math.isclose(profile["past_mean"], average, abs_tol=1e-12)
            assert math.isclose(profile["strong_threshold"], boundary, abs_tol=1e-12)
            personalized += personal
            gains, relevant, high, low = {}, set(), set(), set()
            for visit in query["window"]:
                rating, item = visit["rating"], visit["restaurant_id"]
                grade = 2 if rating >= boundary else (1 if rating >= config["relevance_low_threshold"] else 0)
                assert visit["relevance"] == grade
                old = 2 if rating >= config["relevance_high_threshold"] else (1 if rating >= config["relevance_low_threshold"] else 0)
                label_changes += grade != old
                grade_counts[str(grade)] += 1
                gains[item] = 2 ** grade - 1
                if grade:
                    relevant.add(item)
                if rating >= 4:
                    high.add(item)
                if rating < 3:
                    low.add(item)
            rows = recommendations[query["query_id"]]
            ordered = [row["restaurant_id"] for row in rows]
            assert len(set(ordered)) == len(ordered) <= k
            assert not set(ordered) & {row["restaurant_id"] for row in query["history"]}
            for rank, row in enumerate(rows, 1):
                assert row["final_rank"] == rank
                assert 1 <= row["candidate_rank"] <= ck
            ranks = target_ranks[query["query_id"]]
            assert set(ranks) == set(gains)
            if relevant:
                dcg = math.fsum(gains.get(item, 0) / math.log2(rank + 1) for rank, item in enumerate(ordered, 1))
                ideal = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(sorted(gains.values(), reverse=True)[:k], 1))
                ndcgs.append(dcg / ideal)
                recalls.append(len(set(ordered) & relevant) / len(relevant))
                candidate_recalls.append(sum(ranks[item]["candidate_rank"] is not None and ranks[item]["candidate_rank"] <= ck
                                             for item in relevant) / len(relevant))
            if high:
                high_recalls.append(len(set(ordered) & high) / len(high))
            high_visits += len(high)
            low_visits += len(low)
            low_hits += len(set(ordered) & low)
        mean = lambda values: math.fsum(values) / len(values) if values else 0.0
        recomputed = {
            "queries": len(saved), "evaluated_queries": len(ndcgs),
            "personalized_queries": personalized, "fallback_queries": len(saved) - personalized,
            "label_counts": grade_counts, "labels_changed_vs_absolute": label_changes,
            f"ndcg_at_{k}": mean(ndcgs), f"recall_at_{k}": mean(recalls),
            f"candidate_recall_at_{ck}": mean(candidate_recalls),
            f"high_rating_recall_at_{k}": mean(high_recalls),
            "high_rating_queries": len(high_recalls), "high_rating_visits": high_visits,
            "low_rating_visits": low_visits, f"low_rating_hit_count_at_{k}": low_hits,
            f"low_rating_visit_inclusion_rate_at_{k}": low_hits / low_visits if low_visits else 0.0,
        }
        for key in (f"ndcg_at_{k}", f"recall_at_{k}"):
            assert math.isclose(recomputed[key], metrics[phase]["r1_lambdarank"][key], abs_tol=1e-12)
        assert math.isclose(mean(candidate_recalls), metrics[phase]["c5_c1_lightgcn_rrf"][f"recall_at_{ck}"], abs_tol=1e-12)
        diagnostic = metrics[phase]["rating_diagnostics"]["r1_lambdarank"]
        for key in diagnostic:
            if key in recomputed:
                assert math.isclose(recomputed[key], diagnostic[key], abs_tol=1e-12)
        info = manifest["windows"][phase]["satisfaction"]
        assert info["personalized_users"] == personalized
        assert info["label_counts"] == grade_counts
        assert manifest["windows"][phase]["seen_users"] == len(saved)
        assert manifest["windows"][phase]["evaluated_users"] == len(ndcgs)
        result["phases"][phase] = recomputed
    result["checks"] = {
        "all_seen_future_visits_preserved": True, "history_cutoff_and_mean": True,
        "snapshot_file_checksum": True,
        "history_aware_labels": True, "low_rating_visits_preserved": True,
        "no_seen_items_in_recommendations": True, "recommendations_from_real_candidates": True,
        "full_target_ndcg_and_macro_recall": True, "absolute_rating_diagnostics": True,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    evidence = verify(args.run_dir, args.project)
    path = args.run_dir / "verification.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
