"""Streamlit explorer for runs in ``artifacts/runs/<run_id>/``.

Tabs: the run's report.md, candidate and LTR metrics with the validation
selection, per-user recommendations, and per-restaurant exposure. Data helpers
are pure functions so they can be tested without Streamlit.
"""

from __future__ import annotations

import os
from collections import defaultdict
from math import isclose
from pathlib import Path

from rating_recsys.evaluation.metrics import RankingObservation, query_scores
from rating_recsys.experiments.artifacts import read_json, read_jsonl
from rating_recsys.experiments.prepared import preprocessing_signature


REQUIRED_RUN_FILES = ("manifest.json", "metrics.json", "report.md")
# Runs before 2026-09-30 have C0-C3 only (Stage 1 was C3); stages missing from
# a run are skipped.
CANDIDATE_STAGES = {
    "c5_c1_lightgcn_rrf": "C5 C1+LightGCN RRF (Stage 1)",
    "c1_item_item": "C1 item-item",
    "c4_lightgcn": "C4 LightGCN",
    "c0_popularity": "참고 · C0 전체 인기",
    "c2_region_popularity": "참고 · C2 지역 인기",
    "c3_rrf_union": "참고 · C3 quota RRF",
}
RANKING_STAGES = {
    "r0_candidate_order": "R0 Stage 1 후보 순서",
    "r1_lambdarank": "R1 LambdaRank",
}
METRIC_HELP = {
    "recall": "사용자별 (Top-K 안의 정답 수 / 그 사용자의 전체 정답 수)의 평균",
    "precision": "사용자별 (Top-K 안의 정답 수 / K)의 평균",
    "ndcg": "평점 등급(gain 2^rel−1)과 순위를 함께 반영, 사용자별 ideal DCG로 정규화",
    "map": "정답 위치마다의 precision 합 / min(K, 정답 수)의 평균",
    "mrr": "첫 정답 순위의 역수 평균",
    "catalog_coverage": "전체 후보 catalog 중 한 번 이상 추천된 식당 비율",
    "novelty": "추천 식당의 비인기도 −log2(popularity share) 평균",
    "intra_list_region_diversity": "한 목록 안에서 지역이 다른 식당 쌍의 비율",
}
METRIC_LABELS = {
    "ndcg": "NDCG",
    "recall": "Recall",
    "precision": "Precision",
    "map": "MAP",
    "mrr": "MRR",
    "catalog_coverage": "Catalog coverage",
    "novelty": "Novelty",
    "intra_list_region_diversity": "지역 다양성",
}


# ---------------------------------------------------------------------------
# Pure data helpers
# ---------------------------------------------------------------------------


def discover_runs(artifacts_root: Path) -> list[Path]:
    root = artifacts_root / "runs"
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and all((path / name).exists() for name in REQUIRED_RUN_FILES)
        ),
        reverse=True,
    )


def stage_rows(
    phase_metrics: dict[str, object],
    stages: dict[str, str],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    for stage, label in stages.items():
        values = phase_metrics.get(stage)
        if not isinstance(values, dict):
            continue
        cutoffs = sorted(
            int(key.removeprefix("recall_at_"))
            for key in values
            if key.startswith("recall_at_")
        )
        for cutoff in cutoffs:
            rows.append(
                {
                    "stage": label,
                    "K": cutoff,
                    **{metric: values.get(f"{metric}_at_{cutoff}") for metric in metrics},
                }
            )
    return rows


def bootstrap_rows(bootstrap: dict[str, object] | None) -> list[dict[str, object]]:
    return [
        {
            "metric": key,
            "R0": values["baseline"],
            "R1": values["treatment"],
            "R1 − R0": values["delta"],
            "CI 하한": values["ci95"][0],
            "CI 상한": values["ci95"][1],
            "R1 우세": values["wins"],
            "R1 열세": values["losses"],
            "동률": values["ties"],
        }
        for key, values in (bootstrap or {}).items()
        if isinstance(values, dict)
    ]


def user_rows(
    query: dict[str, object],
    recommendations: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """(window visits with their Top-K rank, Top-K rows with positive flags)."""

    rank_by_item = {int(row["restaurant_id"]): int(row["final_rank"]) for row in recommendations}
    relevance = {int(row["restaurant_id"]): int(row["relevance"]) for row in query["window"]}
    window = [
        {**row, "is_positive": int(row["relevance"]) > 0,
         "rank_in_top_k": rank_by_item.get(int(row["restaurant_id"]))}
        for row in query["window"]
    ]
    top = [
        {**row,
         "is_positive": relevance.get(int(row["restaurant_id"]), 0) > 0,
         "visited_in_window": int(row["restaurant_id"]) in relevance}
        for row in recommendations
    ]
    return window, top


def hit_count(query: dict[str, object], recommendations: list[dict[str, object]]) -> int:
    recommended = {int(row["restaurant_id"]) for row in recommendations}
    return sum(
        1
        for row in query["window"]
        if int(row["relevance"]) > 0 and int(row["restaurant_id"]) in recommended
    )


def diagnostic_index(rows: list[dict[str, object]]) -> dict[str, dict[int, dict[str, object]]]:
    return {
        str(row["query_id"]): {
            int(target["restaurant_id"]): target
            for target in row.get("targets", [])
        }
        for row in rows
    }


def target_rows(
    query: dict[str, object],
    recommendations: list[dict[str, object]],
    *,
    target_diagnostics: dict[int, dict[str, object]] | None = None,
    candidate_ranks: dict[int, int] | None = None,
    ranking_k: int,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, object]]:
    """Explain observed future visits against retrieval and final-ranker output."""

    final_top_k = {
        int(row["restaurant_id"]): int(row["final_rank"])
        for row in recommendations
    }
    diagnostics = target_diagnostics or {}
    audit_available = target_diagnostics is not None or candidate_ranks is not None
    history_ratings = [float(row["rating"]) for row in query["history"] if row.get("rating") is not None]
    history_average = sum(history_ratings) / len(history_ratings) if history_ratings else None
    rows = []
    for target in query["window"]:
        restaurant_id = int(target["restaurant_id"])
        details = diagnostics.get(restaurant_id, {})
        candidate_rank = details.get("candidate_rank")
        if candidate_rank is None and candidate_ranks is not None:
            candidate_rank = candidate_ranks.get(restaurant_id)
        final_rank = details.get("final_rank")
        if final_rank is None:
            final_rank = final_top_k.get(restaurant_id)
        rating = float(target["rating"])
        relevance = int(target.get("relevance", 0))
        if rating < low_threshold:
            rating_group = "저평점 방문 · 비관련"
        elif relevance >= 2:
            rating_group = "강한 만족 · 강한 정답"
        else:
            rating_group = "약한 만족 · 약한 정답"
        if final_rank is not None and int(final_rank) <= ranking_k:
            outcome = "최종 Top-K 추천"
        elif candidate_rank is not None and final_rank is not None:
            outcome = f"후보 검색 성공 · 최종 {int(final_rank)}위"
        elif candidate_rank is not None:
            outcome = "후보 검색 성공 · 저장된 Top-K에는 없음"
        elif audit_available:
            outcome = "Retriever가 후보로 찾지 못함"
        else:
            outcome = "후보 여부 자료 없음"
        rows.append(
            {
                "event_date": target.get("event_date"),
                "restaurant_name": target.get("restaurant_name"),
                "rating": rating,
                "평점−과거 평균": rating - history_average if history_average is not None else None,
                "평가상 분류": rating_group,
                "relevance": relevance,
                "candidate_rank": candidate_rank,
                "final_rank": final_rank,
                "결과": outcome,
            }
        )
    return rows


def rating_threshold_rows(
    queries: list[dict[str, object]],
    recommendations: dict[str, list[dict[str, object]]],
    *,
    thresholds: tuple[float, ...],
    cutoff: int,
) -> list[dict[str, object]]:
    """Re-score the same R1 lists under alternative absolute rating cutoffs."""

    rows = []
    for threshold in thresholds:
        observations = []
        positive_visits = 0
        low_rating_visits = 0
        recommended_low_rating_visits = 0
        for query in queries:
            relevance = {
                int(target["restaurant_id"]): int(float(target["rating"]) >= threshold)
                for target in query["window"]
            }
            positive_visits += sum(relevance.values())
            ranked = tuple(
                int(row["restaurant_id"])
                for row in recommendations.get(str(query["query_id"]), [])
            )
            recommended = set(ranked[:cutoff])
            low_targets = {
                int(target["restaurant_id"])
                for target in query["window"]
                if float(target["rating"]) < min(thresholds)
            }
            low_rating_visits += len(low_targets)
            recommended_low_rating_visits += len(low_targets & recommended)
            observations.append(
                RankingObservation(
                    query_id=str(query["query_id"]),
                    user_id=int(query["user_id"]),
                    relevance_by_item=relevance,
                    ordered_restaurant_ids=ranked,
                )
            )
        evaluated = [observation for observation in observations if observation.relevant_items]
        scores = [query_scores(observation, cutoff) for observation in evaluated]
        denominator = len(scores)
        rows.append(
            {
                "평점 기준": f"≥ {threshold:g}점",
                "평가 사용자": denominator,
                "긍정 정답 없는 사용자": len(queries) - denominator,
                "긍정 방문 수": positive_visits,
                "저평점 방문": low_rating_visits,
                "Top-K에 든 저평점 방문": recommended_low_rating_visits,
                f"NDCG@{cutoff}": sum(score["ndcg"] for score in scores) / denominator if denominator else 0.0,
                f"Recall@{cutoff}": sum(score["recall"] for score in scores) / denominator if denominator else 0.0,
                f"Precision@{cutoff}": sum(score["precision"] for score in scores) / denominator if denominator else 0.0,
            }
        )
    return rows


def review_text_index(snapshot_path: Path) -> dict[tuple[int, int, str], str | None]:
    """Join the text sidecar to snapshot events without putting text in run files."""

    if not snapshot_path.exists():
        return {}
    text_path = snapshot_path.with_name(f"{snapshot_path.stem}.reviews.jsonl")
    if not text_path.exists():
        return {}
    review_ids = {
        int(row["review_id"]): (int(row["user_id"]), int(row["restaurant_id"]), str(row["event_date"]))
        for row in read_jsonl(snapshot_path)
    }
    texts = {
        int(row["review_id"]): row.get("review_text")
        for row in read_jsonl(text_path)
    }
    return {
        key: texts.get(review_id)
        for review_id, key in review_ids.items()
    }


def attach_review_text(
    row: dict[str, object],
    user_id: int,
    index: dict[tuple[int, int, str], str | None],
) -> dict[str, object]:
    key = (int(user_id), int(row["restaurant_id"]), str(row["event_date"]))
    return {**row, "review_text": index.get(key)}


def _prepared_identity_matches(
    identity: dict[str, object], manifest: dict[str, object], phase: str
) -> bool:
    config = manifest["config"]
    expected_lightgcn = {
        "dimension": config["lightgcn_dimension"],
        "layers": config["lightgcn_layers"],
        "epochs": config["lightgcn_epochs"],
        "batch_size": config["lightgcn_batch_size"],
        "learning_rate": config["lightgcn_learning_rate"],
        "regularization": config["lightgcn_regularization"],
        "init_std": 0.1,
        "seed": config["random_seed"],
    }
    expected = {
        "kind": phase,
        "snapshot_id": manifest["snapshot"]["dataset_snapshot_id"],
        "cutoffs": manifest["split"]["cutoffs"],
        "candidate_k": config["candidate_k"],
        "rrf_constant": config["rrf_constant"],
        "region_mode": config["region_mode"],
        "legacy_c3_quota": config["legacy_c3_quota"],
        "lightgcn": expected_lightgcn,
        "features": manifest["feature_schema"],
        "relevance_thresholds": [
            config["relevance_low_threshold"], config["relevance_high_threshold"]
        ],
        "rating_shrinkage_strength": config.get("rating_shrinkage_strength") or 0.0,
    }
    return all(identity.get(key) == value for key, value in expected.items())


def verified_prepared_candidate_ranks(
    artifacts_root: Path,
    manifest: dict[str, object],
    phase: str,
    queries: list[dict[str, object]],
    phase_metrics: dict[str, object],
) -> dict[str, dict[int, int]] | None:
    """Reuse legacy prepared candidates only when their Recall matches this run."""

    snapshot_id = str(manifest["snapshot"]["dataset_snapshot_id"])
    root = artifacts_root / "prepared" / snapshot_id[:16]
    query_ids = {str(row["query_id"]) for row in queries}
    current_preprocessing = preprocessing_signature()
    for path in sorted(root.glob(f"{phase}-*/manifest.json"), reverse=True):
        try:
            cached = read_json(path)
            identity = cached["identity"]
            if not _prepared_identity_matches(identity, manifest, phase):
                continue
            if identity.get("preprocessing") != current_preprocessing:
                continue
            ordered = cached["data"]["ordered"]["c5_c1_lightgcn_rrf"]
            if set(ordered) != query_ids:
                continue
            valid = True
            for key, expected_recall in phase_metrics.get("c5_c1_lightgcn_rrf", {}).items():
                if not key.startswith("recall_at_"):
                    continue
                cutoff = int(key.removeprefix("recall_at_"))
                per_user = []
                for query in queries:
                    relevant = {
                        int(target["restaurant_id"])
                        for target in query["window"]
                        if int(target["relevance"]) > 0
                    }
                    if relevant:
                        found = relevant & set(ordered[str(query["query_id"])][:cutoff])
                        per_user.append(len(found) / len(relevant))
                observed = sum(per_user) / len(per_user) if per_user else 0.0
                if not isclose(observed, float(expected_recall), rel_tol=0.0, abs_tol=1e-8):
                    valid = False
                    break
            if valid:
                return {
                    query_id: {
                        int(restaurant_id): rank
                        for rank, restaurant_id in enumerate(ids, 1)
                    }
                    for query_id, ids in ordered.items()
                }
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return None


def resolve_snapshot_path(artifacts_root: Path, manifest: dict[str, object]) -> Path | None:
    raw = Path(str(manifest["snapshot"].get("path", "")))
    candidates = [raw] if raw.is_absolute() else [artifacts_root.parent / raw]
    candidates.append(artifacts_root / "snapshots" / raw.name)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def item_summary(
    queries: list[dict[str, object]],
    recommendations: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Per restaurant: Top-K exposures and how many exposed users visited it."""

    positives = {
        query["query_id"]: {
            int(row["restaurant_id"]) for row in query["window"] if int(row["relevance"]) > 0
        }
        for query in queries
    }
    grouped: dict[int, dict[str, object]] = defaultdict(
        lambda: {"exposures": 0, "hits": 0, "rank_sum": 0, "name": None}
    )
    for query_id, rows in recommendations.items():
        for row in rows:
            item = grouped[int(row["restaurant_id"])]
            item["exposures"] += 1
            item["rank_sum"] += int(row["final_rank"])
            item["hits"] += int(int(row["restaurant_id"]) in positives.get(query_id, set()))
            item["name"] = row.get("restaurant_name")
    return sorted(
        (
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": values["name"],
                "top_k_exposures": values["exposures"],
                "visited_by_exposed_users": values["hits"],
                "average_rank": values["rank_sum"] / values["exposures"],
            }
            for restaurant_id, values in grouped.items()
        ),
        key=lambda row: (-row["top_k_exposures"], row["restaurant_id"]),
    )


def item_rows(
    restaurant_id: int,
    queries: list[dict[str, object]],
    recommendations: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    by_id = {query["query_id"]: query for query in queries}
    rows = []
    for query_id, recs in recommendations.items():
        for row in recs:
            if int(row["restaurant_id"]) != restaurant_id:
                continue
            query = by_id.get(query_id, {"window": [], "history": []})
            relevance = {
                int(visit["restaurant_id"]): int(visit["relevance"])
                for visit in query["window"]
            }
            rows.append(
                {
                    "user_id": row.get("user_id", query.get("user_id")),
                    "final_rank": row["final_rank"],
                    "candidate_rank": row.get("candidate_rank"),
                    "visited_in_window": restaurant_id in relevance,
                    "is_positive": relevance.get(restaurant_id, 0) > 0,
                    "history_length": len(query["history"]),
                }
            )
    return sorted(rows, key=lambda row: (row["final_rank"], str(row["user_id"])))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "The dashboard requires pandas and Streamlit from the experiment extra"
        ) from exc

    st.set_page_config(page_title="Rating Recsys Explorer", layout="wide")
    st.title("Rating Recsys · 추천 품질")
    artifacts_root = Path(os.getenv("RATING_RECSYS_ARTIFACTS_DIR", "artifacts")).resolve()
    runs = discover_runs(artifacts_root)
    if not runs:
        st.info("No runs found. Run `rating-recsys-experiment` first.")
        return

    @st.cache_data(show_spinner=False)
    def load_json(path: str, modified_ns: int):
        del modified_ns
        return read_json(Path(path))

    @st.cache_data(show_spinner=False)
    def load_jsonl(path: str, modified_ns: int):
        del modified_ns
        return read_jsonl(Path(path))

    @st.cache_data(show_spinner=False)
    def load_legacy_candidates(
        root: str, run_path: str, phase_name: str,
        manifest_ns: int, metrics_ns: int, queries_ns: int,
    ):
        del manifest_ns, metrics_ns, queries_ns
        run_path_ = Path(run_path)
        run_manifest = read_json(run_path_ / "manifest.json")
        phase_results = read_json(run_path_ / "metrics.json")[phase_name]
        phase_queries = read_jsonl(run_path_ / f"queries_{phase_name}.jsonl")
        return verified_prepared_candidate_ranks(
            Path(root), run_manifest, phase_name, phase_queries, phase_results
        )

    @st.cache_data(show_spinner=False)
    def load_reviews(snapshot: str, snapshot_ns: int, review_ns: int):
        del snapshot_ns, review_ns
        return review_text_index(Path(snapshot))

    def cached(path: Path, loader):
        return loader(str(path), path.stat().st_mtime_ns) if path.exists() else []

    labels = {}
    for path in runs:
        label = cached(path / "manifest.json", load_json).get("label")
        labels[path] = f"{path.name}" + (f" · {label}" if label else "")
    run_dir = st.sidebar.selectbox("Run", runs, format_func=labels.get)
    manifest = cached(run_dir / "manifest.json", load_json)
    metrics = cached(run_dir / "metrics.json", load_json)
    config = manifest["config"]
    k = int(config["ranking_k"])
    candidate_k = int(config["candidate_k"])
    phase = st.sidebar.radio("평가 구간", ("test", "validation"), key="phase")
    phase_metrics = metrics[phase]
    queries = cached(run_dir / f"queries_{phase}.jsonl", load_jsonl)
    recommendation_rows = cached(run_dir / f"recommendations_{phase}.jsonl", load_jsonl)
    recommendations = {
        str(row["query_id"]): [{**rec, "user_id": row["user_id"]} for rec in row["recommendations"]]
        for row in recommendation_rows
    }
    diagnostic_path = run_dir / f"target_diagnostics_{phase}.jsonl"
    diagnostic_records = cached(diagnostic_path, load_jsonl)
    diagnostics = diagnostic_index(diagnostic_records)
    candidate_ranks = None
    if not diagnostic_records and queries:
        candidate_ranks = load_legacy_candidates(
            str(artifacts_root),
            str(run_dir),
            phase,
            (run_dir / "manifest.json").stat().st_mtime_ns,
            (run_dir / "metrics.json").stat().st_mtime_ns,
            (run_dir / f"queries_{phase}.jsonl").stat().st_mtime_ns,
        )
    stage1_key = "c5_c1_lightgcn_rrf" if "c5_c1_lightgcn_rrf" in phase_metrics else "c3_rrf_union"
    stage1_recall = phase_metrics.get(stage1_key, {}).get(f"recall_at_{candidate_k}")
    ranker_metrics = phase_metrics.get("r1_lambdarank", {})
    r0_metrics = phase_metrics.get("r0_candidate_order", {})
    ranker_ndcg = ranker_metrics.get(f"ndcg_at_{k}")
    ranker_recall = ranker_metrics.get(f"recall_at_{k}")
    r0_ndcg = r0_metrics.get(f"ndcg_at_{k}")
    columns = st.columns(4)
    columns[0].metric(f"후보 Recall@{candidate_k}", f"{stage1_recall:.2%}" if stage1_recall is not None else "—")
    columns[1].metric(
        f"LambdaRank NDCG@{k}",
        f"{ranker_ndcg:.4f}" if ranker_ndcg is not None else "—",
        delta=f"{ranker_ndcg-r0_ndcg:+.4f} vs 후보 순서" if ranker_ndcg is not None and r0_ndcg is not None else None,
    )
    columns[2].metric(f"LambdaRank Recall@{k}", f"{ranker_recall:.2%}" if ranker_recall is not None else "—")
    columns[3].metric("평가 사용자", f"{manifest['windows'][phase]['evaluated_users']:,}")
    st.caption(
        f"{run_dir.name} · snapshot {manifest['snapshot']['dataset_snapshot_id'][:12]} · "
        f"Train ≤ {manifest['split']['cutoffs']['train_through']} · "
        f"Validation ≤ {manifest['split']['cutoffs']['validation_through']}"
    )
    metric_tab, user_tab, item_tab, report_tab = st.tabs(
        ("지표 요약", "사용자별 진단", "식당별 노출", "상세 보고서")
    )
    with metric_tab:
        candidate_data = stage_rows(
            phase_metrics, CANDIDATE_STAGES, ("recall", "ndcg")
        )
        st.subheader("후보 생성 · 검색이 정답 식당을 찾았나")
        if candidate_data:
            candidate_frame = pd.DataFrame(candidate_data)
            st.bar_chart(
                candidate_frame,
                x="K",
                y="recall",
                color="stage",
                stack=False,
                y_label="사용자별 평균 Recall",
            )
            candidate_frame = candidate_frame.rename(
                columns={"stage": "후보 모델", "recall": "Recall", "ndcg": "NDCG"}
            )
            st.dataframe(
                candidate_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Recall": st.column_config.NumberColumn("Recall", format="%.2%", help=METRIC_HELP["recall"]),
                    "NDCG": st.column_config.NumberColumn("NDCG", format="%.4f", help=METRIC_HELP["ndcg"]),
                },
            )
        else:
            st.info("후보 생성 지표가 이 run에 없습니다.")

        ranking_data = stage_rows(
            phase_metrics, RANKING_STAGES, tuple(METRIC_HELP)
        )
        st.subheader(f"최종 추천 순위 · Top-{k}")
        if ranking_data:
            selected_metric = st.selectbox(
                "비교할 지표",
                tuple(METRIC_LABELS),
                format_func=lambda value: METRIC_LABELS[value],
                key="ranking_metric",
            )
            chart_rows = [
                {"stage": row["stage"], "K": row["K"], selected_metric: row[selected_metric]}
                for row in ranking_data
                if row.get(selected_metric) is not None
            ]
            st.bar_chart(
                pd.DataFrame(chart_rows),
                x="K",
                y=selected_metric,
                color="stage",
                stack=False,
                y_label=METRIC_LABELS[selected_metric],
            )
            ranking_frame = pd.DataFrame(ranking_data).rename(
                columns={"stage": "순위 모델", **METRIC_LABELS}
            )
            st.dataframe(
                ranking_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    column: st.column_config.NumberColumn(
                        column,
                        format="%.2%" if metric in {"recall", "precision"} else "%.4f",
                        help=METRIC_HELP[metric],
                    )
                    for metric, column in METRIC_LABELS.items()
                    if column in ranking_frame.columns
                },
            )
        else:
            st.info("최종 순위 지표가 이 run에 없습니다.")

        if queries and recommendations:
            low = float(config["relevance_low_threshold"])
            high = float(config["relevance_high_threshold"])
            thresholds = tuple(sorted({low, (low + high) / 2, high}))
            st.subheader("평점 기준에 따른 결과 민감도")
            st.caption(
                "같은 LambdaRank Top-K를 두고 미래 방문의 긍정 기준만 바꿔 다시 계산합니다. "
                "모델 재학습이나 실제 추천 노출 효과를 뜻하지 않습니다."
            )
            sensitivity = rating_threshold_rows(
                queries, recommendations, thresholds=thresholds, cutoff=k
            )
            st.dataframe(
                pd.DataFrame(sensitivity),
                hide_index=True,
                width="stretch",
                column_config={
                    f"NDCG@{k}": st.column_config.NumberColumn(f"NDCG@{k}", format="%.4f"),
                    f"Recall@{k}": st.column_config.NumberColumn(f"Recall@{k}", format="%.2%"),
                    f"Precision@{k}": st.column_config.NumberColumn(f"Precision@{k}", format="%.2%"),
                },
            )
            if config.get("satisfaction_mode") == "history-aware":
                st.caption(
                    f"현재 주 지표는 {config['satisfaction_min_history']}건 이상 과거 이력이 있으면 "
                    "사용자 과거 평균으로 강한 만족 경계를 조정합니다. "
                    f"{low:g}점 미만은 비관련입니다. 위 표는 별도의 절대 평점 진단이며 주 평가 기준을 대체하지 않습니다."
                )
            else:
                st.caption(
                    f"현재 주 지표는 {low:g}점 미만=비관련, {low:g}~{high:g}점=약한 정답, "
                    f"{high:g}점 이상=강한 정답입니다. 위 민감도 표는 각 기준 이상 방문을 모두 같은 긍정으로 봅니다."
                )
            st.caption(
                "‘Top-K에 든 저평점 방문’은 같은 사용자가 이후 낮게 평가한 식당과 추천 목록의 겹침입니다. "
                "노출·클릭 로그가 없으므로 추천 때문에 방문·불만족했다고 해석할 수는 없습니다."
            )
        if phase == "test":
            with st.expander("R1 − R0 사용자별 paired bootstrap · 95% CI"):
                st.dataframe(
                    pd.DataFrame(bootstrap_rows(phase_metrics.get("bootstrap_r1_minus_r0"))),
                    hide_index=True,
                    width="stretch",
                )
        selection = metrics.get("selection", {})
        with st.expander("Validation에서 고른 설정과 학습 진단"):
            if "candidate_policy_grid" in selection:
                st.caption(selection["rule"]["candidate_policy"])
                st.dataframe(pd.DataFrame(selection["candidate_policy_grid"]), hide_index=True)
            elif "lightgcn" in metrics:
                st.caption("C4 LightGCN 학습 query용 checkpoint")
                st.dataframe(pd.DataFrame(metrics["lightgcn"]["training_checkpoints"]), hide_index=True)
            if selection.get("ranker_grid"):
                st.caption(selection["rule"]["ranker"])
                st.dataframe(
                    pd.DataFrame(
                        [
                            {key: value for key, value in row.items()
                             if key != "lightgbm_validation_ndcg_curve"}
                            | {"chosen": row["name"] == selection["chosen_ranker"]}
                            for row in selection["ranker_grid"]
                        ]
                    ),
                    hide_index=True,
                )
                curves = {
                    row["name"]: row.get("lightgbm_validation_ndcg_curve") or []
                    for row in selection["ranker_grid"]
                }
                if any(curves.values()):
                    length = max(len(curve) for curve in curves.values())
                    st.markdown(f"**트리 수별 validation NDCG@{k}**")
                    st.line_chart(
                        pd.DataFrame(
                            {name: curve + [None] * (length - len(curve)) for name, curve in curves.items()},
                            index=range(1, length + 1),
                        )
                    )
            importance = metrics.get("feature_importance") or {}
            if importance:
                st.markdown("**최종 ranker feature importance**")
                st.bar_chart(
                    pd.DataFrame(
                        sorted(importance.items(), key=lambda pair: pair[1], reverse=True),
                        columns=["feature", "importance"],
                    ).set_index("feature")
                )

    with user_tab:
        if not queries:
            st.info(f"No {phase} queries in this run")
        else:
            low = float(config["relevance_low_threshold"])
            high = float(config["relevance_high_threshold"])

            def targets_for(query):
                query_id = str(query["query_id"])
                return target_rows(
                    query,
                    recommendations.get(query_id, []),
                    target_diagnostics=diagnostics.get(query_id) if diagnostic_records else None,
                    candidate_ranks=(candidate_ranks.get(query_id) if candidate_ranks is not None else None),
                    ranking_k=k,
                    low_threshold=low,
                    high_threshold=high,
                )

            filter_choice = st.selectbox(
                "사용자·정답 필터",
                (
                    "전체",
                    "Top-K 적중 있음",
                    "Retriever가 놓친 긍정 방문",
                    "후보는 찾았지만 Top-K 밖",
                    "저평점 방문 있음",
                ),
            )

            def include_query(query):
                rows = targets_for(query)
                if filter_choice == "Top-K 적중 있음":
                    return any(row["relevance"] > 0 and row["결과"] == "최종 Top-K 추천" for row in rows)
                if filter_choice == "Retriever가 놓친 긍정 방문":
                    return any(row["relevance"] > 0 and row["결과"] == "Retriever가 후보로 찾지 못함" for row in rows)
                if filter_choice == "후보는 찾았지만 Top-K 밖":
                    return any(
                        row["relevance"] > 0
                        and row["candidate_rank"] is not None
                        and row["결과"] != "최종 Top-K 추천"
                        for row in rows
                    )
                if filter_choice == "저평점 방문 있음":
                    return any(float(row["rating"]) < low for row in rows)
                return True

            options = [query for query in queries if include_query(query)]
            if not options:
                st.info("조건에 맞는 사용자가 없습니다.")
            else:
                query = st.selectbox(
                    "사용자",
                    options,
                    format_func=lambda row: (
                        f"u{row['user_id']} · 이력 {len(row['history'])} · "
                        f"미래 방문 {len(row['window'])} · Top-{k} 적중 "
                        f"{hit_count(row, recommendations.get(str(row['query_id']), []))}"
                    ),
                )
                query_id = str(query["query_id"])
                recs = recommendations.get(query_id, [])
                user_targets = targets_for(query)
                positives = sum(int(row["relevance"] > 0) for row in user_targets)
                low_visits = sum(float(row["rating"]) < low for row in user_targets)
                user_scores = None
                if positives:
                    user_scores = query_scores(
                        RankingObservation(
                            query_id=query_id,
                            user_id=int(query["user_id"]),
                            relevance_by_item={
                                int(row["restaurant_id"]): int(row["relevance"])
                                for row in query["window"]
                            },
                            ordered_restaurant_ids=tuple(
                                int(row["restaurant_id"]) for row in recs
                            ),
                        ),
                        k,
                    )
                history_ratings = [float(row["rating"]) for row in query["history"]]
                future_ratings = [float(row["rating"]) for row in query["window"]]
                summary = st.columns(7)
                summary[0].metric("Cutoff 이전 방문", len(history_ratings))
                summary[1].metric(
                    "과거 평균 별점",
                    f"{sum(history_ratings)/len(history_ratings):.2f}★" if history_ratings else "—",
                )
                summary[2].metric(
                    "미래 방문 평균 별점",
                    f"{sum(future_ratings)/len(future_ratings):.2f}★" if future_ratings else "—",
                )
                summary[3].metric("긍정 정답", positives)
                summary[4].metric("저평점 방문", low_visits)
                summary[5].metric(f"Top-{k} hit", f"{hit_count(query, recs)} / {positives}")
                summary[6].metric(f"NDCG@{k}", f"{user_scores['ndcg']:.3f}" if user_scores else "—")

                if diagnostic_records:
                    st.caption("정답 방문별 Retriever 순위와 전체 LambdaRank 순위를 기록한 run입니다.")
                elif candidate_ranks is not None:
                    st.caption(
                        "후보 순위는 prepared 데이터와 aggregate Recall 일치로 확인했습니다. "
                        "이전 run은 Top-K 바깥 LambdaRank 순위를 저장하지 않아 정확한 순위는 비어 있습니다."
                    )
                else:
                    st.warning("사용자별 후보 기록이 없어 Retriever 누락과 재정렬 누락을 구분할 수 없습니다.")

                st.markdown("**미래 방문별 판정**")
                target_frame = pd.DataFrame(user_targets).rename(
                    columns={
                        "event_date": "방문일",
                        "restaurant_name": "식당",
                        "rating": "실제 평점",
                        "평점−과거 평균": "과거 평균 대비",
                        "평가상 분류": "평점 분류",
                        "candidate_rank": "Retriever 순위",
                        "final_rank": "LambdaRank 순위",
                    }
                ).drop(columns=["relevance"], errors="ignore")
                st.dataframe(target_frame, hide_index=True, width="stretch")
                st.markdown(f"**LambdaRank 추천 Top-{k}**")
                _, top = user_rows(query, recs)
                st.dataframe(pd.DataFrame(top), hide_index=True, width="stretch")

                snapshot_path = resolve_snapshot_path(artifacts_root, manifest)
                if snapshot_path is None:
                    st.info("실행 snapshot을 찾을 수 없어 리뷰 본문을 표시할 수 없습니다.")
                else:
                    review_path = snapshot_path.with_name(f"{snapshot_path.stem}.reviews.jsonl")
                    review_index = (
                        load_reviews(
                            str(snapshot_path),
                            snapshot_path.stat().st_mtime_ns,
                            review_path.stat().st_mtime_ns,
                        )
                        if review_path.exists()
                        else {}
                    )
                    history_with_reviews = [
                        attach_review_text(row, int(query["user_id"]), review_index)
                        for row in query["history"]
                    ]
                    future_with_reviews = [
                        attach_review_text(row, int(query["user_id"]), review_index)
                        for row in query["window"]
                    ]
                    if not review_index:
                        st.info("리뷰 본문 sidecar가 없어 별점·방문 정보만 표시합니다.")
                    elif future_with_reviews:
                        st.markdown("**사람이 확인할 리뷰 근거**")
                        target_index = st.selectbox(
                            "확인할 미래 정답",
                            list(range(len(future_with_reviews))),
                            format_func=lambda index: (
                                f"{future_with_reviews[index]['event_date']} · "
                                f"{future_with_reviews[index]['restaurant_name']} · "
                                f"{future_with_reviews[index]['rating']:g}점"
                            ),
                            key=f"target-review-{query_id}",
                        )
                        target_review = future_with_reviews[target_index]
                        left, right = st.columns(2)
                        left.markdown("**Cutoff 이전 사용자 리뷰**")
                        left.caption("모델이 알 수 있었던 과거 이력입니다.")
                        if history_with_reviews:
                            count = st.number_input(
                                "최근 리뷰 수",
                                min_value=1,
                                max_value=len(history_with_reviews),
                                value=min(10, len(history_with_reviews)),
                                step=1,
                            )
                            for row in history_with_reviews[-int(count):]:
                                with left.expander(
                                    f"{row['event_date']} · {row['restaurant_name']} · {row['rating']:g}점"
                                ):
                                    left.write(row.get("review_text") or "리뷰 본문 없음")
                        right.markdown("**미래 정답 리뷰**")
                        right.caption("cutoff 이후 평가 근거입니다. 모델 입력에는 사용하지 않습니다.")
                        right.write(target_review.get("review_text") or "리뷰 본문 없음")

    with item_tab:
        summaries = item_summary(queries, recommendations)
        if not summaries:
            st.info(f"No {phase} recommendations in this run")
        else:
            by_id = {row["restaurant_id"]: row for row in summaries}
            restaurant_id = st.selectbox(
                "식당",
                list(by_id),
                format_func=lambda value: (
                    f"{by_id[value]['restaurant_name']} · id={value} · "
                    f"Top-{k} 노출 {by_id[value]['top_k_exposures']}회"
                ),
            )
            summary = by_id[restaurant_id]
            columns = st.columns(3)
            columns[0].metric(f"Top-{k} 노출", summary["top_k_exposures"])
            columns[1].metric("노출된 사용자 중 실제 방문", summary["visited_by_exposed_users"])
            columns[2].metric("평균 순위", f"{summary['average_rank']:.2f}")
            st.dataframe(
                pd.DataFrame(item_rows(restaurant_id, queries, recommendations)),
                hide_index=True, width="stretch",
            )
            with st.expander("전체 식당 노출 순위"):
                st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")

    with report_tab:
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        st.download_button("report.md 내려받기", report, file_name=f"report_{run_dir.name}.md")
        st.markdown(report)


if __name__ == "__main__":
    main()
