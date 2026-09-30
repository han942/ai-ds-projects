"""Streamlit explorer for runs in ``artifacts/runs/<run_id>/``.

Tabs: the run's report.md, candidate and LTR metrics with the validation
selection, per-user recommendations, and per-restaurant exposure. Data helpers
are pure functions so they can be tested without Streamlit.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from rating_recsys.experiments.artifacts import read_json, read_jsonl


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
    st.title("Rating Recsys · 후보 생성 → LTR")
    artifacts_root = Path(os.getenv("RATING_RECSYS_ARTIFACTS_DIR", "artifacts"))
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

    def cached(path: Path, loader):
        return loader(str(path), path.stat().st_mtime_ns) if path.exists() else []

    labels = {}
    for path in runs:
        label = cached(path / "manifest.json", load_json).get("label")
        labels[path] = f"{path.name}" + (f" · {label}" if label else "")
    run_dir = st.sidebar.selectbox("Run", runs, format_func=labels.get)
    manifest = cached(run_dir / "manifest.json", load_json)
    metrics = cached(run_dir / "metrics.json", load_json)
    k = int(manifest["config"]["ranking_k"])
    cutoffs = manifest["split"]["cutoffs"]

    columns = st.columns(4)
    columns[0].metric("Snapshot", manifest["snapshot"]["dataset_snapshot_id"][:12])
    columns[1].metric("Train ≤", cutoffs["train_through"])
    columns[2].metric("Validation ≤", cutoffs["validation_through"])
    columns[3].metric("Test 평가 사용자", f"{manifest['windows']['test']['evaluated_users']:,}")

    phase = st.sidebar.radio("Phase", ("test", "validation"), key="phase")
    report_tab, metric_tab, user_tab, item_tab = st.tabs(
        ("보고서", "지표", "사용자별 추천", "식당별 노출")
    )

    with report_tab:
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        st.download_button("report.md 내려받기", report, file_name=f"report_{run_dir.name}.md")
        st.markdown(report)

    phase_metrics = metrics[phase]
    with metric_tab:
        st.subheader(f"후보 생성 ({phase})")
        st.dataframe(
            pd.DataFrame(stage_rows(phase_metrics, CANDIDATE_STAGES, ("recall", "ndcg"))),
            hide_index=True, width="stretch",
        )
        st.subheader(f"LTR 재정렬 ({phase}, Top-{k})")
        st.dataframe(
            pd.DataFrame(stage_rows(phase_metrics, RANKING_STAGES, tuple(METRIC_HELP))),
            hide_index=True, width="stretch",
            column_config={
                name: st.column_config.NumberColumn(name, help=text, format="%.4f")
                for name, text in METRIC_HELP.items()
            },
        )
        if phase == "test":
            st.markdown("**R1 − R0 paired bootstrap (95% CI)**")
            st.dataframe(
                pd.DataFrame(bootstrap_rows(phase_metrics.get("bootstrap_r1_minus_r0"))),
                hide_index=True, width="stretch",
            )
        selection = metrics["selection"]
        st.subheader("Validation 선택")
        left, right = st.columns(2)
        if "candidate_policy_grid" in selection:  # runs with C3 as Stage 1
            left.caption(selection["rule"]["candidate_policy"])
            left.dataframe(pd.DataFrame(selection["candidate_policy_grid"]), hide_index=True)
        elif "lightgcn" in metrics:
            left.caption("C4 LightGCN 학습 query용 checkpoint (고정 설정)")
            left.dataframe(
                pd.DataFrame(metrics["lightgcn"]["training_checkpoints"]), hide_index=True
            )
        right.caption(selection["rule"]["ranker"])
        right.dataframe(
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

    queries = cached(run_dir / f"queries_{phase}.jsonl", load_jsonl)
    recommendations = {
        row["query_id"]: [{**rec, "user_id": row["user_id"]} for rec in row["recommendations"]]
        for row in cached(run_dir / f"recommendations_{phase}.jsonl", load_jsonl)
    }

    with user_tab:
        if not queries:
            st.info(f"No {phase} queries in this run")
        else:
            only_hits = st.toggle("정답을 1개 이상 맞힌 사용자만", value=False)
            options = [
                query for query in queries
                if not only_hits or hit_count(query, recommendations.get(query["query_id"], [])) > 0
            ]
            if not options:
                st.info("조건에 맞는 사용자가 없다")
            else:
                query = st.selectbox(
                    "User",
                    options,
                    format_func=lambda row: (
                        f"u{row['user_id']} · 이력 {len(row['history'])} · "
                        f"window 방문 {len(row['window'])}"
                    ),
                )
                recs = recommendations.get(query["query_id"], [])
                window, top = user_rows(query, recs)
                positives = sum(row["is_positive"] for row in window)
                summary = st.columns(3)
                summary[0].metric("Cutoff 이전 이력", len(query["history"]))
                summary[1].metric("Window 정답", positives)
                summary[2].metric(f"Top-{k} 적중", f"{hit_count(query, recs)} / {positives}")
                left, right = st.columns(2)
                left.markdown("**Cutoff 이전 이력**")
                left.dataframe(pd.DataFrame(query["history"]), hide_index=True, width="stretch")
                right.markdown("**Window 방문 (정답 후보)**")
                right.dataframe(pd.DataFrame(window), hide_index=True, width="stretch")
                st.markdown(f"**모델 추천 Top-{k}**")
                st.dataframe(pd.DataFrame(top), hide_index=True, width="stretch")

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


if __name__ == "__main__":
    main()
