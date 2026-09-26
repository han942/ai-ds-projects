"""Streamlit UI for inspecting candidate retrieval and LTR movement."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from rating_recsys.experiments.artifacts import read_parquet
from rating_recsys.experiments.config import ExperimentConfig


REQUIRED_RUN_FILES = ("manifest.json", "metrics.json", "queries.jsonl")

STAGE_LABELS = {
    "c0_popularity": "C0 Popularity",
    "c1_item_item": "C1 Item-item CF",
    "c2_region_popularity": "C2 Region popularity",
    "c3_rrf_union": "C3 RRF Union",
    "r1_lambdarank": "R1 LambdaMART",
}

METRIC_HELP = {
    "recall": "정답 아이템이 상위 K개 안에 포함된 query 비율입니다. 이 실험은 query당 정답이 하나라 Hit Rate@K와 같습니다.",
    "ndcg": "정답을 더 높은 순위에 배치할수록 높은 값을 주는 순위 품질 지표입니다. 1에 가까울수록 좋습니다.",
    "mrr": "정답 순위의 역수(1/rank)를 query별로 평균한 값입니다. 첫 정답이 위에 있을수록 높습니다.",
    "catalog_coverage": "전체 평가 가능 catalog 중 추천 목록에 한 번 이상 등장한 아이템 비율입니다.",
    "novelty": "추천 아이템의 비인기도를 -log2(popularity probability)로 측정한 평균입니다. 높을수록 덜 인기 있는 아이템을 추천합니다.",
    "region_diversity": "한 추천 목록 안의 아이템 쌍 중 서로 다른 지역에 속한 비율입니다. 높을수록 지역 구성이 다양합니다.",
    "evaluated_queries": "relevance가 0보다 커서 ranking metric 계산에 실제 포함된 query 수입니다.",
    "zero_relevance_rate": "전체 query 중 target rating이 relevance 0으로 변환되어 ranking metric에서 제외된 비율입니다.",
    "target_availability": "query 시점의 candidate catalog에 target 아이템 자체가 존재했던 비율입니다.",
    "retrieval_p95": "candidate 생성 시간의 95백분위입니다. query 100개 중 약 95개가 이 시간 이내에 완료됩니다.",
    "ranking_p95": "LambdaMART 재정렬 시간의 95백분위입니다.",
}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_jsonl_query(path: Path, query_id: str) -> list[dict[str, object]]:
    """Read one query without materialising a potentially large JSONL artifact."""

    if not path.exists():
        return []
    token = f'"query_id":"{query_id}"'
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if token in line]


def _read_detail_query(path: Path, query_id: str) -> list[dict[str, object]]:
    if path.suffix == ".parquet":
        return read_parquet(path, query_id=query_id)
    return _read_jsonl_query(path, query_id)


def _discover_run_dirs(artifacts_root: Path) -> list[Path]:
    """Return only completed-enough runs that the explorer can render."""

    runs_root = artifacts_root / "runs"
    runs = []
    current_schema = ExperimentConfig().schema_version
    for path in runs_root.glob("*"):
        if not path.is_dir() or not all(
            (path / filename).exists() for filename in REQUIRED_RUN_FILES
        ):
            continue
        try:
            manifest = _read_json(path / "manifest.json")
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        config = manifest.get("config")
        if isinstance(config, dict) and config.get("schema_version") == current_schema:
            runs.append(path)
    return sorted(runs, reverse=True)


def _metric_cutoff_rows(
    phase_metrics: dict[str, object], *, region_mode: str = "with_region"
) -> list[dict[str, object]]:
    """Flatten stage metrics to one row per stage and cutoff."""

    rows: list[dict[str, object]] = []
    for stage_key, stage_label in STAGE_LABELS.items():
        if region_mode == "without_region" and stage_key == "c2_region_popularity":
            continue
        if region_mode == "without_region" and stage_key == "c3_rrf_union":
            stage_label = "C0+C1 RRF Union"
        values = phase_metrics.get(stage_key)
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
                    "stage": stage_label,
                    "K": cutoff,
                    "recall_hit_rate": values.get(f"recall_at_{cutoff}"),
                    "ndcg": values.get(f"ndcg_at_{cutoff}"),
                    "mrr": values.get(f"mrr_at_{cutoff}"),
                    "catalog_coverage": values.get(
                        f"catalog_coverage_at_{cutoff}"
                    ),
                    "novelty": values.get(f"novelty_at_{cutoff}"),
                    "region_diversity": values.get(
                        f"intra_list_region_diversity_at_{cutoff}"
                    ),
                }
            )
    return rows


def _dataset_review_index(
    dataset_rows: list[dict[str, object]],
) -> dict[tuple[int, int], int]:
    return {
        (int(row["user_id"]), int(row["restaurant_id"])): int(row["review_id"])
        for row in dataset_rows
    }


def _query_review_ids(
    query: dict[str, object],
    dataset_index: dict[tuple[int, int], int],
) -> tuple[list[int | None], int | None]:
    """Resolve review ids for both new and pre-review-context run artifacts."""

    history_ids = query.get("history_review_ids")
    if history_ids is None:
        user_id = int(query["user_id"])
        history_ids = [
            dataset_index.get((user_id, int(restaurant_id)))
            for restaurant_id in query["history_restaurant_ids"]
        ]
    target_id = query.get("target_review_id")
    if target_id is None:
        try:
            target_id = int(str(query["query_id"]).rsplit(":r", 1)[1])
        except (IndexError, ValueError):
            target_id = dataset_index.get(
                (int(query["user_id"]), int(query["target_restaurant_id"]))
            )
    return list(history_ids), int(target_id) if target_id is not None else None


def _enrich_rankings(
    rankings: list[dict[str, object]],
    queries: list[dict[str, object]],
    *,
    ranking_k: int,
) -> list[dict[str, object]]:
    """Attach offline labels and serving status to ranking records."""

    query_by_id = {row["query_id"]: row for row in queries}
    enriched: list[dict[str, object]] = []
    for ranking in rankings:
        query = query_by_id.get(ranking["query_id"])
        if query is None:
            continue
        final_rank = int(ranking["final_rank"])
        target_id = query["target_restaurant_id"]
        enriched.append(
            {
                **ranking,
                "target_restaurant_id": target_id,
                "target_restaurant_name": query["target_restaurant_name"],
                "target_rating": query["target_rating"],
                "history_restaurant_names": query["history_restaurant_names"],
                "is_target": ranking["restaurant_id"] == target_id,
                "served_top_k": final_rank <= ranking_k,
                "rank_movement": int(ranking["candidate_rank"]) - final_rank,
            }
        )
    return enriched


def _item_summary_rows(
    rankings: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarise an item's candidate and final-ranking exposure."""

    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rankings:
        key = (int(row["restaurant_id"]), str(row["restaurant_name"]))
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (restaurant_id, restaurant_name), rows in grouped.items():
        served = [row for row in rows if row["served_top_k"]]
        summaries.append(
            {
                "restaurant_id": restaurant_id,
                "restaurant_name": restaurant_name,
                "served_queries": len(served),
                "unique_users_served": len({row["user_id"] for row in served}),
                "target_matches": sum(bool(row["is_target"]) for row in served),
                "average_final_rank": (
                    sum(int(row["final_rank"]) for row in served) / len(served)
                    if served
                    else None
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -int(row["served_queries"]),
            int(row["restaurant_id"]),
        ),
    )


def _format_rank(value: object | None) -> str:
    return "후보 밖" if value is None else f"#{int(value)}"


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "The dashboard requires pandas and Streamlit from the experiment extra"
        ) from exc

    st.set_page_config(page_title="Rating Recsys Explorer", layout="wide")
    st.title("Rating Recsys · Candidate → LambdaRank Explorer")

    @st.cache_data(show_spinner=False)
    def load_jsonl(path: str, modified_ns: int) -> list[dict[str, object]]:
        del modified_ns
        return _read_jsonl(Path(path))

    @st.cache_data(show_spinner=False)
    def load_query_rows(
        path: str, modified_ns: int, query_id: str
    ) -> list[dict[str, object]]:
        del modified_ns
        return _read_detail_query(Path(path), query_id)

    def cached_jsonl(path: Path) -> list[dict[str, object]]:
        return load_jsonl(str(path), path.stat().st_mtime_ns)

    artifacts_root = Path(os.getenv("RATING_RECSYS_ARTIFACTS_DIR", "artifacts"))
    run_dirs = _discover_run_dirs(artifacts_root)
    if not run_dirs:
        st.info("No experiment runs found. Run `rating-recsys-experiment` first.")
        return

    run_options = {}
    for path in run_dirs:
        run_manifest = _read_json(path / "manifest.json")
        mode = run_manifest["config"]["region_mode"]
        label = "No region" if mode == "without_region" else "With region"
        run_options[f"{path.name} · {label}"] = path
    selected_name = st.sidebar.selectbox("Run", list(run_options))
    run_dir = run_options[selected_name]
    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics.json")

    st.subheader("Run summary")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Snapshot", manifest["snapshot"]["dataset_snapshot_id"][:12])
    summary_columns[1].metric("Interactions", manifest["snapshot"]["interactions"])
    summary_columns[2].metric("Users", manifest["snapshot"]["users"])
    summary_columns[3].metric("Restaurants", manifest["snapshot"]["restaurants"])
    with st.expander("Run configuration and code"):
        st.json({"config": manifest["config"], "code": manifest["code"]})

    phase = st.radio("Phase", ("validation", "test"), horizontal=True)
    ranking_k = int(manifest["config"]["ranking_k"])
    queries = [
        row for row in cached_jsonl(run_dir / "queries.jsonl") if row["phase"] == phase
    ]
    review_context_path = run_dir / "review_context.jsonl"
    review_context_available = review_context_path.exists()
    review_text_by_id = (
        {
            int(row["review_id"]): row.get("review_text")
            for row in cached_jsonl(review_context_path)
        }
        if review_context_path.exists()
        else {}
    )
    needs_dataset_review_index = any(
        "history_review_ids" not in query or "target_review_id" not in query
        for query in queries
    )
    dataset_review_index = (
        _dataset_review_index(cached_jsonl(run_dir / "dataset.jsonl"))
        if needs_dataset_review_index
        else {}
    )
    recommendations_path = run_dir / f"recommendations_{phase}.jsonl"
    recommendations = cached_jsonl(recommendations_path)
    enriched_recommendations = _enrich_rankings(
        recommendations, queries, ranking_k=ranking_k
    )

    if not queries:
        st.warning(f"No {phase} queries in this run")
        return

    st.subheader("Metrics by purpose and cutoff")
    phase_metrics = metrics[phase]
    cutoff_frame = pd.DataFrame(
        _metric_cutoff_rows(
            phase_metrics, region_mode=manifest["config"]["region_mode"]
        )
    )
    quality_tab, discovery_tab, operations_tab = st.tabs(
        ("순위 품질", "Coverage · discovery", "평가 모수 · latency")
    )

    with quality_tab:
        quality_columns = {
            "stage": st.column_config.TextColumn("Stage"),
            "K": st.column_config.NumberColumn(
                "@K", help="추천 목록의 상위 K개까지만 평가합니다.", format="%d"
            ),
            "recall_hit_rate": st.column_config.NumberColumn(
                "Recall / Hit Rate",
                help=METRIC_HELP["recall"],
                format="%.4f",
            ),
            "ndcg": st.column_config.NumberColumn(
                "NDCG", help=METRIC_HELP["ndcg"], format="%.4f"
            ),
            "mrr": st.column_config.NumberColumn(
                "MRR", help=METRIC_HELP["mrr"], format="%.4f"
            ),
        }
        st.dataframe(
            cutoff_frame[["stage", "K", "recall_hit_rate", "ndcg", "mrr"]],
            width="stretch",
            hide_index=True,
            column_config=quality_columns,
        )
        quality_charts = st.columns(3)
        for column, metric_key, label in zip(
            quality_charts,
            ("recall_hit_rate", "ndcg", "mrr"),
            ("Recall / Hit Rate@K", "NDCG@K", "MRR@K"),
            strict=True,
        ):
            column.markdown(f"**{label}**", help=METRIC_HELP[metric_key.split("_")[0]])
            pivot = cutoff_frame.pivot(index="K", columns="stage", values=metric_key)
            column.line_chart(pivot)

    with discovery_tab:
        discovery_columns = {
            "stage": st.column_config.TextColumn("Stage"),
            "K": st.column_config.NumberColumn(
                "@K", help="추천 목록의 상위 K개까지만 평가합니다.", format="%d"
            ),
            "catalog_coverage": st.column_config.NumberColumn(
                "Catalog coverage",
                help=METRIC_HELP["catalog_coverage"],
                format="%.4f",
            ),
            "novelty": st.column_config.NumberColumn(
                "Novelty", help=METRIC_HELP["novelty"], format="%.3f"
            ),
            "region_diversity": st.column_config.NumberColumn(
                "Region diversity",
                help=METRIC_HELP["region_diversity"],
                format="%.4f",
            ),
        }
        st.dataframe(
            cutoff_frame[
                [
                    "stage",
                    "K",
                    "catalog_coverage",
                    "novelty",
                    "region_diversity",
                ]
            ],
            width="stretch",
            hide_index=True,
            column_config=discovery_columns,
        )
        discovery_charts = st.columns(3)
        for column, metric_key, label, help_key in zip(
            discovery_charts,
            ("catalog_coverage", "novelty", "region_diversity"),
            ("Catalog coverage@K", "Novelty@K", "Region diversity@K"),
            ("catalog_coverage", "novelty", "region_diversity"),
            strict=True,
        ):
            column.markdown(f"**{label}**", help=METRIC_HELP[help_key])
            pivot = cutoff_frame.pivot(index="K", columns="stage", values=metric_key)
            column.line_chart(pivot)

    with operations_tab:
        final_metrics = phase_metrics["r1_lambdarank"]
        availability = phase_metrics["target_availability"]
        population_columns = st.columns(3)
        population_columns[0].metric(
            "Evaluated queries",
            int(final_metrics["evaluated_queries"]),
            help=METRIC_HELP["evaluated_queries"],
        )
        population_columns[1].metric(
            "Zero-relevance rate",
            f"{float(final_metrics['zero_relevance_query_rate']):.2%}",
            help=METRIC_HELP["zero_relevance_rate"],
        )
        population_columns[2].metric(
            "Target availability",
            f"{float(availability['rate']):.2%}",
            help=METRIC_HELP["target_availability"],
        )
        latency_columns = st.columns(2)
        latency_columns[0].metric(
            "Retrieval p95",
            f"{float(phase_metrics['retrieval_latency']['p95_ms']):.2f} ms",
            help=METRIC_HELP["retrieval_p95"],
        )
        latency_columns[1].metric(
            "Ranking p95",
            f"{float(phase_metrics['ranking_latency']['p95_ms']):.2f} ms",
            help=METRIC_HELP["ranking_p95"],
        )
        source_contribution = phase_metrics["c3_rrf_union"].get(
            "source_contribution"
        )
        if source_contribution:
            with st.expander("Candidate source contribution"):
                st.dataframe(
                    pd.DataFrame.from_dict(source_contribution, orient="index")
                    .rename_axis("source")
                    .reset_index(),
                    width="stretch",
                    hide_index=True,
                )

    user_tab, item_tab = st.tabs(("사용자별 추천 검토", "아이템별 추천 검토"))

    with user_tab:
        user_ids = sorted({int(row["user_id"]) for row in queries})
        selected_user_id = st.selectbox(
            "User ID",
            user_ids,
            format_func=lambda value: f"u{value}",
            key="user_inspection_user",
        )
        user_queries = [
            row for row in queries if int(row["user_id"]) == selected_user_id
        ]
        query_labels = {
            f"{row['query_id']} · label: {row['target_restaurant_name']}": row[
                "query_id"
            ]
            for row in user_queries
        }
        selected_label = st.selectbox(
            "Evaluation query", list(query_labels), key="user_inspection_query"
        )
        query_id = query_labels[selected_label]
        query = next(row for row in user_queries if row["query_id"] == query_id)
        history_review_ids, target_review_id = _query_review_ids(
            query, dataset_review_index
        )
        query_recommendations = [
            row for row in enriched_recommendations if row["query_id"] == query_id
        ]
        query_recommendations.sort(key=lambda row: int(row["final_rank"]))

        target_id = query["target_restaurant_id"]
        top_prediction = query_recommendations[0] if query_recommendations else None
        target_recommendation = next(
            (
                row
                for row in query_recommendations
                if row["restaurant_id"] == target_id
            ),
            None,
        )

        st.info(
            "정답 label은 추천 시점 뒤 데이터에서 다음으로 관측된 리뷰 식당입니다. "
            "모델 예측은 이 label을 가린 상태에서 생성한 Top-K 목록입니다. 실제 "
            "서비스 노출 로그를 의미하지는 않습니다."
        )
        label_column, prediction_column = st.columns(2)
        label_column.metric(
            "정답 label · 다음 관측 식당",
            str(query["target_restaurant_name"]),
            help=(
                "평가를 위해 숨겨둔 다음 interaction입니다. 정확히는 다음 방문 전체가 "
                "아니라 데이터에서 다음으로 기록된 리뷰 식당입니다."
            ),
        )
        prediction_column.metric(
            "모델 prediction · Top-1",
            str(top_prediction["restaurant_name"]) if top_prediction else "예측 없음",
            help="정답 label을 보지 않고 모델이 가장 높은 순위로 예측한 식당입니다.",
        )

        overview = st.columns(3)
        overview[0].metric(
            "History interactions",
            len(query["history_restaurant_ids"]),
            help="prediction 시점 이전에 모델이 사용할 수 있었던 사용자 이력 수입니다.",
        )
        overview[1].metric(
            f"Hit@{ranking_k}",
            "Yes" if target_recommendation else "No",
            help="정답 label 식당이 모델의 예측 Top-K 안에 포함됐는지 나타냅니다.",
        )
        overview[2].metric(
            "Label rank in prediction",
            _format_rank(
                target_recommendation["final_rank"]
                if target_recommendation
                else None
            ).replace("후보 밖", "Top-K 밖"),
        )

        st.subheader("Prediction 이전 이력과 정답 label")
        history_rows = [
            {
                "order": index,
                "restaurant_id": restaurant_id,
                "restaurant_name": name,
                "review": (
                    review_text_by_id.get(review_id) or "작성된 리뷰 없음"
                    if review_context_available
                    else "이 run에는 review context artifact가 없음"
                ),
            }
            for index, (restaurant_id, name, review_id) in enumerate(
                zip(
                    query["history_restaurant_ids"],
                    query["history_restaurant_names"],
                    history_review_ids,
                    strict=True,
                ),
                start=1,
            )
        ]
        history_col, target_col = st.columns((2, 1))
        history_col.dataframe(
            pd.DataFrame(history_rows), width="stretch", hide_index=True
        )
        target_col.write(
            {
                "cutoff": query["cutoff"],
                "role": "정답 label · 다음 관측 interaction",
                "label_restaurant_id": query["target_restaurant_id"],
                "label_restaurant_name": query["target_restaurant_name"],
                "label_rating": query["target_rating"],
                "relevance": query["relevance"],
                "review": (
                    review_text_by_id.get(target_review_id) or "작성된 리뷰 없음"
                    if review_context_available
                    else "이 run에는 review context artifact가 없음"
                ),
            }
        )
        st.caption(
            "리뷰 본문은 추천 feature로 사용되지 않는 display-only context입니다."
        )

        st.subheader(f"모델 예측 추천 Top-{ranking_k} · offline prediction")
        served_rows = query_recommendations
        served_columns = [
            "final_rank",
            "restaurant_id",
            "restaurant_name",
            "region",
            "candidate_sources",
            "candidate_rank",
            "rank_movement",
            "ranking_score",
            "is_target",
        ]
        served_frame = pd.DataFrame(served_rows).reindex(columns=served_columns).rename(
            columns={
                "final_rank": "predicted_rank",
                "is_target": "matches_label",
            }
        )
        st.dataframe(
            served_frame,
            width="stretch",
            hide_index=True,
        )

        if served_rows:
            explanation_labels = {
                f"#{row['final_rank']} · {row['restaurant_name']} "
                f"(id={row['restaurant_id']})": row
                for row in served_rows
            }
            explanation_label = st.selectbox(
                "추천 근거값을 확인할 아이템",
                list(explanation_labels),
                key="user_explanation_item",
            )
            explanation = explanation_labels[explanation_label]
            st.write(
                {
                    "candidate_sources": explanation["candidate_sources"],
                    "source_ranks": explanation["source_ranks"],
                    "source_scores": explanation["source_scores"],
                    "candidate_rank": explanation["candidate_rank"],
                    "final_rank": explanation["final_rank"],
                    "ranking_score": explanation["ranking_score"],
                }
            )
            st.caption(
                "source score와 rank는 candidate 생성 근거입니다. ranking score의 "
                "항목별 기여도를 해석하려면 아래 상세 결과 또는 SHAP이 필요합니다."
            )

        show_details = st.toggle(
            "전체 candidate와 모델 feature 불러오기",
            value=False,
            help="큰 ranking artifact에서 현재 query만 읽으므로 처음에는 잠시 걸릴 수 있습니다.",
        )
        if show_details:
            parquet_path = run_dir / f"rankings_{phase}.parquet"
            rankings_path = (
                parquet_path
                if parquet_path.exists()
                else run_dir / f"rankings_{phase}.jsonl"
            )
            with st.spinner("현재 query의 ranking detail을 읽는 중입니다..."):
                detailed_rows = load_query_rows(
                    str(rankings_path), rankings_path.stat().st_mtime_ns, query_id
                )
            detailed_rows = _enrich_rankings(
                detailed_rows, queries, ranking_k=ranking_k
            )
            detailed_rows.sort(key=lambda row: int(row["final_rank"]))
            detailed_target = next(
                (row for row in detailed_rows if row["restaurant_id"] == target_id), None
            )
            detail_columns = st.columns(2)
            detail_columns[0].metric(
                "Label retrieved into candidates",
                "Yes" if detailed_target else "No",
            )
            detail_columns[1].metric(
                "Label rank among all candidates",
                _format_rank(detailed_target["final_rank"] if detailed_target else None),
            )
            st.dataframe(
                pd.DataFrame(detailed_rows), width="stretch", hide_index=True
            )

    with item_tab:
        summaries = _item_summary_rows(enriched_recommendations)
        if not summaries:
            st.info(f"No {phase} item rankings in this run")
        else:
            summary_by_id = {int(row["restaurant_id"]): row for row in summaries}
            selected_item_id = st.selectbox(
                "Restaurant ID or name",
                list(summary_by_id),
                format_func=lambda value: (
                    f"{summary_by_id[value]['restaurant_name']} · id={value} · "
                    f"예측 Top-{ranking_k} {summary_by_id[value]['served_queries']}회"
                ),
                key="item_inspection_item",
            )
            item_rows = [
                row
                for row in enriched_recommendations
                if int(row["restaurant_id"]) == selected_item_id
            ]
            item_rows.sort(
                key=lambda row: (
                    not bool(row["served_top_k"]),
                    int(row["final_rank"]),
                    int(row["user_id"]),
                )
            )
            served_item_rows = [row for row in item_rows if row["served_top_k"]]
            summary = summary_by_id[selected_item_id]
            item_overview = st.columns(4)
            item_overview[0].metric(
                f"Predicted Top-{ranking_k} appearances", summary["served_queries"]
            )
            item_overview[1].metric(
                "Users receiving this prediction", summary["unique_users_served"]
            )
            item_overview[2].metric(
                "Average final rank", f"{summary['average_final_rank']:.2f}"
            )
            item_overview[3].metric(
                "Matches with label", summary["target_matches"]
            )

            if served_item_rows:
                rank_counts = Counter(int(row["final_rank"]) for row in served_item_rows)
                rank_frame = pd.DataFrame(
                    [
                        {"final_rank": rank, "exposures": count}
                        for rank, count in sorted(rank_counts.items())
                    ]
                ).set_index("final_rank")
                st.subheader("모델 예측 Top-K 순위 분포")
                st.bar_chart(rank_frame)

            st.subheader("이 아이템이 모델 예측 Top-K에 포함된 사용자/query")
            item_columns = [
                "user_id",
                "query_id",
                "served_top_k",
                "final_rank",
                "candidate_rank",
                "rank_movement",
                "candidate_sources",
                "ranking_score",
                "is_target",
                "target_restaurant_name",
                "target_rating",
                "history_restaurant_names",
            ]
            item_frame = pd.DataFrame(item_rows).reindex(columns=item_columns).rename(
                columns={
                    "served_top_k": "in_predicted_top_k",
                    "final_rank": "predicted_rank",
                    "is_target": "matches_label",
                    "target_restaurant_name": "label_restaurant_name",
                    "target_rating": "label_rating",
                }
            )
            st.dataframe(
                item_frame,
                width="stretch",
                hide_index=True,
            )

    importance_path = run_dir / "feature_importance.json"
    if importance_path.exists():
        importance = _read_json(importance_path)
        importance_frame = pd.DataFrame(
            sorted(importance.items(), key=lambda pair: pair[1], reverse=True),
            columns=["feature", "importance"],
        )
        st.subheader("Global feature importance")
        st.bar_chart(importance_frame.set_index("feature"))


if __name__ == "__main__":
    main()
