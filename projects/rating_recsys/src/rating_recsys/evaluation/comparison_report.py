"""Render ``report.md`` and ``learning_curve.png`` for a candidate comparison run.

Everything is read from the run's ``manifest.json`` and ``metrics.json``
(model title, stage labels and fusions included), so any model registered in
``experiments.candidate_models`` gets the same report.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from rating_recsys.experiments.artifacts import read_json


STAGE1 = "c5_c1_lightgcn_rrf"
REFERENCE_STAGES = ("c1_item_item", "c4_lightgcn", "c0_popularity", "c2_region_popularity", "c3_rrf_union")
CURVE_FILE = "learning_curve.png"


def write_report(run_dir: Path, *, overwrite: bool = False) -> Path:
    path = run_dir / "report.md"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.write_text(
        render_report(
            read_json(run_dir / "manifest.json"),
            read_json(run_dir / "metrics.json"),
            has_plot=(run_dir / CURVE_FILE).exists(),
        ),
        encoding="utf-8",
    )
    return path


def write_learning_curve(run_dir: Path, manifest: dict, metrics: dict) -> Path | None:
    """Relative training loss, validation Recall and (if recorded) RMSE per epoch."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    k = int(manifest["config"]["candidate_k"])
    selection = metrics["selection"]
    grid, chosen = selection["grid"], selection["chosen_config"]
    validation = metrics["validation"]
    has_rmse = any("rmse" in p for row in grid for p in row["curve"])
    columns = 3 if has_rmse else 2
    figure, axes = plt.subplots(1, columns, figsize=(6 * columns, 4.5))
    loss_axis, recall_axis = axes[0], axes[1]
    for row in grid:
        style = {"linewidth": 2.4 if row["name"] == chosen else 1.2}
        curve = row["curve"]
        first = curve[0]["loss"] or 1.0
        loss_axis.plot(
            [p["epoch"] for p in curve], [p["loss"] / first for p in curve],
            label=row["name"], **style,
        )
        evaluated = [p for p in curve if f"recall_at_{k}" in p]
        recall_axis.plot(
            [p["epoch"] for p in evaluated],
            [100 * p[f"recall_at_{k}"] for p in evaluated],
            marker="o", markersize=3, label=row["name"], **style,
        )
        recall_axis.scatter(
            [row["best_epoch"]], [100 * row["best"][f"recall_at_{k}"]], marker="*", s=120, zorder=5
        )
        rated = [p for p in curve if "rmse" in p]
        if has_rmse and rated:
            axes[2].plot(
                [p["epoch"] for p in rated], [p["rmse"] for p in rated],
                marker="o", markersize=3, label=row["name"], **style,
            )
    for stage, name, style in ((STAGE1, "C5", "--"), ("c1_item_item", "C1", ":")):
        if stage not in validation:
            continue
        value = 100 * validation[stage][f"recall_at_{k}"]
        recall_axis.axhline(value, color="gray", linestyle=style, label=f"{name} {value:.2f}%")
    loss_axis.set(title="Train loss / epoch-1 loss", xlabel="epoch", ylabel="relative loss")
    recall_axis.set(title=f"Validation Recall@{k} (★ = best epoch)", xlabel="epoch", ylabel="%")
    if has_rmse:
        reference = metrics.get("references", {}).get("validation_mean_rating_rmse")
        if reference is not None:
            axes[2].axhline(reference, color="gray", linestyle="--", label=f"mean rating {reference:.3f}")
        axes[2].set(title="Validation window RMSE (rating objective)", xlabel="epoch", ylabel="RMSE")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle(f"{manifest['model']['title']} · {manifest['run_id']}", fontsize=9)
    figure.tight_layout()
    path = run_dir / CURVE_FILE
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def render_report(manifest: dict, metrics: dict, *, has_plot: bool = False) -> str:
    config = manifest["config"]
    k, rk = int(config["candidate_k"]), int(config["ranking_k"])
    model = manifest["model"]
    name, title = model["name"], model["title"]
    labels = manifest["stage_labels"]
    selection = metrics["selection"]
    validation, test = metrics["validation"], metrics["test"]
    comparisons = metrics["test_comparisons"]
    policy = selection["chosen_policy"]
    rows = selection["grid"]
    chosen = next(row for row in rows if row["name"] == selection["chosen_config"])
    refit = metrics["refit"]
    references = metrics.get("references") or {}
    snapshot, code = manifest["snapshot"], manifest["code"]
    texts = manifest.get("review_texts")
    alone_vs_c5 = comparisons["vs_stage1"][name][f"at_{k}"].get(f"recall_at_{k}")
    policy_vs_c5 = comparisons["vs_stage1"][policy][f"at_{k}"].get(f"recall_at_{k}")
    cutoffs = _cutoffs(test[STAGE1])
    diff_note = (
        " (미커밋 변경 포함, `source.diff`)"
        if code.get("git_diff_artifact")
        else " (미커밋 파일 포함, 목록은 manifest.json `code.git_status`)"
        if code.get("git_dirty")
        else ""
    )
    data_line = (
        f"- 데이터: snapshot `{snapshot['dataset_snapshot_id'][:16]}` · "
        f"{snapshot['interactions']:,} interactions · 사용자 {snapshot['users']:,} · "
        f"식당 {snapshot['restaurants']:,}"
    )
    if texts:
        data_line += (
            f" · 리뷰 본문 `{texts.get('artifact', '–')}` "
            f"({texts.get('non_empty', 0):,}건 비어 있지 않음)"
        )

    lines = [
        f"# {title} 후보 실험 · {manifest['run_id']}",
        "",
        f"- 실행: {manifest['created_at'][:16].replace('T', ' ')} UTC"
        + (f" · label `{manifest['label']}`" if manifest.get("label") else ""),
        data_line,
        f"- 코드: commit `{(code.get('git_commit') or 'unknown')[:10]}`" + diff_note,
        "- 범위: Stage 1 후보 생성만 비교한다. Stage 2 ranker(R1)는 다시 학습하지 않았다.",
        "",
        "## 1. 요약",
        "",
        f"- 학습: `{chosen['name']}`, validation 최고 epoch {chosen['best_epoch']} "
        f"(Recall@{k} {_pct(chosen['best'][f'recall_at_{k}'])}, 같은 window의 C5 "
        f"{_pct(validation[STAGE1][f'recall_at_{k}'])}). Test용 재학습 "
        f"{refit['training_interactions']:,} interactions · {refit['fit_seconds']:.0f}s",
        f"- Test Recall@{k}: {title} 단독 {_pct(test[name][f'recall_at_{k}'])} vs C5 "
        f"{_pct(test[STAGE1][f'recall_at_{k}'])} ({_ci(alone_vs_c5, percent=True)})",
        f"- Validation으로 고른 결합: {labels[policy]} → test Recall@{k} "
        f"{_pct(test[policy][f'recall_at_{k}'])} ({_ci(policy_vs_c5, percent=True)} vs C5)",
    ]
    if manifest.get("evidence_status") == "exploratory-reused-holdout":
        lines += ["- Evidence: exploratory only. This historical test was seen in earlier experiments; bootstrap does not undo holdout reuse."]
    rmse = [float(p["rmse"]) for row in rows for p in row["curve"] if "rmse" in p]
    if rmse and "validation_mean_rating_rmse" in references:
        lines.append(
            f"- 평점 예측: validation window RMSE 최저 {min(rmse):.4f} vs 학습 평균 평점 "
            f"{references['validation_mean_rating_rmse']:.4f}"
        )

    windows, split = manifest["windows"], manifest["split"]
    lines += [
        "",
        "## 2. 데이터와 조건",
        "",
        "Split, 평가 query, C0~C5는 `rating-recsys-experiment`와 같다(C4 LightGCN은 config "
        "고정 설정).",
        "",
        "| 구간 | 모델 입력 (학습 interaction) | 평가 사용자 | 사용자당 정답 |",
        "|---|---|---:|---:|",
        f"| Validation | ≤ {split['cutoffs']['train_through']} "
        f"({chosen['training_interactions']:,}) | {windows['validation']['evaluated_users']:,} | "
        f"{windows['validation']['mean_relevant_per_user']:.2f} |",
        f"| Test | ≤ {split['cutoffs']['validation_through']} "
        f"({refit['training_interactions']:,}) | {windows['test']['evaluated_users']:,} | "
        f"{windows['test']['mean_relevant_per_user']:.2f} |",
        "",
        *[f"- {line}" for line in model.get("description", [])],
        f"- 결합: 각 source의 Top-{k}를 RRF 1/({config['rrf_constant']} + rank)로 합해 "
        f"Top-{k}. Quota 없음.",
        f"- 선택 기준: {selection['rule']['epochs']}. 설정은 {selection['rule']['config']}. "
        f"결합은 {selection['rule']['policy']}.",
        "",
        "## 3. 학습 곡선 (validation)",
        "",
    ]
    if chosen.get("profile_preprocessing"):
        position = lines.index("## 3. 학습 곡선 (validation)")
        lines[position:position] = _embedding_notes(chosen, refit)
    if chosen.get("retriever_preprocessing"):
        position = lines.index("## 3. 학습 곡선 (validation)")
        lines[position:position] = [
            "### 로컬 BM25 색인과 프로필", "",
            "| 구간 | 검색 사용자 | query 토큰 있음 | query 토큰 없음 | 색인 식당 | 색인 토큰 | 어휘 수 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[f"| {phase} | {row['input_users']:,} | {row['users_with_query_tokens']:,} | "
              f"{row['users_without_query_tokens']:,} | {row['indexed_restaurants']:,} | "
              f"{row['indexed_tokens']:,} | {row['vocabulary_size']:,} |"
              for phase, row in (("Validation", chosen), ("Test", refit))],
            "", "학습 loss나 반복 학습은 없다. epoch 1은 cutoff별 색인 생성 한 번을 나타낸다. "
            "본문·색인 토큰 hash와 전처리는 metrics.json에 기록했다. "
            "Test 색인·식당 ID 대응·사용자 query 토큰은 bm25_index_test/에 저장했다.", "",
        ]
    if has_plot:
        lines += [f"![learning curve]({CURVE_FILE})", ""]
    has_rmse = any("rmse" in p for p in chosen["curve"])
    lines += [
        f"선택한 `{chosen['name']}`의 평가 지점. Loss는 {model.get('loss_description', 'epoch 평균 학습 loss')}.",
        "",
        f"| epoch | loss | Recall@20 | Recall@{k} | NDCG@{rk} | Coverage@{k} | Novelty@{k} |"
        + (" RMSE |" if has_rmse else ""),
        "|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if has_rmse else ""),
    ]
    for point in _thin([p for p in chosen["curve"] if f"recall_at_{k}" in p], 24):
        mark = "**" if point["epoch"] == chosen["best_epoch"] else ""
        lines.append(
            f"| {mark}{point['epoch']}{mark} | {point['loss']:.4f} | "
            f"{_pct(point.get('recall_at_20'))} | {_pct(point[f'recall_at_{k}'])} | "
            f"{_num(point.get(f'ndcg_at_{rk}'))} | {_pct(point.get(f'catalog_coverage_at_{k}'))} | "
            f"{_float(point.get(f'novelty_at_{k}'))} |"
            + (f" {_num(point.get('rmse'))} |" if has_rmse else "")
        )

    any_rmse = bool(rmse)
    lines += [
        "",
        "## 4. Validation 선택",
        "",
        f"| 설정 | best epoch | 중단 epoch | Val Recall@20 | Val Recall@{k} | Val NDCG@{rk} |"
        + (" 최저 RMSE |" if any_rmse else "") + " 학습 시간 |",
        "|---|---:|---:|---:|---:|---:|" + ("---:|" if any_rmse else "") + "---:|",
    ]
    for row in rows:
        bold = "**" if row["name"] == chosen["name"] else ""
        row_rmse = [float(p["rmse"]) for p in row["curve"] if "rmse" in p]
        lines.append(
            f"| {bold}{row['name']}{bold} | {row['best_epoch']} | {row['stopped_epoch']} | "
            f"{_pct(row['best'].get('recall_at_20'))} | {_pct(row['best'][f'recall_at_{k}'])} | "
            f"{_num(row['best'].get(f'ndcg_at_{rk}'))} |"
            + (f" {_num(min(row_rmse)) if row_rmse else '–'} |" if any_rmse else "")
            + f" {row['fit_seconds']:.0f}s |"
        )
    if "validation_mean_rating_rmse" in references:
        lines += [
            "",
            f"평균 평점으로 모두 예측할 때 RMSE: validation "
            f"{references['validation_mean_rating_rmse']:.4f}, test "
            f"{references['test_mean_rating_rmse']:.4f}.",
        ]

    stages = [STAGE1, *REFERENCE_STAGES[:2], name, *manifest["fusions"], *REFERENCE_STAGES[2:]]
    stages = [stage for stage in stages if stage in test]
    lines += [
        "",
        "## 5. 후보 결과",
        "",
        "Recall@K = 사용자별 (후보 K개 안의 정답 수 / 전체 정답 수)의 평균. "
        f"@{rk}는 후보 순서를 그대로 자른 값이다.",
        "",
        f"| 후보 | Val Recall@{k} | "
        + " | ".join(f"Test Recall@{c}" for c in cutoffs)
        + f" | Test NDCG@{rk} | Coverage@{k} | Novelty@{k} |",
        "|---|---:|" + "---:|" * len(cutoffs) + "---:|---:|---:|",
    ]
    for stage in stages:
        bold = "**" if stage == policy else ""
        row = test[stage]
        lines.append(
            f"| {bold}{labels[stage]}{bold} | {_pct(validation[stage][f'recall_at_{k}'])} | "
            + " | ".join(_pct(row[f"recall_at_{c}"]) for c in cutoffs)
            + f" | {_num(row[f'ndcg_at_{rk}'])} | {_pct(row[f'catalog_coverage_at_{k}'])} | "
            f"{_float(row.get(f'novelty_at_{k}'))} |"
        )

    lines += ["", f"후보 순서의 Top-{rk} 추가 지표 (재랭킹 없음):", "",
              f"| 후보 | Precision@{rk} | MRR@{rk} | MAP@{rk} |",
              "|---|---:|---:|---:|"]
    for stage in (STAGE1, name, *manifest["fusions"]):
        row = test[stage]
        lines.append(f"| {labels[stage]} | {_pct(row[f'precision_at_{rk}'])} | "
                     f"{_num(row[f'mrr_at_{rk}'])} | {_num(row[f'map_at_{rk}'])} |")
    pairs = metrics.get("observed_pair_diagnostics", {}).get("test")
    if pairs:
        lines += ["", "### 관측 만족도 쌍 순서 정확도 · 보조 진단", "",
                  f"실제 미래 방문 중 만족도 등급이 다른 두 식당이 모두 Top-{k} 후보에 있을 때만 "
                  "비교한다. 사용자별 정확도의 평균이며, 미검색·미방문 식당의 순서는 추정하지 않는다. "
                  "설정·결합 선택에는 사용하지 않는다. 각 후보의 평가 가능한 사용자 집합이 다르므로 "
                  "이 수치만으로 후보 전체의 우열을 판단하지 않는다.", "",
                  "| 후보 | 쌍 순서 정확도 | 비교 사용자 | 비교 쌍 | 올바른 쌍 |",
                  "|---|---:|---:|---:|---:|"]
        for stage, row in pairs.items():
            lines.append(f"| {labels[stage]} | {_pct(row['accuracy'])} | "
                         f"{row['evaluated_queries']:,} | {row['compared_pairs']:,} | "
                         f"{row['correct_pairs']:,} |")

    lines += [
        "",
        "## 6. Test 비교 (paired bootstrap)",
        "",
        f"같은 사용자끼리 짝지은 bootstrap {config['bootstrap_samples']:,}회, 95% CI. "
        "CI가 0을 포함하면 개선·악화를 확정할 근거가 부족하다. 차이 없음의 증명은 아니다.",
        "",
        f"| 후보 − C5 | Recall@20 | Recall@{k} | 이긴/진 사용자 (@{k}) |",
        "|---|---|---|---:|",
    ]
    for stage, by_cutoff in comparisons["vs_stage1"].items():
        at_k = by_cutoff[f"at_{k}"].get(f"recall_at_{k}")
        at_20 = by_cutoff.get("at_20", {}).get("recall_at_20")
        record = f"{at_k['wins']:,} / {at_k['losses']:,}" if at_k else "–"
        lines.append(
            f"| {labels[stage]} | {_ci(at_20, percent=True)} | {_ci(at_k, percent=True)} | {record} |"
        )
    overlap = comparisons["positive_overlap_vs_stage1"]
    lines += [
        "",
        f"정답 방문 {overlap['relevant_positives']:,}건 중 Top-{k} 포함: 둘 다 {overlap['both']:,} · "
        f"{title}만 {overlap['left_only']:,} · C5만 {overlap['right_only']:,} · 둘 다 못 찾음 "
        f"{overlap['neither']:,}.",
    ]
    baseline = comparisons.get("vs_baseline_run")
    if baseline:
        lines += [
            "",
            f"기준 run `{baseline['run_id']}`의 최종 Top-{rk}와 비교 (같은 snapshot·split·test query):",
            "",
            f"| 목록 | Recall@{rk} | NDCG@{rk} | MRR@{rk} | Coverage@{rk} | − R1 NDCG@{rk} | − R1 Recall@{rk} |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for stage, row in baseline["metrics"].items():
            boot = baseline["bootstrap_vs_r1"].get(stage) or {}
            lines.append(
                f"| {labels[stage]} | {_pct(row[f'recall_at_{rk}'])} | {_num(row[f'ndcg_at_{rk}'])} | "
                f"{_num(row[f'mrr_at_{rk}'])} | {_pct(row[f'catalog_coverage_at_{rk}'])} | "
                f"{_ci(boot.get(f'ndcg_at_{rk}'), percent=False)} | "
                f"{_ci(boot.get(f'recall_at_{rk}'), percent=True)} |"
            )

    lines += [
        "",
        "## 7. 해석 (작성자 기입)",
        "",
        "- 결론:",
        "- 근거:",
        "- 한계:",
        "- 다음 실험:",
        "",
        "## 8. 재현",
        "",
        "```bash",
        _command(manifest),
        "```",
        "",
        "같은 폴더의 `manifest.json`(조건·grid·환경"
        + ("·리뷰 본문 파일 hash" if texts else "")
        + "), `metrics.json`(학습 곡선과 모든 수치), `candidates_test.jsonl`(사용자별 "
        f"{title}·선택 결합·C5 후보)이 이 보고서의 원본이다. MLflow에는 기록하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def _command(manifest: dict) -> str:
    command = manifest.get("command")
    if command:
        return "rating-recsys-compare " + " ".join(shlex.quote(part) for part in command)
    parts = [manifest["model"]["name"], "--snapshot", manifest["snapshot"]["path"]]
    if manifest.get("baseline_run"):
        parts += ["--baseline-run", manifest["baseline_run"]]
    return "rating-recsys-compare " + " ".join(shlex.quote(part) for part in parts)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _cutoffs(values: dict) -> list[int]:
    return sorted(
        int(key.removeprefix("recall_at_"))
        for key in values
        if key.startswith("recall_at_") and key.removeprefix("recall_at_").isdigit()
    )


def _thin(points: list[dict], limit: int) -> list[dict]:
    if len(points) <= limit:
        return points
    step = -(-len(points) // limit)
    keep = points[::step]
    return keep if keep[-1] is points[-1] else keep + [points[-1]]


def _pct(value) -> str:
    return "–" if value is None else f"{100 * float(value):.2f}%"


def _num(value) -> str:
    return "–" if value is None else f"{float(value):.4f}"


def _float(value) -> str:
    return "–" if value is None else f"{float(value):.2f}"


def _ci(entry, *, percent: bool) -> str:
    if not entry:
        return "–"
    low, high = entry["ci95"]
    if percent:
        return f"{100 * entry['delta']:+.2f}%p [{100 * low:+.2f}, {100 * high:+.2f}]"
    return f"{entry['delta']:+.4f} [{low:+.4f}, {high:+.4f}]"



def _embedding_notes(chosen: dict, refit: dict) -> list[str]:
    config = chosen["config"]
    lines = [
        "### 리뷰 임베딩 입력과 API",
        "",
        f"- 모델 `{config['model']}` · {config['dimensions']:,}차원 · "
        f"배치 최대 {config['batch_size']}개 · 동시 요청 최대 {config['concurrency']}개 · "
        f"분당 {config['requests_per_minute']:g}회. 성공한 배치를 SQLite에 즉시 저장한다.",
        f"- 집계 `{config.get('aggregation', 'concat')}` · 사용자 표현 `{config.get('user_profile', 'own_reviews')}`. "
        "own_reviews는 사용자 리뷰에 `query: `, 식당 리뷰에 `document: `를 붙인다. "
        "liked_items는 선택한 사용자 리뷰 이벤트의 식당 document 벡터를 평균한다.",
        "- concat은 본문을 묶은 전체 프로필을 토큰 예산까지 보존한다. review_mean은 "
        "리뷰별 벡터를 정규화·평균·정규화하며 토큰 예산은 각 리뷰에 적용한다. "
        "아래 잘림 비율의 단위는 concat=프로필, review_mean=개별 리뷰다.",
        "- 외부 encoder는 고정했다. epoch 1과 loss 0은 공통 비교 실행기의 "
        "프로필 생성 단계를 표시하며 모델 학습 loss가 아니다.",
        "",
        "| 구간 | 사용자 프로필 | 식당 프로필 | 잘린 API 입력 | 최대 입력 토큰 | "
        "새 API 요청 | 캐시 재사용 | 새 요청 토큰 | 새 요청 비용 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Validation", chosen), ("Test", refit)):
        prep, usage = row["profile_preprocessing"], row["openrouter_usage"]
        fraction = prep["truncated_profiles"] / prep["profiles"] if prep["profiles"] else 0
        lines.append(
            f"| {label} | {row['users_with_review_profiles']:,} | "
            f"{row['restaurants_with_review_profiles']:,} | {fraction:.2%} | "
            f"{prep['max_sent_tokens']} | {usage['api_requests']} | {usage['cache_hits']:,} | "
            f"{usage['prompt_tokens']:,} | ${usage['cost_usd']:.6f} |"
        )
    total = refit.get("embedding_cache_totals", {})
    lines += [
        "",
        f"- 동일 모델 캐시의 누적 성공 요청 {total.get('successful_requests', 0):,}회 · "
        f"입력 {total.get('documents', 0):,}개 · {total.get('prompt_tokens', 0):,}토큰 · "
        f"${total.get('cost_usd', 0):.6f}. 사전 API 검증 호출도 포함한다.",
        "- 토크나이저 revision·파일 SHA-256·프로필 버전·전체 설정과 API 사용량은 "
        "`manifest.json`과 `metrics.json`에 기록한다.",
        "",
    ]
    return lines
