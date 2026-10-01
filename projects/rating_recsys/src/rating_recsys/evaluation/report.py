"""Render ``report.md`` for one run from its ``manifest.json`` and ``metrics.json``.

The report is the single human-readable result of a run. Section 6 is left for
the author, so an existing report is never overwritten unless asked.
"""

from __future__ import annotations

from pathlib import Path

from rating_recsys.experiments.artifacts import read_json


STAGE1 = "c5_c1_lightgcn_rrf"
CANDIDATE_LABELS = (
    (STAGE1, "**C5 C1+LightGCN RRF (Stage 1)**"),
    ("c1_item_item", "C1 item-item"),
    ("c4_lightgcn", "C4 LightGCN"),
    ("c0_popularity", "참고 · C0 전체 인기"),
    ("c2_region_popularity", "참고 · C2 지역 인기"),
    ("c3_rrf_union", "참고 · C3 quota RRF (이전 기준선)"),
)
KEY_CONDITIONS = (
    "candidate_k",
    "ranking_k",
    "region_mode",
    "relevance_high_threshold",
    "relevance_low_threshold",
    "train_fraction",
    "validation_fraction",
    "random_seed",
    "ranker_training_mode",
    "ranker_label_mode",
)


def write_report(run_dir: Path, *, overwrite: bool = False) -> Path:
    path = run_dir / "report.md"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.write_text(
        render_report(read_json(run_dir / "manifest.json"), read_json(run_dir / "metrics.json")),
        encoding="utf-8",
    )
    return path


def render_report(manifest: dict, metrics: dict) -> str:
    config = manifest["config"]
    k = int(config["ranking_k"])
    ck = int(config["candidate_k"])
    test = metrics["test"]
    selection = metrics["selection"]
    boot = test.get("bootstrap_r1_minus_r0") or {}
    stage1_boot = (test.get("bootstrap_c5_minus_c3") or {}).get(f"recall_at_{ck}")
    snapshot = manifest["snapshot"]
    code = manifest["code"]
    r0, r1 = test["r0_candidate_order"], test["r1_lambdarank"]
    final = selection["final_ranker_params"]
    graph = metrics["lightgcn"]["config"]

    lines = [
        f"# 실험 보고서 · {manifest['run_id']}",
        "",
        f"- 실행: {manifest['created_at'][:16].replace('T', ' ')} UTC"
        + (f" · label `{manifest['label']}`" if manifest.get("label") else ""),
        f"- 데이터: snapshot `{snapshot['dataset_snapshot_id'][:16]}` · "
        f"{snapshot['interactions']:,} interactions · 사용자 {snapshot['users']:,} · "
        f"식당 {snapshot['restaurants']:,} · {snapshot['minimum_event_date']} ~ "
        f"{snapshot['maximum_event_date']}",
        f"- 코드: commit `{(code.get('git_commit') or 'unknown')[:10]}`"
        + (" (미커밋 변경 포함, `source.diff`)" if code.get("git_dirty") else ""),
        "",
        "## 1. 요약",
        "",
        f"- 후보 생성: C5 Recall@{ck} {_pct(test[STAGE1][f'recall_at_{ck}'])}"
        f" (C1 단독 {_pct(test['c1_item_item'][f'recall_at_{ck}'])}, "
        f"C4 LightGCN 단독 {_pct(test['c4_lightgcn'][f'recall_at_{ck}'])}; "
        f"참고 C3 {_pct(test['c3_rrf_union'][f'recall_at_{ck}'])}, C5 − C3 "
        f"{_ci(stage1_boot, percent=True)})",
        f"- LTR: NDCG@{k} {_num(r0[f'ndcg_at_{k}'])} → {_num(r1[f'ndcg_at_{k}'])} "
        f"({_ci(boot.get(f'ndcg_at_{k}'), percent=False)}), Recall@{k} "
        f"{_pct(r0[f'recall_at_{k}'])} → {_pct(r1[f'recall_at_{k}'])} "
        f"({_ci(boot.get(f'recall_at_{k}'), percent=True)})",
        f"- Validation 선택: LambdaRank num_leaves {final['num_leaves']} · "
        f"min_child_samples {final['min_child_samples']} · trees {final['n_estimators']}. "
        f"LightGCN은 고정 설정({graph['layers']}층 · {graph['dimension']}차원 · "
        f"L2 {graph['regularization']:g} · {graph['epochs']} epoch)",
        "",
        "## 2. 데이터와 조건",
        "",
    ]
    lines += _data_lines(manifest, metrics)
    lines += ["", "## 3. 설정 선택", ""]
    lines += _selection_lines(manifest, metrics)
    lines += [
        "",
        "## 4. 후보 생성 (test)",
        "",
        "Recall@K = 사용자별 (후보 K개 안의 정답 수 / 전체 정답 수)의 평균.",
        "",
    ]
    lines.append(_candidate_table(test))
    if stage1_boot:
        lines += [
            "",
            f"C5 − C3 Recall@{ck}: {_ci(stage1_boot, percent=True)} (paired bootstrap, "
            f"C5가 나은 사용자 {stage1_boot['wins']:,}명, 나쁜 사용자 "
            f"{stage1_boot['losses']:,}명). C0·C2·C3는 Stage 1에 결합하지 않고 비교용으로만 "
            "남긴다. C0 순위와 C2 지역 비율은 ranker feature로 쓴다.",
        ]
    lines += [
        "",
        f"## 5. LTR 재정렬 (test, Top-{k})",
        "",
        "R0는 C5 후보 순서를 그대로 자른 목록, R1은 LambdaRank로 재정렬한 목록이다. "
        f"CI는 같은 사용자끼리 짝지은 paired bootstrap {boot.get('samples', 0):,}회.",
        "",
    ]
    lines += _ltr_lines(test, k, boot)
    lines += [
        "",
        "## 6. 해석 (작성자 기입)",
        "",
        "- 결론:",
        "- 근거:",
        "- 한계:",
        "- 다음 실험:",
        "",
        "## 7. 재현",
        "",
        "```bash",
        "rating-recsys-experiment \\",
        f"  --snapshot {snapshot.get('path', 'artifacts/snapshots/<snapshot>.jsonl')}"
        + _changed_flags(config),
        "```",
        "",
        "같은 폴더의 `manifest.json`(조건·구간·환경), `metrics.json`(모든 수치), "
        "`queries_*.jsonl`·`recommendations_*.jsonl`(사용자별 정답과 Top-K), "
        "`model.txt`(최종 ranker)가 이 보고서의 원본이다.",
    ]
    return "\n".join(lines) + "\n"


def _data_lines(manifest: dict, metrics: dict) -> list[str]:
    split = manifest["split"]
    windows = manifest["windows"]
    counts = split["interactions"]
    cutoffs = split["cutoffs"]
    val, test = windows["validation"], windows["test"]
    val_avail = metrics["validation"]["positive_availability"]["rate"]
    test_avail = metrics["test"]["positive_availability"]["rate"]
    lines = [
        _table(
            ["구간", "기간", "interactions", "평가 사용자", "사용자당 정답", "정답이 catalog에 있음"],
            [
                ["Train", f"~ {cutoffs['train_through']}", f"{counts['train']:,}", "–", "–", "–"],
                [
                    "Validation",
                    f"{cutoffs['train_through']} 이후 ~ {cutoffs['validation_through']}",
                    f"{counts['validation']:,}",
                    f"{val['evaluated_users']:,}",
                    f"{val['mean_relevant_per_user']:.2f}",
                    _pct(val_avail),
                ],
                [
                    "Test",
                    f"{cutoffs['validation_through']} 이후",
                    f"{counts['test']:,}",
                    f"{test['evaluated_users']:,}",
                    f"{test['mean_relevant_per_user']:.2f}",
                    _pct(test_avail),
                ],
            ],
            ["---", "---", "---:", "---:", "---:", "---:"],
        ),
        "",
        "평가 사용자는 cutoff 이전 이력이 있고 window에 평점 "
        f"{manifest['config']['relevance_low_threshold']} 이상 방문이 있는 사용자다. "
        f"이력이 없는 사용자(validation {val['new_users_not_evaluated']:,}명, test "
        f"{test['new_users_not_evaluated']:,}명)는 평가하지 않는다. Ranker 학습 정답은 "
        f"튜닝 시 {manifest['leakage_checks']['tuning_training_targets_through']}, 최종 "
        f"재학습 시 {manifest['leakage_checks']['refit_training_targets_through']}까지다.",
        "",
    ]
    config = manifest["config"]
    lines.append(
        "주요 조건: "
        + ", ".join(f"`{key}` {config[key]}" for key in KEY_CONDITIONS if key in config)
    )
    changed = _changed_conditions(config)
    lines += ["", "기본값과 다른 조건: " + (", ".join(changed) if changed else "없음")]
    return lines


def _selection_lines(manifest: dict, metrics: dict) -> list[str]:
    config = manifest["config"]
    k = int(config["ranking_k"])
    ck = int(config["candidate_k"])
    selection = metrics["selection"]
    chosen_ranker = selection["chosen_ranker"]
    validation = metrics["validation"]
    graph = metrics["lightgcn"]
    settings = graph["config"]
    windows = graph["window_models"]
    checkpoints = graph["training_checkpoints"]
    training = metrics["training"]
    scored = sum(int(part["lightgcn_scored_queries"]) for part in training.values())
    queries = sum(int(part["queries"]) for part in training.values())
    fitted = [row for row in checkpoints if "fit_seconds" in row]
    groups = sum(int(part["usable_groups"]) for part in training.values())
    relevant = sum(int(part["relevant_queries"]) for part in training.values())
    injected = sum(int(part.get("injected_positive_queries", 0)) for part in training.values())
    group_line = (
        f"Ranker 학습 group은 정답이 그 query의 C5 후보 {ck}개 안에 있는 query만 쓴다"
        f"(정답을 후보에 끼워 넣지 않음): 정답 있는 학습 query {relevant:,}개 중 "
        f"{groups:,}개 ({_pct(groups / relevant if relevant else 0)})."
        if not injected
        else f"Ranker 학습 query 중 정답이 후보 밖에 있던 {injected:,}개는 정답을 끼워 넣었다"
        " (2026-09-30 이전 run 방식)."
    )
    mode = config.get("ranker_training_mode", "prefix")
    label_mode = config.get("ranker_label_mode", "relevance")
    extra = [
        f"LTR 학습 query: `{mode}`. 학습 label: `{label_mode}`. "
        "평가의 relevance 기준은 학습 label과 독립적으로 유지한다.",
    ]
    if label_mode == "rating":
        extra += [
            "원래 평점을 반점 단위 정수 index(평점 × 2)로 전달하고 gain은 원래 평점으로 설정한다. "
            "4·4.5·5점의 차이를 보존한다. 미관측 후보의 0은 실제 0점 평가가 아닌 약한 negative다.",
        ]
    if mode == "window":
        final_training = training["refit_all_windows"]
        extra += [
            f"최종 window 학습: group {final_training['usable_groups']:,}개 중 여러 positive가 검색된 group "
            f"{final_training['multi_positive_groups']:,}개, 서로 다른 관측 label이 있는 group "
            f"{final_training['distinct_observed_label_groups']:,}개, 실제 평점이 다른 관측 쌍 "
            f"{final_training['observed_preference_pairs']:,}개. 평균 검색 positive "
            f"{final_training['mean_retrieved_positives_per_group']:.2f}개/group.",
            "각 학습 window 시작 전까지 후보와 feature를 고정한다. 마지막 부분 window는 cutoff까지 자르고, "
            "최종 refit에서는 T2까지 전체 window를 재구성하여 중복 학습하지 않는다.",
        ]
    return extra + ["",
        f"C4 LightGCN: {settings['layers']}층 · {settings['dimension']}차원 · L2 "
        f"{settings['regularization']:g} · learning rate {settings['learning_rate']:g} · "
        f"batch {settings['batch_size']} · {settings['epochs']} epoch (config 고정값, 이 run에서 "
        "고르지 않음). Validation window 모델은 train "
        f"{windows['validation']['edges']:,} edges, test window 모델은 train + validation "
        f"{windows['test']['edges']:,} edges로 학습했다.",
        "",
        f"Ranker 학습 query는 {graph['checkpoint_months']}개월 단위 구간 시작일 이전 "
        f"interaction으로 다시 학습한 LightGCN을 쓴다. 모델 {len(fitted)}개 "
        f"(학습 {sum(float(row['fit_seconds']) for row in fitted):.0f}s), LightGCN 후보가 "
        f"있었던 학습 query {scored:,}/{queries:,} ({_pct(scored / queries if queries else 0)}). "
        "나머지는 그 시점 그래프에 없던 사용자라 C1 순서만 쓴다.",
        "",
        group_line,
        "",
        f"LambdaRank: 기준은 {selection['rule']['ranker']}. `{_reference_name(selection)}`은 "
        "early stopping 없이 고정 설정으로 학습한 비교 기준이다.",
        "",
        _table(
            ["설정", "leaves", "min_child", "trees", f"Val NDCG@{k}", f"Val Recall@{k}"],
            [
                [
                    _bold(row["name"], row["name"] == chosen_ranker),
                    str(row["num_leaves"]),
                    str(row["min_child_samples"]),
                    str(row["best_iteration"]),
                    _num(row[f"ndcg_at_{k}"]),
                    _pct(row[f"recall_at_{k}"]),
                ]
                for row in selection["ranker_grid"]
            ],
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        f"최종 모델: {selection['rule']['final_refit']}. Validation에서 C5 Recall@{ck} "
        f"{_pct(validation[STAGE1][f'recall_at_{ck}'])}, R1 NDCG@{k} "
        f"{_num(validation['r1_lambdarank'][f'ndcg_at_{k}'])}.",
    ]


def _candidate_table(test: dict) -> str:
    cutoffs = _cutoffs(test[STAGE1])
    return _table(
        ["후보"] + [f"Recall@{c}" for c in cutoffs],
        [
            [label] + [_pct(test[key][f"recall_at_{c}"]) for c in cutoffs]
            for key, label in CANDIDATE_LABELS
            if key in test
        ],
    )


def _ltr_lines(test: dict, k: int, boot: dict) -> list[str]:
    r0, r1 = test["r0_candidate_order"], test["r1_lambdarank"]
    rows = []
    for metric, label, percent in (
        ("recall", "Recall", True),
        ("precision", "Precision", True),
        ("ndcg", "NDCG", False),
        ("map", "MAP", False),
        ("mrr", "MRR", False),
    ):
        for cutoff in _cutoffs(r1, metric):
            key = f"{metric}_at_{cutoff}"
            formatter = _pct if percent else _num
            rows.append(
                [f"{label}@{cutoff}", formatter(r0[key]), formatter(r1[key]),
                 _ci(boot.get(key), percent=percent)]
            )
    for metric, label, formatter in (
        ("catalog_coverage", "Catalog coverage", _pct),
        ("novelty", "Novelty", lambda v: _num(v, 2)),
        ("intra_list_region_diversity", "지역 다양성", _pct),
    ):
        key = f"{metric}_at_{k}"
        if key in r1:
            rows.append([f"{label}@{k}", formatter(r0[key]), formatter(r1[key]), "–"])
    lines = [
        _table(["지표", "R0 C5 순서", "R1 LambdaRank", "R1 − R0 (95% CI)"], rows,
               ["---", "---:", "---:", "---"])
    ]
    ndcg = boot.get(f"ndcg_at_{k}")
    if ndcg:
        lines += [
            "",
            f"NDCG@{k} 기준 R1이 나은 사용자 {ndcg['wins']:,}명, 나쁜 사용자 "
            f"{ndcg['losses']:,}명, 같은 사용자 {ndcg['ties']:,}명.",
        ]
    return lines


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _cutoffs(values: dict, metric: str = "recall") -> list[int]:
    prefix = f"{metric}_at_"
    return sorted(
        int(key.removeprefix(prefix))
        for key in values
        if key.startswith(prefix) and key.removeprefix(prefix).isdigit()
    )


def _pct(value: object) -> str:
    return "–" if value is None else f"{float(value) * 100:.2f}%"


def _num(value: object, digits: int = 4) -> str:
    return "–" if value is None else f"{float(value):.{digits}f}"


def _ci(values: dict | None, *, percent: bool) -> str:
    if not values:
        return "–"
    low, high = values["ci95"]
    if percent:
        return f"{values['delta'] * 100:+.2f}%p [{low * 100:+.2f}, {high * 100:+.2f}]"
    return f"{values['delta']:+.4f} [{low:+.4f}, {high:+.4f}]"


def _bold(value: object, chosen: bool) -> str:
    return f"**{value}**" if chosen else str(value)


def _reference_name(selection: dict) -> str:
    return selection["ranker_grid"][0]["name"] if selection["ranker_grid"] else "–"


def _table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] + ["---:"] * (len(headers) - 1)
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _defaults() -> dict:
    from rating_recsys.experiments.config import ExperimentConfig

    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in ExperimentConfig().to_dict().items()
    }


def _changed_conditions(config: dict) -> list[str]:
    defaults = _defaults()
    return [
        f"`{key}` {config[key]} (기본 {defaults[key]})"
        for key in sorted(config)
        if key in defaults and key != "n_jobs" and config[key] != defaults[key]
    ]


FLAG_NAMES = {
    "candidate_k": "--candidate-k",
    "ranking_k": "--ranking-k",
    "rrf_constant": "--rrf-constant",
    "random_seed": "--seed",
    "region_mode": "--region-mode",
    "relevance_high_threshold": "--relevance-high",
    "relevance_low_threshold": "--relevance-low",
    "ranker_training_mode": "--ranker-training-mode",
    "ranker_label_mode": "--ranker-label-mode",
    "train_fraction": "--train-fraction",
    "validation_fraction": "--validation-fraction",
    "legacy_c3_quota": "--legacy-c3-quota",
    "lightgcn_layers": "--lightgcn-layers",
    "lightgcn_dimension": "--lightgcn-dimension",
    "lightgcn_epochs": "--lightgcn-epochs",
    "lightgcn_regularization": "--lightgcn-regularization",
    "lightgcn_learning_rate": "--lightgcn-learning-rate",
    "lightgcn_batch_size": "--lightgcn-batch-size",
    "lightgcn_checkpoint_months": "--lightgcn-checkpoint-months",
    "num_leaves_grid": "--num-leaves-grid",
    "min_child_samples_grid": "--min-child-samples-grid",
    "learning_rate": "--learning-rate",
    "max_estimators": "--max-estimators",
    "early_stopping_rounds": "--early-stopping-rounds",
    "bootstrap_samples": "--bootstrap-samples",
}


def _changed_flags(config: dict) -> str:
    defaults = _defaults()
    flags = []
    for key, flag in FLAG_NAMES.items():
        if key in config and config[key] != defaults.get(key):
            value = config[key]
            text = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
            flags.append(f" \\\n  {flag} {text}")
    return "".join(flags)
