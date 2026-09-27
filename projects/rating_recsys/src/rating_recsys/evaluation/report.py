"""Render ``report.md`` for one run from its ``manifest.json`` and ``metrics.json``.

The report is the single human-readable result of a run. Section 6 is left for
the author, so an existing report is never overwritten unless asked.
"""

from __future__ import annotations

from pathlib import Path

from rating_recsys.experiments.artifacts import read_json


CANDIDATE_LABELS = (
    ("c0_popularity", "C0 전체 인기"),
    ("c1_item_item", "C1 item-item"),
    ("c2_region_popularity", "C2 지역 인기"),
    ("c3_rrf_union", "C3 quota RRF"),
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
    snapshot = manifest["snapshot"]
    code = manifest["code"]
    r0, r1 = test["r0_candidate_order"], test["r1_lambdarank"]
    chosen_quota = selection["chosen_candidate_policy"]["base_quota_fraction"]
    final = selection["final_ranker_params"]

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
        f"- 후보 생성: C3 Recall@{ck} {_pct(test['c3_rrf_union'][f'recall_at_{ck}'])}"
        f" (C1 단독 {_pct(test['c1_item_item'][f'recall_at_{ck}'])})",
        f"- LTR: NDCG@{k} {_num(r0[f'ndcg_at_{k}'])} → {_num(r1[f'ndcg_at_{k}'])} "
        f"({_ci(boot.get(f'ndcg_at_{k}'), percent=False)}), Recall@{k} "
        f"{_pct(r0[f'recall_at_{k}'])} → {_pct(r1[f'recall_at_{k}'])} "
        f"({_ci(boot.get(f'recall_at_{k}'), percent=True)})",
        f"- Validation 선택: C3 quota {chosen_quota}, LambdaRank "
        f"num_leaves {final['num_leaves']} · min_child_samples "
        f"{final['min_child_samples']} · trees {final['n_estimators']}",
        "",
        "## 2. 데이터와 조건",
        "",
    ]
    lines += _data_lines(manifest, metrics)
    lines += ["", "## 3. Validation에서 고른 설정", ""]
    lines += _selection_lines(manifest, metrics)
    lines += [
        "",
        "## 4. 후보 생성 (test)",
        "",
        "Recall@K = 사용자별 (후보 K개 안의 정답 수 / 전체 정답 수)의 평균.",
        "",
    ]
    lines.append(_candidate_table(test))
    lines += [
        "",
        f"## 5. LTR 재정렬 (test, Top-{k})",
        "",
        "R0는 C3 후보 순서를 그대로 자른 목록, R1은 LambdaRank로 재정렬한 목록이다. "
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
        + ", ".join(f"`{key}` {config[key]}" for key in KEY_CONDITIONS)
    )
    changed = _changed_conditions(config)
    lines += ["", "기본값과 다른 조건: " + (", ".join(changed) if changed else "없음")]
    return lines


def _selection_lines(manifest: dict, metrics: dict) -> list[str]:
    config = manifest["config"]
    k = int(config["ranking_k"])
    ck = int(config["candidate_k"])
    selection = metrics["selection"]
    chosen_quota = selection["chosen_candidate_policy"]["base_quota_fraction"]
    chosen_ranker = selection["chosen_ranker"]
    validation = metrics["validation"]
    return [
        f"후보 quota: C3에서 C0+C1 순서로 먼저 채우는 후보 비율. 기준은 "
        f"{selection['rule']['candidate_policy']}.",
        "",
        _table(
            ["quota", f"Val C3 Recall@{ck}"],
            [
                [_bold(row["base_quota_fraction"], row["base_quota_fraction"] == chosen_quota),
                 _pct(row[f"recall_at_{ck}"])]
                for row in selection["candidate_policy_grid"]
            ],
        ),
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
        f"최종 모델: {selection['rule']['final_refit']}. Validation에서 선택한 구성의 "
        f"C3 Recall@{ck} {_pct(validation['c3_rrf_union'][f'recall_at_{ck}'])}, "
        f"R1 NDCG@{k} {_num(validation['r1_lambdarank'][f'ndcg_at_{k}'])}.",
    ]


def _candidate_table(test: dict) -> str:
    cutoffs = _cutoffs(test["c3_rrf_union"])
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
        _table(["지표", "R0 C3 순서", "R1 LambdaRank", "R1 − R0 (95% CI)"], rows,
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
    "train_fraction": "--train-fraction",
    "validation_fraction": "--validation-fraction",
    "quota_grid": "--quota-grid",
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
