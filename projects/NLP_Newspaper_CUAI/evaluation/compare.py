"""Multi-system paired evaluation comparing C0 LLM outputs with Gemma 3-1B and synthetic references.

Run from ``projects/NLP_Newspaper_CUAI``:
    python -m evaluation.compare --pred-dir evaluation/runs/c0_<model> --out evaluation/C0_REPORT.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import judge as judge_mod
from . import korean, lexical, neural, report, rules

PROJECT = Path(__file__).resolve().parent.parent
COLUMNS = ("original", "simplified_human", "simplified_by_model")
SYSTEM_LABELS = {
    "c0": "C0 (LLM Single-shot)",
    "model": "Model (Gemma 3-1B)",
    "reference": "Synthetic reference",
    "copy": "Copy source",
}


def _load_predictions(pred_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = pred_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pred_jsonl = pred_dir / "predictions.jsonl"
    if not pred_jsonl.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl in {pred_dir}")

    preds: dict[int, dict[str, Any]] = {}
    target_model = manifest.get("model")
    target_prompt_sha = manifest.get("prompt_sha256")
    target_temp = manifest.get("temperature")

    fallback_preds: dict[int, dict[str, Any]] = {}
    for line in pred_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("error") or not str(rec.get("simplified_c0", "")).strip():
            continue
        idx = int(rec["idx"])
        fallback_preds[idx] = rec
        if target_model is not None and rec.get("model") != target_model:
            continue
        if target_prompt_sha is not None and rec.get("prompt_sha256") != target_prompt_sha:
            continue
        if target_temp is not None and round(float(rec.get("temperature", -1)), 6) != round(float(target_temp), 6):
            continue
        preds[idx] = rec

    err_records: list[dict[str, Any]] = []
    err_jsonl = pred_dir / "errors.jsonl"
    if err_jsonl.exists():
        for line in err_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                err_records.append(json.loads(line))
            except ValueError:
                continue

    return (preds if preds else fallback_preds), manifest, err_records


def _safe_lexical_per_sample(
    sources: list[str], predictions: list[str], references: list[str]
) -> pd.DataFrame:
    try:
        return lexical.per_sample(sources, predictions, references)
    except ValueError:
        frames = []
        for s, p, r in zip(sources, predictions, references):
            try:
                frames.append(lexical.per_sample([s], [p], [r]))
            except ValueError:
                # rouge raises ValueError if prediction/reference has only punctuation.
                src_m = korean.morph_string(s)
                pred_m = korean.morph_string(p)
                ref_m = korean.morph_string(r)
                ws = lexical.sari_sentence(s, p, [r])
                mo = lexical.sari_sentence(src_m, pred_m, [ref_m])
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "sari_ws": ws["sari"],
                                "sari_morph": mo["sari"],
                                "sari_morph_keep": mo["sari_keep"],
                                "sari_morph_del": mo["sari_del"],
                                "sari_morph_add": mo["sari_add"],
                                "bleu_morph": lexical._BLEU.sentence_score(pred_m, [ref_m]).score,
                                "chrf_pp": lexical._CHRF.sentence_score(p, [r]).score,
                                "rouge_l_ws": 0.0,
                                "rouge_l_morph": lexical.rouge_l_f1(pred_m.split(), ref_m.split()),
                            }
                        ]
                    )
                )
        return pd.concat(frames, ignore_index=True)


def _long(values: dict[str, dict[str, np.ndarray | list]], n: int) -> pd.DataFrame:
    frames = []
    for system, cols in values.items():
        f = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})
        f.insert(0, "idx", np.arange(n))
        f.insert(1, "system", system)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


class _CompareTable:
    def __init__(self, summary: pd.DataFrame, per: pd.DataFrame):
        self.s = summary.set_index(["system", "metric"])
        self.per = per

    def cell(self, system: str, metric: str, ci: bool = True, **kw: Any) -> str:
        if (system, metric) not in self.s.index:
            return "—"
        r = self.s.loc[(system, metric)]
        return report._fmt(r["mean"], r["lo"] if ci else None, r["hi"] if ci else None, **kw)

    def render(
        self,
        systems: list[str],
        rows: list[tuple[str, str, dict[str, Any]]],
        diffs: list[tuple[str, str]] | None = None,
    ) -> str:
        diffs = diffs or []
        head = "| Metric | " + " | ".join(SYSTEM_LABELS[s] for s in systems)
        for a, b in diffs:
            head += f" | {SYSTEM_LABELS[a]} − {SYSTEM_LABELS[b]}"
        lines = [head + " |", "|---" * (len(systems) + 1 + len(diffs)) + "|"]
        for label, metric, kw in rows:
            cells = [self.cell(s, metric, **kw) for s in systems]
            for a, b in diffs:
                d = report.paired_diff(self.per, metric, a, b)
                kw2 = {k: v for k, v in kw.items() if k != "ci"}
                if kw2.get("pct"):
                    kw2.update(pct=False, unit=" pp")
                cells.append(report._fmt(*d, **kw2) if not np.isnan(d[0]) else "—")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _fmt_num_set(vals: set[float]) -> str:
    if not vals:
        return "없음"
    formatted = []
    for v in sorted(vals):
        formatted.append(str(int(v)) if v == int(v) else f"{v:g}")
    return ", ".join(formatted)


def _operational_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [float(r["latency_sec"]) for r in records if isinstance(r.get("latency_sec"), (int, float))]
    p_tok = [int(r["prompt_tokens"]) for r in records if isinstance(r.get("prompt_tokens"), (int, float))]
    c_tok = [int(r["completion_tokens"]) for r in records if isinstance(r.get("completion_tokens"), (int, float))]
    t_tok = [int(r["total_tokens"]) for r in records if isinstance(r.get("total_tokens"), (int, float))]
    costs = [float(r["cost_usd"]) for r in records if isinstance(r.get("cost_usd"), (int, float))]
    attempts = [int(r["attempts"]) for r in records if isinstance(r.get("attempts"), (int, float))]
    thinking = sum(1 for r in records if r.get("had_thinking"))
    return {
        "n": len(records),
        "thinking_count": thinking,
        "thinking_rate": thinking / len(records) if records else 0.0,
        "mean_attempts": float(np.mean(attempts)) if attempts else None,
        "retry_count": sum(1 for a in attempts if a > 1),
        "latency_mean": float(np.mean(lat)) if lat else None,
        "latency_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95": float(np.percentile(lat, 95)) if lat else None,
        "prompt_tokens_total": sum(p_tok) if p_tok else None,
        "prompt_tokens_mean": float(np.mean(p_tok)) if p_tok else None,
        "completion_tokens_total": sum(c_tok) if c_tok else None,
        "completion_tokens_mean": float(np.mean(c_tok)) if c_tok else None,
        "total_tokens_total": sum(t_tok) if t_tok else None,
        "total_tokens_mean": float(np.mean(t_tok)) if t_tok else None,
        "total_cost_usd": sum(costs) if costs else None,
    }


def _build_c0_report(
    *,
    matched_indices: list[int],
    target_indices: list[int],
    missing_indices: list[int],
    raw_df: pd.DataFrame,
    preds: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    err_records: list[dict[str, Any]],
    per: pd.DataFrame,
    corpus: dict[str, float],
    judge_df: pd.DataFrame | None,
    split_payload: dict[str, Any],
    split_name: str,
    stress_indices: list[int],
    out_path: Path,
    command: str,
    skip_neural: bool,
    seed: int,
) -> str:
    summary = report.summarize(per, seed=seed)
    t = _CompareTable(summary, per)
    c0_per = per[per.system == "c0"].set_index("idx")
    model_per = per[per.system == "model"].set_index("idx")

    matched_records = [preds[i] for i in matched_indices]
    op = _operational_stats(matched_records)
    model_id = manifest.get("model") or matched_records[0].get("model", "unknown")
    provider = manifest.get("provider") or matched_records[0].get("provider", "unknown")
    prompt_ver = manifest.get("prompt_version") or matched_records[0].get("prompt_version", "unknown")
    prompt_sha = manifest.get("prompt_sha256") or matched_records[0].get("prompt_sha256", "")
    temp = manifest.get("temperature", matched_records[0].get("temperature", ""))

    def diff_str(metric: str, a: str, b: str, **kw: Any) -> str:
        d = report.paired_diff(per, metric, a, b, seed=seed)
        return report._fmt(*d, **kw)

    pct_diff_kw = {"scale": 100, "digits": 1, "unit": " pp"}
    lines = [
        "# C0 Single-shot LLM vs Fine-tuned Baseline Evaluation",
        "",
        (
            "**규칙·어휘 지표 평가 완료 (신경망 지표 `--skip-neural`로 생략).**"
            if skip_neural
            else "**로컬 자동 지표(규칙·어휘·SBERT·BERTScore·NLI) 평가 완료. LLM Judge 포함 여부는 해당 절을 참고한다.**"
        ),
        "",
        f"`{split_name}` 분할의 매칭 문장 **{len(matched_indices)}/{len(target_indices)}건**에 대해 "
        f"OpenRouter/OpenAI 단일 호출(`C0`: `{model_id}`)과 기존 파인튜닝(`Gemma 3-1B`), 합성 참조문(`ChatGPT 1회 생성`)을 "
        "동일 행(paired) 기준으로 비교한 보고서다.",
        f"평가일: {datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')} (KST).",
        "",
        "## Setup & Coverage",
        "",
        f"- **모델 / 프로바이더**: `{model_id}` (`{provider}`, `temperature={temp}`)",
        f"- **프롬프트**: `news-simplify-c0` (`{prompt_ver}`, sha256 `{prompt_sha[:12]}`)",
        f"- **평가 분할**: `{split_payload.get('version', 'custom')}` / `{split_name}` "
        f"(요청 {len(target_indices)}건 중 유효 출력 **{len(matched_indices)}건** 매칭, 누락/실패 {len(missing_indices)}건)",
        f"- **오류 로그(`errors.jsonl`) 건수**: {len(err_records)}건",
    ]
    if missing_indices:
        lines.append(f"- **미매칭 인덱스**: `{missing_indices}`")
    lines += [
        "",
        "```bash",
        command,
        "```",
        "",
        "## Summary (Paired Differences, 95% Bootstrap CI)",
        "",
        f"- **원문 숫자 모두 지킴**: C0 {t.cell('c0', 'num_all_kept', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'num_all_kept', ci=False, **report._pct())} "
        f"(차이 {diff_str('num_all_kept', 'c0', 'model', **pct_diff_kw)}) / 합성 참조문 대비 {diff_str('num_all_kept', 'c0', 'reference', **pct_diff_kw)}",
        f"- **원문 고유명사 재현율**: C0 {t.cell('c0', 'name_recall', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'name_recall', ci=False, **report._pct())} "
        f"(차이 {diff_str('name_recall', 'c0', 'model', **pct_diff_kw)})",
        f"- **드문 명사 비율**: C0 {t.cell('c0', 'rare_noun_ratio', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'rare_noun_ratio', ci=False, **report._pct())} "
        f"(차이 {diff_str('rare_noun_ratio', 'c0', 'model', **pct_diff_kw)})",
        f"- **끝맺지 않은 출력**: C0 {t.cell('c0', 'unfinished', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'unfinished', ci=False, **report._pct())} "
        f"(차이 {diff_str('unfinished', 'c0', 'model', **pct_diff_kw)})",
    ]
    if not skip_neural:
        lines += [
            f"- **NLI 모든 문장 뒷받침**: C0 {t.cell('c0', 'nli_supported', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'nli_supported', ci=False, **report._pct())} "
            f"(차이 {diff_str('nli_supported', 'c0', 'model', **pct_diff_kw)})",
            f"- **NLI 모순 문장 포함**: C0 {t.cell('c0', 'nli_contradiction', ci=False, **report._pct())} vs Gemma 3-1B {t.cell('model', 'nli_contradiction', ci=False, **report._pct())} "
            f"(차이 {diff_str('nli_contradiction', 'c0', 'model', **pct_diff_kw)})",
        ]
    lines += [
        f"- **한국어 형태소 SARI (합성 참조문 기준, 보조)**: C0 {t.cell('c0', 'sari_morph', ci=False)} vs Gemma 3-1B {t.cell('model', 'sari_morph', ci=False)} "
        f"(차이 {diff_str('sari_morph', 'c0', 'model')}점)",
        "",
        "## Operational Efficiency (C0)",
        "",
        "| 지표 | 값 |",
        "|---|---:|",
        f"| 평가 완료 건수 | {op['n']} / {len(target_indices)} |",
        f"| 재시도 발생 건수 (attempts > 1) | {op['retry_count']} |",
        f"| `<think>` 블록 제거 비율 | {op['thinking_count']}건 ({op['thinking_rate'] * 100:.1f}%) |",
        f"| 지연시간 (mean / p50 / p95) | "
        + (
            f"{op['latency_mean']:.2f}s / {op['latency_p50']:.2f}s / {op['latency_p95']:.2f}s"
            if op["latency_mean"] is not None
            else "—"
        )
        + " |",
        f"| 평균 토큰 (prompt / completion / total) | "
        + (
            f"{op['prompt_tokens_mean']:.1f} / {op['completion_tokens_mean']:.1f} / {op['total_tokens_mean']:.1f}"
            if op["total_tokens_mean"] is not None
            else "—"
        )
        + " |",
        f"| 총 토큰 (total) | {op['total_tokens_total']:,} |" if op["total_tokens_total"] is not None else "| 총 토큰 (total) | — |",
        f"| 보고된 API 비용 (USD) | \\${op['total_cost_usd']:.6f} |" if op["total_cost_usd"] is not None else "| 보고된 API 비용 (USD) | — |",
        "",
        "## Meaning Preservation (Non-reference)",
        "",
    ]

    if skip_neural:
        lines += ["`--skip-neural` 옵션으로 SBERT·BERTScore·KLUE NLI 계산을 생략했다.", ""]
    else:
        lines += [
            "참조문 없이 원문과 직접 비교한 결과다. 각 차이 열은 대응 행 bootstrap 95% 신뢰구간이다.",
            "",
            t.render(
                ["c0", "model", "reference", "copy"],
                [
                    ("SBERT cosine (원문)", "sbert_src", {"digits": 3}),
                    ("BERTScore P (출력 → 원문)", "bertscore_src_p", {"digits": 3}),
                    ("BERTScore R (원문 → 출력)", "bertscore_src_r", {"digits": 3}),
                    ("NLI: 모든 문장이 원문으로 뒷받침됨", "nli_supported", report._pct()),
                    ("NLI: 원문과 모순되는 문장 있음", "nli_contradiction", report._pct()),
                    ("NLI coverage (원문 문장이 남은 정도)", "nli_coverage", {"digits": 3}),
                ],
                diffs=[("c0", "model"), ("c0", "reference")],
            ),
            "",
        ]

    num_src_n = int(c0_per.num_src_count.gt(0).sum())
    name_src_n = int(c0_per.name_src_count.gt(0).sum())
    lines += [
        "## Numbers and Names",
        "",
        f"매칭 표본 중 숫자가 검출된 원문은 {num_src_n}개, 고유명사(NNP)가 검출된 원문은 {name_src_n}개다.",
        "",
        t.render(
            ["c0", "model", "reference"],
            [
                ("원문 숫자 재현율", "num_recall", report._pct()),
                ("원문 숫자를 모두 지킴", "num_all_kept", report._pct()),
                ("원문에 없는 숫자 등장", "num_novel", report._pct()),
                ("원문 고유명사 재현율", "name_recall", report._pct()),
            ],
            diffs=[("c0", "model"), ("c0", "reference")],
        ),
        "",
        "## Simplicity, Style and Output Defects",
        "",
        t.render(
            ["c0", "model", "reference", "copy"],
            [
                ("글자 수", "chars", {"digits": 1}),
                ("문장 수", "sentences", {}),
                ("문장당 어절 수", "eojeol_per_sent", {}),
                ("문장당 형태소 수", "morph_per_sent", {"digits": 1}),
                ("드문 명사 비율", "rare_noun_ratio", report._pct()),
                ("해요체 문장", "reg_haeyo", report._pct()),
                ("해라체(-다) 문장", "reg_haera", report._pct()),
                ("반말 문장", "reg_banmal", report._pct()),
                ("종결어미 없는 문장", "reg_none", report._pct()),
                ("원문과 완전히 같음", "identical_to_source", report._pct()),
                ("원문과 글자 유사도", "copy_similarity", {"digits": 3}),
                ("끝맺지 않은 출력", "unfinished", report._pct()),
                ("같은 어절 3-gram 3회 이상 반복", "repetition_loop", report._pct()),
                ("한자 글자 수", "hanja_chars", {"digits": 3}),
            ],
            diffs=[("c0", "model"), ("c0", "reference")],
        ),
        "",
        "## Reference-Based Metrics (Auxiliary)",
        "",
        "`simplified_human` 역시 과거 ChatGPT 1회 생성 참조문으로 오류가 포함돼 있으므로, 아래 수치는 참고용 보조 지표로만 해석한다.",
        "",
        t.render(
            ["c0", "model", "copy"],
            [
                ("SARI (형태소)", "sari_morph", {}),
                ("　SARI keep", "sari_morph_keep", {}),
                ("　SARI delete", "sari_morph_del", {}),
                ("　SARI add", "sari_morph_add", {}),
                ("SARI (어절, 기존 방식)", "sari_ws", {}),
                ("BLEU (형태소, 문장 평균)", "bleu_morph", {}),
                ("chrF++", "chrf_pp", {}),
                ("ROUGE-L (형태소)", "rouge_l_morph", {"digits": 3}),
                *(
                    [
                        ("BERTScore F1 (mBERT)", "bertscore_ref_f", {"digits": 3}),
                        ("SBERT cosine", "sbert_ref", {"digits": 3}),
                    ]
                    if not skip_neural
                    else []
                ),
            ],
            diffs=[("c0", "model")],
        ),
        "",
        f"Corpus BLEU(형태소): C0 {corpus.get('c0_bleu_morph_corpus', np.nan):.2f}, "
        f"Gemma 3-1B {corpus.get('model_bleu_morph_corpus', np.nan):.2f}, "
        f"Copy {corpus.get('copy_bleu_morph_corpus', np.nan):.2f}. "
        f"Corpus chrF++: C0 {corpus.get('c0_chrf_pp_corpus', np.nan):.2f}, "
        f"Gemma 3-1B {corpus.get('model_chrf_pp_corpus', np.nan):.2f}, "
        f"Copy {corpus.get('copy_chrf_pp_corpus', np.nan):.2f}.",
        "",
    ]

    bounds = split_payload.get("tertile_bounds") or {"short_max": 70, "medium_max": 118}
    short_max = int(bounds["short_max"])
    med_max = int(bounds["medium_max"])
    src_lens = pd.Series([len(str(raw_df.loc[i, "original"])) for i in matched_indices], index=np.arange(len(matched_indices)))
    buckets = {
        "short": src_lens[src_lens <= short_max].index,
        "medium": src_lens[(src_lens > short_max) & (src_lens <= med_max)].index,
        "long": src_lens[src_lens > med_max].index,
    }
    length_metrics = [
        ("숫자 모두 지킴", "num_all_kept", report._pct()),
        ("드문 명사 비율", "rare_noun_ratio", report._pct()),
        ("끝맺지 않은 출력", "unfinished", report._pct()),
        ("SARI (형태소)", "sari_morph", {}),
    ]
    if not skip_neural:
        length_metrics += [
            ("SBERT cosine (원문)", "sbert_src", {"digits": 3}),
            ("NLI 모든 문장 뒷받침", "nli_supported", report._pct()),
            ("NLI 모순 있음", "nli_contradiction", report._pct()),
            ("NLI coverage", "nli_coverage", {"digits": 3}),
        ]

    lines += [
        "## By Source Length",
        "",
        f"전체 데이터셋 원문 글자 수 3분위 기준(`short ≤ {short_max}`, `medium ≤ {med_max}`, `long > {med_max}`)으로 "
        f"`C0`({len(buckets['short'])}/{len(buckets['medium'])}/{len(buckets['long'])}건)와 `Gemma 3-1B`를 나란히 비교한다.",
        "",
        "| Metric | short (C0 / Gemma) | medium (C0 / Gemma) | long (C0 / Gemma) |",
        "|---|---|---|---|",
    ]
    for label, metric, kw in length_metrics:
        if metric not in c0_per.columns:
            continue
        cells = []
        for bname in ("short", "medium", "long"):
            bidx = buckets[bname]
            if len(bidx) == 0:
                cells.append("—")
                continue
            v_c0 = c0_per.loc[bidx, metric].to_numpy(dtype=float)
            v_m = model_per.loc[bidx, metric].to_numpy(dtype=float)
            s_c0 = report._fmt(report._boot(v_c0, np.random.default_rng(seed))[0], **kw)
            s_m = report._fmt(report._boot(v_m, np.random.default_rng(seed))[0], **kw)
            cells.append(f"{s_c0} / {s_m}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    if stress_indices:
        lines += [
            "## Stress-Test Diagnostic Comparison",
            "",
            "기존 `evaluation/REPORT.md`에서 확인된 오류·경계 사례 10건에 대한 정성·규칙 비교다.",
            "",
        ]
        for s_idx in stress_indices:
            if not 0 <= s_idx < len(raw_df):
                continue
            row = raw_df.loc[s_idx]
            orig = str(row["original"])
            gemma = str(row["simplified_by_model"])
            ref_txt = str(row["simplified_human"])
            c0_rec = preds.get(s_idx)
            c0_txt = str(c0_rec["simplified_c0"]) if c0_rec else "(미실행)"

            src_nums = korean.numbers(orig)
            g_missing = src_nums - korean.numbers(gemma)
            g_trunc = korean.looks_truncated(gemma)
            if c0_rec:
                c0_missing = src_nums - korean.numbers(c0_txt)
                c0_novel = korean.numbers(c0_txt) - src_nums
                c0_trunc = korean.looks_truncated(c0_txt)
                diag = (
                    f"규칙 진단 — 원문 숫자: `{_fmt_num_set(src_nums)}` | "
                    f"C0 누락: `{_fmt_num_set(c0_missing)}`, 새 숫자: `{_fmt_num_set(c0_novel)}`, 미완성: `{c0_trunc}` | "
                    f"Gemma 누락: `{_fmt_num_set(g_missing)}`, 미완성: `{g_trunc}`"
                )
            else:
                diag = f"규칙 진단 — 원문 숫자: `{_fmt_num_set(src_nums)}` | Gemma 누락: `{_fmt_num_set(g_missing)}`, 미완성: `{g_trunc}`"

            lines += [
                f"**Stress #{s_idx}**",
                "",
                f"- 원문: {report._clip(orig, 320)}",
                f"- Gemma 3-1B: {report._clip(gemma, 320)}",
                f"- 합성 참조문: {report._clip(ref_txt, 320)}",
                f"- C0: {report._clip(c0_txt, 320)}",
                f"- {diag}",
                "",
            ]

    lines += [
        "## C0 Error Candidates",
        "",
        "자동 지표로 찾은 C0 문제 후보 사례다.",
        "",
    ]
    picks: list[tuple[str, list[int]]] = []
    if not skip_neural and "nli_contra_max" in c0_per.columns:
        contra_idx = c0_per[c0_per.nli_contradiction == 1].nli_contra_max.sort_values(ascending=False).head(3).index.tolist()
        picks.append(("원문과 모순으로 판별된 후보 (NLI)", contra_idx))
    num_drop_idx = c0_per[c0_per.num_all_kept == 0].sort_values("num_recall").head(3).index.tolist()
    picks.append(("숫자 누락·변경", num_drop_idx))
    unfin_idx = c0_per[c0_per.unfinished == 1].head(2).index.tolist()
    picks.append(("끝맺지 않은 출력", unfin_idx))

    any_pick = False
    for title, local_idxs in picks:
        for li in local_idxs:
            any_pick = True
            orig_idx = matched_indices[int(li)]
            row = raw_df.loc[orig_idx]
            lines += [
                f"**{title}** · #{orig_idx}",
                "",
                f"- 원문: {report._clip(row['original'], 280)}",
                f"- C0: {report._clip(preds[orig_idx]['simplified_c0'], 280)}",
                f"- Gemma 3-1B: {report._clip(row['simplified_by_model'], 280)}",
                "",
            ]
    if not any_pick:
        lines += ["규칙·NLI 기준 주요 오류 후보가 검출되지 않았다.", ""]

    if judge_df is not None and not judge_df.empty and "judge_meaning" in judge_df.columns:
        jt = _CompareTable(
            report.summarize(
                judge_df.drop(
                    columns=[
                        c
                        for c in ("judge_reason", "judge_model", "judge_prompt", "judge_error")
                        if c in judge_df.columns
                    ]
                ),
                seed=seed,
            ),
            judge_df,
        )
        lines += [
            "## LLM Judge",
            "",
            f"`{judge_df.judge_model.iloc[0]}`가 원문 길이 구간별로 고른 {judge_df.idx.nunique()}개 문장을 블라인드 채점했다.",
            "",
            jt.render(
                ["c0", "model", "reference"],
                [
                    ("Meaning", "judge_meaning", {}),
                    ("Completeness", "judge_completeness", {}),
                    ("Simplicity", "judge_simplicity", {}),
                    ("Fluency", "judge_fluency", {}),
                    ("Hallucination", "judge_hallucination", report._pct()),
                    ("Number error", "judge_number_error", report._pct()),
                ],
                diffs=[("c0", "model"), ("c0", "reference")],
            ),
            "",
        ]
    else:
        lines += [
            "## LLM Judge",
            "",
            "이번 실행에는 외부 LLM 채점이 포함되지 않았다 (`--judge-n 0`). 필요 시 교차 패밀리 모델로 `--judge-n`을 지정해 실행한다.",
            "",
        ]

    auto = "\n".join(lines)
    manual = "## Interpretation\n\n(직접 작성)\n"
    if out_path.exists():
        prev = out_path.read_text(encoding="utf-8")
        if report.MANUAL_MARKER in prev:
            manual = prev.split(report.MANUAL_MARKER, 1)[1].lstrip("\n")
    return f"{auto}\n{report.MANUAL_MARKER}\n\n{manual}"


def run_comparison(
    *,
    pred_dir: Path,
    csv_path: Path = PROJECT / "result.csv",
    split_file: Path = PROJECT / "evaluation" / "splits" / "stratified_v1.json",
    split: str = "eval",
    stress_split: str = "stress",
    out_path: Path = PROJECT / "evaluation" / "C0_REPORT.md",
    work_dir: Path | None = None,
    threads: int = min(8, os.cpu_count() or 1),
    offline: bool = False,
    skip_neural: bool = False,
    judge_n: int = 0,
    judge_model: str | None = None,
    seed: int = 0,
    command: str | None = None,
) -> dict[str, Any]:
    """Compare C0 predictions against Gemma 3-1B and synthetic references on matched rows."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    import torch

    torch.set_num_threads(max(1, int(threads)))

    pred_dir = Path(pred_dir)
    csv_path = Path(csv_path)
    split_file = Path(split_file)
    out_path = Path(out_path)
    work = Path(work_dir) if work_dir is not None else pred_dir / "eval_cache"
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_METRICS_CACHE", str(work / "metric-cache"))

    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing_cols = [c for c in COLUMNS if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")
    raw_df = raw_df[list(COLUMNS)].fillna("").astype(str)

    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    splits_map = split_payload.get("splits", {})
    if split not in splits_map:
        raise KeyError(f"Split {split!r} not found in {split_file}")
    target_indices = [int(i) for i in splits_map[split]]
    stress_indices = [int(i) for i in splits_map.get(stress_split, [])]

    preds, manifest, err_records = _load_predictions(pred_dir)
    matched_indices = [i for i in target_indices if i in preds]
    missing_indices = [i for i in target_indices if i not in preds]
    n = len(matched_indices)
    if n < 3:
        raise SystemExit(
            f"Need at least 3 matched predictions in split {split!r}; found {n}."
        )
    if not 0 <= judge_n <= n:
        raise ValueError("--judge-n must be between 0 and matched row count")

    src = [raw_df.loc[i, "original"] for i in matched_indices]
    ref = [raw_df.loc[i, "simplified_human"] for i in matched_indices]
    gemma = [raw_df.loc[i, "simplified_by_model"] for i in matched_indices]
    c0_out = [str(preds[i]["simplified_c0"]) for i in matched_indices]
    systems = {"c0": c0_out, "model": gemma, "reference": ref, "copy": src}

    packages = {}
    for pkg in (
        "torch", "numpy", "pandas", "sacrebleu", "rouge", "textstat", "kiwipiepy",
        "wordfreq", "transformers", "sentence-transformers", "bert-score", "evaluate",
    ):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = "missing"

    input_blob = json.dumps(
        {
            "matched_indices": matched_indices,
            "src": src,
            "ref": ref,
            "gemma": gemma,
            "c0": c0_out,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    identity = {
        "input_sha256": hashlib.sha256(input_blob).hexdigest(),
        "rows": n,
        "seed": seed,
        "skip_neural": bool(skip_neural),
        "metric_code": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
            if p.name
            in ("__init__.py", "lexical.py", "rules.py", "sari.py", "korean.py", "neural.py", "compare.py")
        },
        "packages": packages,
    }
    manifest_cache = work / "manifest.json"
    if manifest_cache.exists():
        if json.loads(manifest_cache.read_text(encoding="utf-8"))["identity"] != identity:
            raise SystemExit(
                f"Cached inputs, metric code, or packages changed; choose a new --work-dir: {work}"
            )
    else:
        manifest_cache.write_text(
            json.dumps(
                {"identity": identity, "pred_dir": str(pred_dir.resolve()), "split": split},
                indent=2,
            ),
            encoding="utf-8",
        )

    def stage(name: str, fn):
        path = work / f"{name}.pkl"
        if path.exists():
            print(f"[compare:{name}] reuse {path}", flush=True)
            return pd.read_pickle(path)
        t0 = time.time()
        print(f"[compare:{name}] running…", flush=True)
        res = fn()
        tmp = path.with_suffix(".tmp")
        pd.to_pickle(res, tmp)
        tmp.replace(path)
        print(f"[compare:{name}] done in {time.time() - t0:.1f}s", flush=True)
        return res

    def run_lexical():
        frames = []
        for s in ("c0", "model", "copy"):
            f = _safe_lexical_per_sample(src, systems[s], ref)
            f.insert(0, "idx", np.arange(n))
            f.insert(1, "system", s)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def run_rules():
        frames = []
        for s in ("c0", "model", "reference", "copy"):
            f = rules.per_sample(src, systems[s])
            f.insert(0, "idx", np.arange(n))
            f.insert(1, "system", s)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def run_sbert():
        sims = neural.sbert_similarity(
            {
                "c0|src": (src, c0_out),
                "model|src": (src, gemma),
                "reference|src": (src, ref),
                "c0|ref": (ref, c0_out),
                "model|ref": (ref, gemma),
                "copy|ref": (ref, src),
            }
        )
        return _long(
            {
                "c0": {"sbert_src": sims["c0|src"], "sbert_ref": sims["c0|ref"]},
                "model": {"sbert_src": sims["model|src"], "sbert_ref": sims["model|ref"]},
                "reference": {"sbert_src": sims["reference|src"]},
                "copy": {"sbert_src": np.ones(n), "sbert_ref": sims["copy|ref"]},
            },
            n,
        )

    def run_bertscore():
        bs = neural.bertscore(
            {
                "c0|src": (c0_out, src),
                "model|src": (gemma, src),
                "reference|src": (ref, src),
                "c0|ref": (c0_out, ref),
                "model|ref": (gemma, ref),
                "copy|ref": (src, ref),
            }
        )
        to_src = lambda k: {
            "bertscore_src_p": bs[k]["p"],
            "bertscore_src_r": bs[k]["r"],
            "bertscore_src_f": bs[k]["f"],
        }
        return _long(
            {
                "c0": {**to_src("c0|src"), "bertscore_ref_f": bs["c0|ref"]["f"]},
                "model": {**to_src("model|src"), "bertscore_ref_f": bs["model|ref"]["f"]},
                "reference": to_src("reference|src"),
                "copy": {
                    "bertscore_src_p": np.ones(n),
                    "bertscore_src_r": np.ones(n),
                    "bertscore_src_f": np.ones(n),
                    "bertscore_ref_f": bs["copy|ref"]["f"],
                },
            },
            n,
        )

    def run_nli():
        return _long(
            neural.nli_faithfulness(
                {s: (src, systems[s]) for s in ("c0", "model", "reference", "copy")},
                checkpoint_dir=work / "nli",
            ),
            n,
        )

    def run_corpus():
        values: dict[str, float] = {}
        for s in ("c0", "model", "copy"):
            values.update(
                {f"{s}_{k}": v for k, v in lexical.corpus(src, systems[s], ref).items()}
            )
        return values

    per = pd.DataFrame(
        {"idx": np.repeat(np.arange(n), len(systems)), "system": list(systems) * n}
    )
    stages: list[tuple[str, Any]] = [("lexical", run_lexical), ("rules", run_rules)]
    if not skip_neural:
        stages += [("sbert", run_sbert), ("bertscore", run_bertscore), ("nli", run_nli)]
    for name, fn in stages:
        per = per.merge(stage(name, fn), on=["idx", "system"], how="left")

    corpus = stage("corpus", run_corpus)
    per.to_csv(work / "per_sample.csv", index=False)
    summary_df = report.summarize(per, seed=seed)
    summary_df.to_csv(work / "summary.csv", index=False)
    (work / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")

    judged = None
    if judge_n:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env", override=False)
        resolved_judge = judge_model or (
            os.getenv("JUDGE_MODEL")
            or ("openai/gpt-4o-mini" if os.getenv("OPENROUTER_API_KEY") else "gpt-4o-mini")
        )
        indices_local = judge_mod.stratified_sample(
            pd.Series([len(s) for s in src]), judge_n, seed
        )
        judged = judge_mod.judge(
            src,
            {"c0": c0_out, "model": gemma, "reference": ref},
            indices_local,
            resolved_judge,
            checkpoint=work / f"judge_{resolved_judge.replace('/', '_')}_{judge_n}.jsonl",
        )
        judged.to_csv(work / "judge.csv", index=False)

    if command is None:
        command = "python -m evaluation.compare --pred-dir " + shlex.quote(
            str(pred_dir.relative_to(PROJECT) if pred_dir.is_relative_to(PROJECT) else pred_dir)
        )
        command += f" --split {shlex.quote(split)}"
        if skip_neural:
            command += " --skip-neural"
        if offline:
            command += " --offline"

    text = _build_c0_report(
        matched_indices=matched_indices,
        target_indices=target_indices,
        missing_indices=missing_indices,
        raw_df=raw_df,
        preds=preds,
        manifest=manifest,
        err_records=err_records,
        per=per,
        corpus=corpus,
        judge_df=judged,
        split_payload=split_payload,
        split_name=split,
        stress_indices=stress_indices,
        out_path=out_path,
        command=command,
        skip_neural=skip_neural,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return {
        "matched": n,
        "requested": len(target_indices),
        "missing": len(missing_indices),
        "out": str(out_path),
        "work_dir": str(work),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT / "result.csv")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT / "evaluation" / "splits" / "stratified_v1.json",
    )
    parser.add_argument("--split", default="eval")
    parser.add_argument("--stress-split", default="stress")
    parser.add_argument("--out", type=Path, default=PROJECT / "evaluation" / "C0_REPORT.md")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--judge-n", type=int, default=0)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cmd = "python -m evaluation.compare " + " ".join(
        shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:])
    )
    run_comparison(
        pred_dir=args.pred_dir,
        csv_path=args.csv,
        split_file=args.split_file,
        split=args.split,
        stress_split=args.stress_split,
        out_path=args.out,
        work_dir=args.work_dir,
        threads=args.threads,
        offline=args.offline,
        skip_neural=args.skip_neural,
        judge_n=args.judge_n,
        judge_model=args.judge_model,
        seed=args.seed,
        command=cmd,
    )


if __name__ == "__main__":
    main()
