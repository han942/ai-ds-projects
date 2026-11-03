"""Aggregate per-sample scores into REPORT.md with bootstrap confidence intervals."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

MANUAL_MARKER = "<!-- manual: 이 줄 아래는 직접 작성한 해석이며, 파이프라인을 다시 실행해도 유지된다. -->"
SYSTEM_LABELS = {"model": "Model (Gemma 3-1B)", "reference": "Synthetic reference", "copy": "Copy source", "shuffled": "Shuffled model"}
N_BOOT = 1000


def _boot(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return v.mean(), *np.percentile(means, [2.5, 97.5])


def summarize(per: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metrics = [c for c in per.columns if c not in ("idx", "system") and pd.api.types.is_numeric_dtype(per[c])]
    for system, g in per.groupby("system"):
        for m in metrics:
            mean, lo, hi = _boot(g[m].to_numpy(dtype=float), rng)
            if not np.isnan(mean):
                rows.append({"system": system, "metric": m, "mean": mean, "lo": lo, "hi": hi, "n": int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def paired_diff(per: pd.DataFrame, metric: str, a: str, b: str, seed: int = 0) -> tuple[float, float, float]:
    if metric not in per.columns:
        return np.nan, np.nan, np.nan
    wide = per[per.system.isin([a, b])].pivot(index="idx", columns="system", values=metric).dropna()
    if wide.empty:
        return np.nan, np.nan, np.nan
    return _boot((wide[a] - wide[b]).to_numpy(dtype=float), np.random.default_rng(seed))


def _fmt(mean, lo=None, hi=None, scale=1.0, digits=2, pct=False, unit=""):
    if mean is None or np.isnan(mean):
        return "—"
    unit = "%" if pct else unit
    s = f"{mean * scale:.{digits}f}{unit}"
    if lo is not None and not np.isnan(lo):
        s += f" [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"
    return s


class _Table:
    def __init__(self, summary: pd.DataFrame, per: pd.DataFrame):
        self.s = summary.set_index(["system", "metric"])
        self.per = per

    def cell(self, system, metric, ci=True, **kw):
        if (system, metric) not in self.s.index:
            return "—"
        r = self.s.loc[(system, metric)]
        return _fmt(r["mean"], r["lo"] if ci else None, r["hi"] if ci else None, **kw)

    def mean(self, system, metric):
        return self.s.loc[(system, metric), "mean"] if (system, metric) in self.s.index else np.nan

    def render(self, systems, rows, diff=None):
        head = "| Metric | " + " | ".join(SYSTEM_LABELS[s] for s in systems)
        if diff:
            head += f" | {SYSTEM_LABELS[diff[0]]} − {SYSTEM_LABELS[diff[1]]}"
        lines = [head + " |", "|---" * (len(systems) + 1 + bool(diff)) + "|"]
        for label, metric, kw in rows:
            cells = [self.cell(s, metric, **kw) for s in systems]
            if diff:
                d = paired_diff(self.per, metric, *diff)
                kw2 = {k: v for k, v in kw.items() if k != "ci"}
                if kw2.get("pct"):
                    kw2.update(pct=False, unit=" pp")
                cells.append(_fmt(*d, **kw2) if not np.isnan(d[0]) else "—")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


def _pct(**kw):
    return {"scale": 100, "digits": 1, "pct": True, **kw}


def _clip(text: str, n: int = 220) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_report(data: pd.DataFrame, per: pd.DataFrame, corpus: dict, judge: pd.DataFrame | None,
                 out_path: Path, command: str, seed: int = 0, token_limits: dict | None = None,
                 pending_stages: tuple[str, ...] = ()) -> str:
    summary = summarize(per)
    t = _Table(summary, per)
    model = per[per.system == "model"].set_index("idx")
    n = len(data)

    fk = per[per.system == "model"][["fkgl_textstat", "textstat_words_per_sent"]].dropna()
    slope, intercept = np.polyfit(fk.textstat_words_per_sent, fk.fkgl_textstat, 1)
    fk_corr = np.corrcoef(fk.textstat_words_per_sent, fk.fkgl_textstat)[0, 1]

    sari_gap = t.mean("model", "sari_ws") - t.mean("shuffled", "sari_ws")
    lines = [
        "# News Simplification Re-evaluation",
        "",
        ("**진행 중: " + ", ".join(pending_stages) + " 계산이 남아 있다. 아래 — 표시는 아직 계산되지 않은 값이다.**" if pending_stages else "**로컬 자동 지표 평가 완료. 외부 LLM 채점 포함 여부는 마지막 LLM Judge 절을 참고한다.**"),
        "",
        f"`result.csv`의 test 문장 {n:,}개에 대해 Gemma 3-1B 파인튜닝 모델의 출력을 다시 평가한 보고서다. "
        "기존 지표를 재현하고, 한국어에 맞는 지표와 기준선(원문 복사, 출력 섞기)을 함께 계산했다. "
        "이 파일의 표와 수치는 평가 파이프라인이 자동으로 생성한다.",
        "추가 추론이나 재학습 없이 저장된 출력만 평가했으며, `agent/`의 LangGraph 에이전트 성능은 이 결과의 범위에 포함되지 않는다.",
        f"평가일: {datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')} (KST). 저장된 test 출력에 대한 평가이며 현재 시점의 뉴스나 새 모델 출력은 포함하지 않는다.",
        "",
        "## Summary",
        "",
        f"- 기존 방식의 SARI는 {t.mean('model', 'sari_ws'):.2f}이며, 원문을 그대로 복사해도 {t.mean('copy', 'sari_ws'):.2f}, "
        f"다른 문장의 출력과 무작위로 짝지어도 {t.mean('shuffled', 'sari_ws'):.2f}가 나온다. 무작위 대비 실제 차이는 {sari_gap:.1f}점이다.",
        f"- 영어용 가독성 지표 FKGL은 한국어 읽기 학년으로 해석할 수 없다. 본 데이터의 회귀식은 `FKGL ≈ {slope:.3f} × 문장당 어절 수 {intercept:+.2f}`"
        f"(상관 {fk_corr:.3f})로, 주로 문장 길이를 반영한다.",
        f"- NLI 자동 판별에서 모델 출력의 모든 문장이 원문으로 뒷받침된다고 분류된 비율은 {t.cell('model', 'nli_supported', ci=False, **_pct())}, "
        f"모순 문장이 있다고 분류된 비율은 {t.cell('model', 'nli_contradiction', ci=False, **_pct())}다 "
        f"(합성 참조문은 각각 {t.cell('reference', 'nli_supported', ci=False, **_pct())}, {t.cell('reference', 'nli_contradiction', ci=False, **_pct())}). 실제 오류율을 측정한 수치는 아니다.",
        f"- 숫자가 있는 문장 {int(model.num_src_count.gt(0).sum()):,}개 중 원문의 숫자를 모두 지킨 비율은 "
        f"{t.cell('model', 'num_all_kept', ci=False, **_pct())}다(합성 참조문 {t.cell('reference', 'num_all_kept', ci=False, **_pct())}). 숫자 값의 집합을 비교한 규칙 지표다.",
        f"- 해요체로 끝나는 문장 비율은 원문 {t.cell('copy', 'reg_haeyo', ci=False, **_pct())}, 모델 {t.cell('model', 'reg_haeyo', ci=False, **_pct())}, "
        f"합성 참조문 {t.cell('reference', 'reg_haeyo', ci=False, **_pct())}다. 드문 명사 비율은 원문 {t.cell('copy', 'rare_noun_ratio', ci=False, **_pct())}, "
        f"모델 {t.cell('model', 'rare_noun_ratio', ci=False, **_pct())}, 합성 참조문 {t.cell('reference', 'rare_noun_ratio', ci=False, **_pct())}다.",
        "",
        "## How to Reproduce",
        "",
        "`projects/NLP_Newspaper_CUAI`에서 실행한다. CPU를 사용하며 실행 시간은 장비와 문장 길이에 따라 달라진다. 중간 결과와 입력·패키지 정보는 `--work-dir`에 보존된다.",
        "",
        "```bash",
        "python3.12 -m venv .venv-eval",
        "source .venv-eval/bin/activate",
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "pip install -r evaluation/requirements-eval.txt",
        command,
        "```",
        "",
        "LLM judge는 외부 API를 호출하므로 `--judge-n`을 줄 때만 실행한다. `OPENROUTER_API_KEY` 또는 `OPENAI_API_KEY`가 필요하다.",
        "",
        "## Setup Notes",
        "",
        "실험 설정은 `Final_NLP_Newspaper.ipynb`, 참조문 생성 방식은 프로젝트 README의 설명을 기준으로 정리했다.",
        "기본 입력 자료: [원문·참조문·모델 출력 CSV](../result.csv), [원래 실험 노트북](../Final_NLP_Newspaper.ipynb), [프로젝트 설명](../README_KOR.md). 실행·중간 결과 파일은 [재현 안내](README.md)를 참고한다.",
        "",
        "- `simplified_human`이라는 열 이름과 달리, README는 ChatGPT API로 생성한 합성 병렬 데이터라고 설명한다. 개별 행의 생성 로그나 사람 검수 여부는 확인할 수 없어 합성 참조문으로 표기한다.",
        "- 학습 프롬프트(\"초등학생이 이해하기 쉽게 간단하게 바꿔줘\")와 추론 프롬프트(\"청소년이 이해할 수 있도록 변환…중요한 정보는 유지\")가 다르다.",
        "- 출력은 `temperature=0.6` 샘플링으로 한 번 생성했고, `max_new_tokens=250`으로 길이를 제한했다.",
        "- 학습·검증 데이터는 저장소에 없어서 test와의 중복 여부는 확인할 수 없다.",
        "- 동일한 프롬프트로 생성한 미세조정 전 기본 모델의 출력이 없어, 미세조정 자체의 개선 효과는 분리해 평가하지 못한다.",
        f"- 평가 단위는 CSV의 한 행이다. 원문이 중복된 추가 행은 {int(data.original.duplicated().sum())}개이며, 원래 평가와 비교하기 위해 제거하지 않았다. bootstrap은 행 단위 1,000회 재표집의 95% 구간으로, 실제 독자 평가나 모델·프롬프트 변화의 불확실성을 포함하지 않는다.",
        "- 해요체·드문 명사·끝맺음·숫자 검사는 해석 보조용 휴리스틱이다. 실제 읽기 난이도나 사실 오류를 확정하지 않는다.",
        "",
        "## Original Metrics Check",
        "",
        "README에 보고된 수치를 노트북과 같은 방식으로 다시 계산했다.",
        "",
        "| Metric | Reported | Reproduced | Note |",
        "|---|---:|---:|---|",
        f"| ROUGE-1 / 2 / L (F1) | 0.363 / 0.162 / 0.355 | {corpus['rouge1_ws']:.3f} / {corpus['rouge2_ws']:.3f} / {corpus['rougeL_ws']:.3f} | 합성 참조문 기준, 어절 단위 |",
        f"| BLEU-1 | 0.338 | {corpus['bleu1_vs_source']:.3f} | 노트북은 참조 문장이 아니라 원문과 비교했다. 참조 문장 기준은 {corpus['bleu1_vs_reference']:.3f} |",
        f"| SARI | 57.02 | {corpus['sari_ws']:.2f} | 원문 복사 {t.mean('copy', 'sari_ws'):.2f}, 출력 섞기 {t.mean('shuffled', 'sari_ws'):.2f} |",
        f"| FKGL (model / reference) | 1.28 / 1.34 | {t.mean('model', 'fkgl_textstat'):.2f} / {t.mean('reference', 'fkgl_textstat'):.2f} | 본 데이터의 회귀식은 `{slope:.3f} × 문장당 어절 수 {intercept:+.2f}` |",
        "",
        "README의 \"SARI 40 이상이면 우수\"라는 기준은 원문 복사와 무작위 출력도 넘거나 비슷한 값이라 성능 근거가 되지 않는다. "
        "또한 `agent/evaluation.py`가 쓰는 `rouge_score`의 기본 토크나이저는 한글을 제거하므로 한국어 문장의 ROUGE-L 평가에 부적합하다([Google Research 토크나이저 구현](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py)). 이 보고서는 해당 토크나이저를 사용하지 않는다.",
        "",
        "## Reference-Based Metrics",
        "",
        "합성 참조문과 얼마나 비슷한지 본다. 참조문이 하나뿐이므로 높은 점수만으로 정확성이나 대상 독자의 이해도를 보장하지 않는다. "
        "괄호는 행 단위 bootstrap 95% 구간이다. `morph`는 Kiwi 형태소 단위다. SARI는 추가·삭제·유지 연산을 비교하는 지표이며([Xu et al., 2016](https://aclanthology.org/Q16-1029/)), 이 실행은 원 노트북의 Hugging Face 구현과 같은 0/0=1 규칙과 13a 정규화를 사용한다.",
        "",
        "![SARI 기준선 비교: 어절 및 형태소 점수와 95% bootstrap 구간](figures/sari_baselines.png)",
        "",
        t.render(["model", "copy", "shuffled"], [
            ("SARI (어절, 기존 방식)", "sari_ws", {}),
            ("SARI (형태소)", "sari_morph", {}),
            ("　SARI keep", "sari_morph_keep", {}),
            ("　SARI delete", "sari_morph_del", {}),
            ("　SARI add", "sari_morph_add", {}),
            ("BLEU (형태소, 문장 평균)", "bleu_morph", {}),
            ("chrF++", "chrf_pp", {}),
            ("ROUGE-L (형태소)", "rouge_l_morph", {"digits": 3}),
            ("BERTScore F1 (mBERT)", "bertscore_ref_f", {"digits": 3}),
            ("SBERT cosine", "sbert_ref", {"digits": 3}),
        ]),
        "",
        f"형태소 SARI의 모델−원문 복사 차이는 {_fmt(*paired_diff(per, 'sari_morph', 'model', 'copy'))}점이다(대응 행 bootstrap 95% 구간). 무작위 기준선은 seed {seed}의 한 번의 출력 섞기이며, 구간은 섞기 자체의 변동성을 포함하지 않는다.",
        "",
        f"Corpus BLEU(형태소)는 model {corpus.get('model_bleu_morph_corpus', np.nan):.2f}, copy {corpus.get('copy_bleu_morph_corpus', np.nan):.2f}, "
        f"shuffled {corpus.get('shuffled_bleu_morph_corpus', np.nan):.2f}다. Corpus chrF++는 model {corpus.get('model_chrf_pp_corpus', np.nan):.2f}, "
        f"copy {corpus.get('copy_chrf_pp_corpus', np.nan):.2f}, shuffled {corpus.get('shuffled_chrf_pp_corpus', np.nan):.2f}다.",
        "",
        "## Meaning Preservation",
        "",
        "참조 문장 없이 원문과 직접 비교한다. SBERT는 한국어 STS로 학습한 문장 유사도 모델(`snunlp/KR-SBERT-V40K-klueNLI-augSTS`)이다. "
        "NLI는 KLUE-NLI로 학습한 `Huffon/klue-roberta-base-nli`로, 출력의 각 문장이 원문에서 추론되는지(support)와 원문의 각 문장이 출력에 남아 있는지(coverage)를 본다.",
        "NLI coverage는 각 원문 문장을 전체 출력에서 추론할 수 있다는 entailment 확률을 행별로 평균한 보조 점수다. 실제 보존된 사실의 비율이나 사람이 확인한 재현율을 뜻하지 않는다.",
        "SBERT는 배포된 모델의 기본 설정인 최대 128토큰을 사용하며, BERTScore는 mBERT의 9번째 층과 최대 512토큰을 사용한다. BERTScore는 IDF 가중치와 기준선 재조정을 사용하지 않았다.",
        (f"SBERT의 128토큰 한도를 넘는 원문은 {token_limits['counts_exceeding_sbert_limit']['original']['n']}건, "
         f"모델 출력은 {token_limits['counts_exceeding_sbert_limit']['simplified_by_model']['n']}건, "
         f"합성 참조문은 {token_limits['counts_exceeding_sbert_limit']['simplified_human']['n']}건이다. 이 문장들의 점수는 뒷부분을 모두 반영하지 못한다."
         if token_limits else "SBERT의 입력 한도를 넘는 문장의 의미 유사도는 뒷부분을 모두 반영하지 못할 수 있다."),
        "",
        t.render(["model", "reference", "copy", "shuffled"], [
            ("SBERT cosine (원문)", "sbert_src", {"digits": 3}),
            ("BERTScore P (출력 → 원문)", "bertscore_src_p", {"digits": 3}),
            ("BERTScore R (원문 → 출력)", "bertscore_src_r", {"digits": 3}),
            ("NLI: 모든 문장이 원문으로 뒷받침됨", "nli_supported", _pct()),
            ("NLI: 원문과 모순되는 문장 있음", "nli_contradiction", _pct()),
            ("NLI coverage (원문 문장이 남은 정도)", "nli_coverage", {"digits": 3}),
        ], diff=("model", "reference")),
        "",
        f"NLI는 모델의 분류 결과이며 사실 검증의 정답이 아니다. 원문 복사 기준선도 {t.cell('copy', 'nli_contradiction', ci=False, **_pct())}에서 모순으로 분류된다. 임계값은 entailment/contradiction 확률 0.5이며 별도 한국어 쉬운 글 검증셋으로 보정하지 않았다. 이 체크포인트는 위치 임베딩 512개와 패딩 위치 오프셋 때문에 실제 입력 한도가 510토큰이며, 긴 입력은 이 한도에서 잘린다. 숫자는 아래에서 규칙으로 따로 확인한다.",
        (f"실제 한도를 넘는 NLI 입력은 모델 근거 확인 {token_limits['nli_pair_limits']['model_support']['over_model_limit']}쌍, "
         f"참조문 근거 확인 {token_limits['nli_pair_limits']['reference_support']['over_model_limit']}쌍, "
         f"원문 복사 근거 확인·정보 보존 확인 각각 {token_limits['nli_pair_limits']['copy_support']['over_model_limit']}쌍이다. 모델·참조문의 정보 보존 확인에서는 한도를 넘는 입력이 없었다."
         if token_limits and "over_model_limit" in token_limits.get("nli_pair_limits", {}).get("model_support", {}) else ""),
        "",
        "## Numbers and Names",
        "",
        "원문의 숫자를 \"100만원 = 100만 원\", \"2000억 = 2천억\", \"세 곳 = 3곳\"처럼 값으로 바꿔 비교했다. 고유명사는 Kiwi의 NNP 태그로 뽑았다.",
        "숫자는 중복 횟수·단위·주체를 구분하지 않는 값의 집합이므로 숫자를 엉뚱한 대상에 붙인 오류는 놓칠 수 있다. 복합 한글 수사나 범위 표현도 완전히 처리하지 않으며, 법 조항의 '제6조'에서 '조'를 큰 수 단위로 해석하는 등 문맥을 혼동할 수 있다. 재현율과 '모두 지킴'의 분모는 원문에 검출된 숫자가 있는 행이며, 새 숫자 등장 비율의 분모는 전체 행이다. 고유명사 재현율은 검출된 고유명사가 있는 행을 사용한다.",
        "",
        t.render(["model", "reference"], [
            ("원문 숫자 재현율", "num_recall", _pct()),
            ("원문 숫자를 모두 지킴", "num_all_kept", _pct()),
            ("원문에 없는 숫자 등장", "num_novel", _pct()),
            ("원문 고유명사 재현율", "name_recall", _pct()),
        ], diff=("model", "reference")),
        "",
        f"숫자가 있는 원문은 {int(model.num_src_count.gt(0).sum()):,}개, 고유명사가 있는 원문은 {int(model.name_src_count.gt(0).sum()):,}개다.",
        "",
        "## Simplicity and Style",
        "",
        f"FKGL 대신 한국어에서 해석할 수 있는 지표를 썼다. 드문 명사는 `wordfreq` 한국어 빈도 목록에 없거나 Zipf 3.0 미만인 일반명사(NNG)다. "
        "말투는 문장의 종결어미로 나눴다.",
        "",
        t.render(["model", "reference", "copy"], [
            ("글자 수", "chars", {"digits": 1}),
            ("문장 수", "sentences", {}),
            ("문장당 어절 수", "eojeol_per_sent", {}),
            ("문장당 형태소 수", "morph_per_sent", {"digits": 1}),
            ("드문 명사 비율", "rare_noun_ratio", _pct()),
            ("해요체 문장", "reg_haeyo", _pct()),
            ("해라체(-다) 문장", "reg_haera", _pct()),
            ("반말 문장", "reg_banmal", _pct()),
            ("종결어미 없는 문장", "reg_none", _pct()),
        ], diff=("model", "reference")),
        "",
        "Copy source 열은 원문 자체의 값이다. 길이·말투 비율은 각 행의 값을 평균한 수치이며, 전체 문장을 합친 가중 비율과는 다르다. 친근한 말투로의 변경만으로 이해도가 높아졌다고 볼 수 없다. 비율의 차이 열은 퍼센트포인트(pp) 단위다.",
        "",
        "## Output Defects",
        "",
        t.render(["model", "reference"], [
            ("원문과 완전히 같음", "identical_to_source", _pct()),
            ("원문과 글자 유사도", "copy_similarity", {"digits": 3}),
            ("끝맺지 않은 출력", "unfinished", _pct()),
            ("같은 어절 3-gram 3회 이상 반복", "repetition_loop", _pct()),
            ("한자 글자 수", "hanja_chars", {"digits": 3}),
        ], diff=("model", "reference")),
        "",
    ]

    lengths = data.original.str.len()
    tertile = pd.qcut(lengths, 3, labels=["short", "medium", "long"])
    bounds = [int(lengths[tertile == b].max()) for b in ["short", "medium"]]
    keys = [("SARI (형태소)", "sari_morph", {}), ("SBERT cosine (원문)", "sbert_src", {"digits": 3}),
            ("NLI 모순 있음", "nli_contradiction", _pct()), ("숫자 모두 지킴", "num_all_kept", _pct()),
            ("NLI coverage", "nli_coverage", {"digits": 3}), ("끝맺지 않은 출력", "unfinished", _pct())]
    lines += ["## By Source Length", "",
              f"원문 글자 수로 세 구간을 나눴다(short ≤ {bounds[0]}, medium ≤ {bounds[1]}, long > {bounds[1]}). 모델 출력만 표시한다.", "",
              "| Metric | short | medium | long |", "|---|---|---|---|"]
    for label, metric, kw in keys:
        if metric not in model.columns:
            continue
        cells = []
        for b in ["short", "medium", "long"]:
            v = model.loc[tertile[tertile == b].index, metric].to_numpy(dtype=float)
            cells.append(_fmt(*_boot(v, np.random.default_rng(0)), **kw))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    if judge is not None and not judge.empty and "judge_meaning" in judge.columns:
        jt = _Table(summarize(judge.drop(columns=[c for c in ("judge_reason", "judge_model", "judge_prompt", "judge_error") if c in judge.columns])), judge)
        lines += ["## LLM Judge", "",
                  f"`{judge.judge_model.iloc[0]}`가 원문 길이 구간별로 고른 {judge.idx.nunique()}개 문장을 채점했다(프롬프트 `{judge.judge_prompt.iloc[0]}`). "
                  "출력이 모델과 GPT 참조 문장 중 무엇인지는 알려주지 않았다. 점수는 1~5점이다.", "",
                  jt.render(["model", "reference"], [
                      ("Meaning", "judge_meaning", {}), ("Completeness", "judge_completeness", {}),
                      ("Simplicity", "judge_simplicity", {}), ("Fluency", "judge_fluency", {}),
                      ("Hallucination", "judge_hallucination", _pct()), ("Number error", "judge_number_error", _pct()),
                  ], diff=("model", "reference")), ""]
        errors = int(judge.get("judge_error", pd.Series(dtype=object)).notna().sum())
        if errors:
            lines += [f"API 오류로 {errors}건은 채점하지 못했다.", ""]

    lines += ["## Examples", "", "자동 지표로 찾은 문제 후보 사례다. 원문, 모델 출력, 합성 참조문 순서다. 전체 오류율을 추정하기 위한 대표 표본은 아니다.", ""]
    picks = [
        ("원문과 모순으로 판별된 후보 (NLI)", model[model.nli_contradiction == 1].nli_contra_max.sort_values(ascending=False).head(3).index if "nli_contra_max" in model else []),
        ("숫자 누락·변경", model[model.num_all_kept == 0].sort_values("num_recall").head(3).index),
        ("끝맺지 않은 출력", model[model.unfinished == 1].head(2).index),
    ]
    for title, idxs in picks:
        for i in idxs:
            row = data.loc[i]
            lines += [f"**{title}** · #{i}", "",
                      f"- 원문: {_clip(row.original)}", f"- 모델: {_clip(row.simplified_by_model)}",
                      f"- 합성 참조문: {_clip(row.simplified_human)}", ""]

    if judge is None:
        lines += ["## LLM Judge", "", "이번 실행에는 외부 LLM 채점이 포함되지 않았다. 로컬 지표만으로 사람의 의미 보존·이해도 평가를 대신할 수 없다.", ""]

    auto = "\n".join(lines)
    manual = "## Interpretation\n\n(직접 작성)\n"
    if out_path.exists():
        prev = out_path.read_text(encoding="utf-8")
        if MANUAL_MARKER in prev:
            manual = prev.split(MANUAL_MARKER, 1)[1].lstrip("\n")
    return f"{auto}\n{MANUAL_MARKER}\n\n{manual}"
