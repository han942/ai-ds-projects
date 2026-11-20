# AI Agent 실험 계획 및 진행 현황

- 기존 파인튜닝 재평가: [evaluation/REPORT.md](./evaluation/REPORT.md)
- 에이전트 구현 개요: [README_AGENT.md](./README_AGENT.md)
- 고정 분할 파일: [evaluation/splits/stratified_v1.json](./evaluation/splits/stratified_v1.json)

## 1. 연구 목적 및 단계 전략

본 실험은 **동일한 한국어 뉴스 쉬운 글 변환 태스크**에서 추론 패러다임별 성능·사실 보존·운영 비용을 정량 비교한다.

- **핵심 질문 (1단계, $H_0$ 검증)**: 최신 LLM API 1회 호출(`C0: Single-shot LLM`)만으로 기존 파인튜닝(`B0: Gemma 3-1B QLoRA`)의 약점(숫자 누락·문장 미완성·어려운 용어 잔존)이 이미 해소되는가, 아니면 자기검수 루프(`C2`)·결정론적 검증 도구(`C3`)·외부 용어 사전 RAG(`C4`)가 실질적으로 필요한가?
- **현재 구현 및 실행 범위**: **OpenRouter `:free` LLM 기반 `C0` 기준선**을 먼저 구축하고, 동일 표본 위에서 `B0(Gemma 3-1B)` 및 `B1(ChatGPT 합성 참조문)`과 1:1 비교한다. (`C2`/`C3` 및 Blind LLM Judge 확장 여부는 `C0` 실측 결과를 보고 결정한다.)

| ID | 시스템 | 구성 | 비교 역할 |
|---|---|---|---|
| **B0** | `Gemma 3-1B QLoRA` | `result.csv` 저장 출력 | 기존 소형 모델 파인튜닝 기준선 |
| **B1** | `Synthetic Reference` | `result.csv` 내 ChatGPT 생성 참조문 | 참고선 (오류 포함, 정답 아님) |
| **C0** | **OpenRouter Single-shot LLM** | `news-simplify-c0` 단일 프롬프트 1회 호출 | **1단계 목표**: 에이전트/RAG 없이 API 단독 기준선 |

## 2. 데이터 분할 및 누수 방지

[result.csv](./result.csv) 2,340행(고유 원문 2,336건, 중복 원문 4쌍)에서 중복 원문을 그룹화한 뒤 **원문 길이 3분위(`short ≤ 70`, `medium ≤ 118`, `long > 118`) × 숫자 유무**의 6개 층으로 비례 층화 추출한다.

| 분할 | 규모 | 용도 | 비고 |
|---|---:|---|---|
| `eval` | 150 | **Held-out 본 비교 평가셋** (`seed=0`) | `dev`와 원문 중복 0건 |
| `dev` | 100 | 후속 프롬프트·검증 규칙 튜닝용 (`seed=42`) | `eval`·`stress`와 완전 분리 |
| `stress` | 10 | 정성·회귀 진단 셋 (`#0, #1, #5, #24, #25, #65, #100, #167, #1285, #2008`) | `evaluation/REPORT.md` 오류 사례 |

## 3. 평가 체계 및 후속 단계 전환 기준

### 평가 지표
- **사실 보존 (주 가드레일, 비참조)**: `num_all_kept`, `num_recall`, `num_novel`, `name_recall`, `klue-roberta-base-nli`(`nli_supported`, `nli_contradiction`, `nli_coverage`), `KR-SBERT`, `mBERTScore`
- **단순화·문체·결함 (비참조)**: `rare_noun_ratio`, `eojeol_per_sent`, `morph_per_sent`, `reg_haeyo`, `copy_similarity`, `unfinished`, `repetition_loop`, `hanja_chars`
- **참조문 기반 (보조)**: 한국어 형태소 `SARI`, `ROUGE-L`, `chrF++` (합성 참조문 오류로 주 판정 기준에서 제외)
- **운영 지표**: 호출 지연시간(`mean`/`p50`/`p95`), 입출력 토큰, 추정 비용, `<think>` 제거 비율, 실패/재시도 건수
- **통계 비교**: 모든 핵심 차이(`C0 − Gemma 3-1B`, `C0 − Reference`)는 대응 행(paired) bootstrap 95% 신뢰구간(1,000회)으로 판정

### C0 결과에 따른 후속 결정 기준
1. **규칙·NLI 기준 오류(`num_all_kept < 100%`, `unfinished > 0`, `nli_contradiction` 등)가 여전히 유의미하게 남는 경우**: `dev` 셋을 이용해 **Tool-Verified Agent(`C3`: 숫자·고유명사·드문명사·끝맺음 결정론적 검증 도구 + Critique-Refine)** 및 교차 패밀리 Blind LLM Judge로 확장한다.
2. **`C0`만으로 숫자·결함·NLI·단순화가 모두 안정적으로 포화하는 경우**: 무조건적 Agent 루프 대신 **다른 `:free` / 저비용 모델 간 비용–품질 비교** 또는 **긴 원문(`> 118자`)에 한정한 조건부 검수**로 범위를 좁힌다.

## 4. 실행 안내 (OpenRouter 기준)

`projects/NLP_Newspaper_CUAI` 디렉터리에서 실행한다.

### 4-1. 환경 변수 설정
`.env.example`을 복사해 `.env`를 만들고 `OPENROUTER_API_KEY`와 사용할 `:free` 모델 ID를 채운다.

```bash
cp .env.example .env
# .env 내 OPENROUTER_API_KEY=... 및 OPENROUTER_MODEL=... 설정
```

현재 사용 가능한 OpenRouter 무료 모델 목록은 아래 명령으로 조회할 수 있다.

```bash
.venv/bin/python -m agent.cli list-free-models --limit 30
```

### 4-2. Smoke Test (10건)
실제 API 한도·레이턴시·`<think>` 처리·체크포인트 재개가 정상 동작하는지 먼저 소규모로 확인한다.

```bash
.venv/bin/python -m agent.cli c0-run \
  --model "<YOUR_MODEL>:free" \
  --splits eval stress \
  --limit 10 \
  --temperature 0.0 \
  --min-interval 2.0 \
  --out-dir evaluation/runs/c0_smoke
```

### 4-3. Held-out 본 실행 (`eval` 150건 + `stress` 10건)
중단 시 동일 `--out-dir`로 재실행하면 `predictions.jsonl`에 저장된 성공 행을 건너뛰고 이어서 실행한다.

```bash
.venv/bin/python -m agent.cli c0-run \
  --model "<YOUR_MODEL>:free" \
  --splits eval stress \
  --temperature 0.0 \
  --min-interval 2.0 \
  --out-dir evaluation/runs/c0_main
```

### 4-4. 동일 표본 비교 평가 및 리포트 생성 (`evaluation/C0_REPORT.md`)
규칙·어휘·SBERT·BERTScore·KLUE NLI를 실행해 `C0` vs `Gemma 3-1B` vs `합성 참조문` 비교 표를 생성한다. (처음 실행 시 Hugging Face 평가 모델 캐시가 필요하므로 오프라인이 아닌 상태에서 1회 실행 후 `--offline` 사용 가능)

```bash
.venv/bin/python -m evaluation.compare \
  --pred-dir evaluation/runs/c0_main \
  --split eval \
  --out evaluation/C0_REPORT.md \
  --threads 8
```

빠르게 규칙·어휘 지표만 먼저 보려면 `--skip-neural`을 추가하고, 선택적으로 Blind LLM Judge까지 포함하려면 `--judge-n 50 --judge-model "<JUDGE_MODEL>"`을 지정한다.
