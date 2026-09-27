# Rating Recommender System v2

DiningCode 리뷰로 사용자가 아직 가지 않은 식당을 추천하는 2-stage 추천 시스템이다.
v1(리뷰 텍스트로 평점 예측, [legacy/](./legacy/))을 추천 순위 문제로 바꿨다.

```text
Supabase PostgreSQL ─▶ 고정 snapshot ─▶ 전역 날짜 cutoff 분할
   ─▶ Stage 1 후보 생성 (C0 인기 · C1 item-item · C2 지역 인기 → C3 quota RRF, Top-100)
   ─▶ Stage 2 LightGBM LambdaRank 재정렬 (Top-10)
   ─▶ artifacts/runs/<run_id>/report.md · MLflow · Streamlit
```

- 모델과 평가 방식: [BASELINE_MODEL.md](./BASELINE_MODEL.md)
- 전체 계획과 마일스톤: [RECOMMENDER_V2_PLAN.md](./RECOMMENDER_V2_PLAN.md)
- 기록 문서: [analysis/](./analysis/README.md) · 로컬 산출물: [artifacts/](./artifacts/README.md)

## 폴더 구조

```text
rating_recsys/
├── src/rating_recsys/
│   ├── ingestion/        CSV·크롤링 결과 → PostgreSQL 적재
│   ├── db/               연결, migration
│   ├── datasets/         DB 조회, 전역 날짜 cutoff 분할
│   ├── retrieval/        Stage 1 후보 생성 (C0~C3)
│   ├── ranking/          Stage 2 feature, LambdaRank
│   ├── evaluation/       지표, run 보고서
│   ├── experiments/      실험 설정, 파이프라인, CLI
│   └── observability/    MLflow 기록, Streamlit 앱
├── tests/
├── migrations/           SQL schema
├── queries/              검증·분석용 SQL
├── crawler/              DiningCode Playwright 크롤러
├── analysis/             데이터 점검 기록, 보관한 이전 평가 결과
├── artifacts/            snapshot, run 결과, MLflow DB (보고서만 git에 올라감)
└── legacy/               v1 평점 예측 프로젝트
```

## 설치

```bash
cd projects/rating_recsys
conda create --prefix ./.venv python=3.10 pip libgomp -y
conda activate ./.venv
pip install -e '.[experiment,dev]'
cp .env.example .env   # DATABASE_URL, USER_HASH_SALT 설정
```

## 실험 실행

```bash
# 현재 DB를 읽어 snapshot으로 고정한 뒤 실행
rating-recsys-experiment

# 이미 고정한 snapshot으로 재실행
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

한 번 실행하면 `artifacts/runs/<run_id>/`가 생긴다. 결과는 그 안의 `report.md` 하나로
본다. 전체 데이터(88,554 interactions) 기준 로컬 8 thread로 약 20분 걸린다.

진행 순서:

1. 모든 사용자에게 같은 두 날짜 T1, T2를 적용해 train / validation / test로 나눈다.
2. Validation window로 C3 quota를 고르고, LambdaRank 설정과 트리 수(early
   stopping)를 고른다.
3. 고른 설정으로 T2까지의 데이터로 다시 학습하고, test window를 한 번 평가한다.
4. `report.md`를 쓰고 MLflow에 지표를 기록한다.

### 조정할 수 있는 조건

기본값과 다르게 준 조건은 보고서 2절에 모두 표시된다. 전체 목록은
`rating-recsys-experiment --help`.

| 구분 | 플래그 | 기본값 | 의미 |
|---|---|---|---|
| 분할 | `--train-fraction`, `--validation-fraction` | 0.8, 0.1 | T1, T2를 정하는 interaction 날짜 분위수 |
| 정답 | `--relevance-high`, `--relevance-low` | 4.0, 3.0 | relevance 2 / 1이 되는 평점. low 미만은 정답이 아님 |
| 후보 | `--candidate-k` | 100 | Stage 1이 넘기는 후보 수 |
| 후보 | `--region-mode` | `with_region` | `without_region`이면 C2와 지역 feature 제거 |
| 후보 | `--rrf-constant` | 60 | RRF 점수 1/(c + rank)의 c |
| LTR | `--ranking-k` | 10 | 최종 추천 수, 평가 cutoff |
| 선택 | `--quota-grid` | 0,0.25,0.5,0.75,1 | Validation에서 고를 C3 quota 후보 |
| 선택 | `--num-leaves-grid`, `--min-child-samples-grid` | 15,31,63 / 10,100 | LambdaRank grid |
| 선택 | `--learning-rate`, `--max-estimators`, `--early-stopping-rounds` | 0.05, 1000, 50 | 트리 수는 early stopping으로 결정 |
| 기타 | `--seed`, `--bootstrap-samples`, `--n-jobs` | 42, 2000, 8 | |
| 출력 | `--label`, `--no-mlflow`, `--artifacts-dir` | | run 이름표, MLflow 기록 끄기, 출력 위치 |

예: 후보를 200개로 늘려 비교 → `rating-recsys-experiment --snapshot <file> --candidate-k 200 --label k200`

### 결과 보기

- 보고서: `artifacts/runs/<run_id>/report.md`
- Streamlit: `rating-recsys-dashboard --address 127.0.0.1` → 사이드바에서 run과
  phase 선택 → 보고서 · 지표 · 사용자별 추천 · 식당별 노출 탭
- MLflow: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000`
  → experiment `rating-recsys`. Run마다 후보·LTR 지표(@K는 step), validation
  quota·ranker grid가 child run으로 있고, 여러 run을 Compare로 비교할 수 있다.
  파일은 복사하지 않으며 `report` tag가 보고서 경로를 가리킨다.

Streamlit과 MLflow UI에는 인증이 없으므로 `127.0.0.1`에서만 연다.

## 데이터 적재

모델링 코드는 CSV를 읽지 않고 Supabase의 `recsys` schema만 읽는다. 원본 사용자명은
저장하지 않고 고정 salt의 SHA-256 pseudonym으로 바꾼다. 파일 hash와 review
content hash로 중복을 막으므로 같은 파일을 다시 적재해도 안전하다.

```bash
rating-recsys-ingest --dry-run          # DB 연결 없이 schema·필수값·날짜 검사
rating-recsys-migrate                   # schema 적용
rating-recsys-ingest --skip-migrations  # 기존 5개 CSV 적재
python -m rating_recsys.ingestion.import_crawler   # 전국 크롤링 결과 적재
rating-recsys-dataset                   # 읽기 전용: 데이터 규모와 분할 크기 출력
```

`.env`의 `DATABASE_URL`은 Supabase Dashboard의 `Connect`에서 복사한다(IPv4 전용
네트워크면 port 5432 Session pooler, `sslmode=require` 유지). `USER_HASH_SALT`는 한 번
정하면 바꾸지 않는다. 검증 SQL은 [queries/](./queries/)에 있다.

2026-09-27 DB 기준 규모 (전국 크롤링은 미완료 중간 스냅샷):

| 항목 | 값 |
|---|---:|
| 원본 리뷰 (기존 / 전국 수집) | 96,922 (23,207 / 73,715) |
| 최초 사용자·식당 interaction | 88,554 |
| 사용자 / 식당 | 14,008 / 4,587 |
| 분할 train / validation / test | 70,883 / 8,934 / 8,737 |
| 구간 경계 | ~ 2025-12-19 / ~ 2026-05-04 / 이후 |

출처별 변화는 [데이터 스냅샷 기록](./analysis/data/2026-09-27_data_snapshot.md),
적재 품질은 [크롤링 적재 기록](./analysis/data/2026-09-26_crawler_import.md)에 있다.

## 테스트

```bash
pytest
```

분할 경계, 누수(학습 정답 날짜 ≤ cutoff, test 구간을 바꿔도 선택과 모델이 동일),
결정성, 지표 수식, 보고서, CLI, MLflow 기록, Streamlit 화면을 검사한다.

## 이전 평가 방식

2026-09-27까지는 사용자별 leave-last-two-out으로 평가했다. 코드는 제거했고 결과와
재현 방법은 [analysis/archive/primary/](./analysis/archive/primary/README.md)에 있다.
