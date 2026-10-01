# Rating Recommender System v2

DiningCode 리뷰로 사용자가 아직 가지 않은 식당을 추천하는 2-stage 추천 시스템이다.
v1(리뷰 텍스트로 평점 예측, [legacy/](./legacy/))을 추천 순위 문제로 바꿨다.

```text
Supabase PostgreSQL ─▶ 고정 snapshot ─▶ 전역 날짜 cutoff 분할
   ─▶ Stage 1 후보 생성 (C1 item-item + C4 LightGCN → C5 RRF, Top-100)
   ─▶ Stage 2 LightGBM LambdaRank 재정렬 (Top-10)
   ─▶ artifacts/runs/<run_id>/report.md · MLflow · Streamlit
```

Stage 1 기준선은 2026-09-30에 C3(인기·item-item·지역 quota 결합)에서 C5(item-item +
LightGCN)로 바꿨다. C0 인기, C2 지역 인기, C3는 결합하지 않고 비교용으로만 계산한다.

| 2026-09-30 기준선 (test 1,573명) | 값 |
|---|---:|
| Stage 1 Recall@100 | C5 20.16% (이전 C3 16.66%, +3.50%p [+2.30, +4.70]) |
| Top-10 NDCG@10 | R0(C5 순서) 0.0276 · R1(LambdaRank) 0.0276 · R1 − R0 −0.0000 [−0.0046, +0.0044] |

[보고서](./artifacts/runs/20260930T135424227862Z-e7896add/report.md). R1은 아직 C5 순서보다
나아지지 않는다(보고서 6절). 같은 날 앞선 run은 학습 정답을 후보 밖에 끼워 넣어 R1이
그 위치를 학습했기 때문에 대체했다.

2026-10-01부터 R1 점수가 같으면 C5 순위를 유지한다. 위 지표는 이 변경 전 실행 결과다.

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
│   ├── retrieval/        Stage 1 후보 생성 (C0~C5, LightGCN, DeepCoNN)
│   ├── ranking/          Stage 2 feature, LambdaRank
│   ├── evaluation/       지표, run·후보 비교 보고서
│   ├── experiments/      실험 설정, 파이프라인, 후보 모델 비교, CLI
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
pip install -e '.[deepconn]' --extra-index-url https://download.pytorch.org/whl/cpu  # DeepCoNN 실험만
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
본다. 전체 데이터(88,554 interactions) 기준 로컬 8 thread로 약 24분 걸린다(그중 학습
query용 LightGCN 40개 학습 약 4분).

진행 순서:

1. 모든 사용자에게 같은 두 날짜 T1, T2를 적용해 train / validation / test로 나눈다.
2. Window 시작 이전 interaction으로 LightGCN을 학습해 C5 후보를 만든다. Ranker 학습
   query에는 3개월마다 다시 학습한 LightGCN(그 구간 시작일 이전 interaction)을 쓴다.
   정답이 자기 C5 후보 안에 있는 학습 query만 ranker 학습에 쓴다.
3. Validation window로 LambdaRank 설정과 트리 수(early stopping)를 고른다.
4. 고른 설정으로 T2까지의 데이터로 다시 학습하고, test window를 한 번 평가한다.
5. `report.md`를 쓰고 MLflow에 지표를 기록한다.

### 조정할 수 있는 조건

기본값과 다르게 준 조건은 보고서 2절에 모두 표시된다. 전체 목록은
`rating-recsys-experiment --help`.

| 구분 | 플래그 | 기본값 | 의미 |
|---|---|---|---|
| 분할 | `--train-fraction`, `--validation-fraction` | 0.8, 0.1 | T1, T2를 정하는 interaction 날짜 분위수 |
| 정답 | `--relevance-high`, `--relevance-low` | 4.0, 3.0 | relevance 2 / 1이 되는 평점. low 미만은 정답이 아님 |
| 후보 | `--candidate-k` | 100 | Stage 1이 넘기는 후보 수 |
| 후보 | `--region-mode` | `with_region` | `without_region`이면 참고 C2와 지역 feature 제거 |
| 후보 | `--rrf-constant` | 60 | RRF 점수 1/(c + rank)의 c |
| 후보 | `--lightgcn-layers`, `--lightgcn-dimension`, `--lightgcn-epochs`, `--lightgcn-regularization` | 3, 64, 20, 1e-4 | C4 LightGCN 고정 설정 |
| 후보 | `--lightgcn-checkpoint-months` | 3 | 학습 query용 LightGCN 재학습 간격 (12의 약수) |
| 참고 | `--legacy-c3-quota` | 0.75 | 비교용 C3의 quota. Stage 1에는 영향 없음 |
| LTR | `--ranking-k` | 10 | 최종 추천 수, 평가 cutoff |
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
  ranker grid가 child run으로 있고, 여러 run을 Compare로 비교할 수 있다.
  파일은 복사하지 않으며 `report` tag가 보고서 경로를 가리킨다.

Streamlit과 MLflow UI에는 인증이 없으므로 `127.0.0.1`에서만 연다.

### 후보 모델 비교 실험

파이프라인과 같은 split·평가 query로 새 후보 소스를 C5와 비교한다(ranker는 재학습하지
않음). Validation window로 epoch별 학습 곡선을 그리고 grid와 epoch 수, 결합 방식(단독,
C1과 RRF, C1+C4와 RRF)을 고른 뒤, T2까지 다시 학습해 test를 한 번 평가한다. 결과는
`artifacts/comparisons/<model>/<run_id>/`의 `report.md`와 `learning_curve.png`. MLflow에는
기록하지 않는다.

| 실험 | 비교 대상 | 로컬 시간 | 결과 |
|---|---|---:|---|
| LightGCN | 당시 기준선 C3 | 약 13분 | [2026-09-28](./artifacts/comparisons/lightgcn/20260928T064358512632Z-e7896add/report.md): C3 16.66% → C1+LightGCN RRF 20.16% (Recall@100). 이 결과로 C5를 기준선으로 정함 |
| DeepCoNN (리뷰 텍스트 CNN) | 현재 기준선 C5 | 약 1시간 | [2026-09-30](./artifacts/comparisons/deepconn/20260930T111734850000Z-e7896add/report.md): 단독 8.69%, C1+LightGCN+DeepCoNN RRF 18.44%로 C5 20.16%보다 낮음 (Recall@100, −1.72%p [−2.65, −0.86]). C5 유지 |

```bash
rating-recsys-compare lightgcn \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --baseline-run artifacts/runs/<run_id>   # 선택: 같은 snapshot의 R1 Top-10과 비교

# 리뷰 본문은 snapshot 옆 <snapshot>.reviews.jsonl에서 읽는다. 없으면 DB에서 한 번 읽어 만든다.
rating-recsys-compare deepconn --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl

rating-recsys-compare <model> --help      # 모델별 grid 플래그
```

위 두 결과는 모델별 실험 파일로 실행했던 run이다(보고서 8절의 `python -m ...` 명령).
지금은 같은 절차가 `rating-recsys-compare` 하나로 합쳐졌다.

새 후보 모델 추가:

1. `retrieval/<model>.py`에 모델을 구현한다. `fit(..., callback=...)`이 epoch마다
   `callback(epoch_stats, model)`을 부르고(True를 돌려주면 중단), `history`와
   `recommend(user_ids, exclude, k)`가 있으면 된다.
2. `experiments/candidate_models.py`에 `CandidateModel` 하위 클래스 하나(grid 플래그,
   설정 타입, grid 축, 학습 호출, loss 이름)를 만들고 `CANDIDATE_MODELS`에 넣는다.

`config_type()`에서 모델의 frozen dataclass를 지연 import하면, 공통 코드가 설정
필드에서 `--field-name` 옵션을 자동으로 만든다. `epochs`와 `seed`는 공통 옵션
`--max-epochs`, `--seed`를 사용한다. Grid를 만들 필드만 아래처럼 선언한다.
기존 LightGCN·DeepCoNN 옵션과 기본 grid는 유지한다.

```python
grid_parameters = {
    "dimension": ("--dimension-grid", (32, 64)),
}
```

단일 옵션은 int·float·str·bool을 지원하며 bool은 `--normalize`/`--no-normalize`
형식이다. 서로 묶어서 비교할 설정(DeepCoNN의 objective·activation 등)은
`variant_fields`와 `variants()`로 정의한다. 복잡한 설정 타입은
`add_arguments()`와 `grid_from_args()`를 재정의할 수 있다. 값 검사는 모델 설정
dataclass가 담당한다. 별도 모델 CLI나 실행·보고서 파일은 추가하지 않는다.

실행 절차, CLI, 보고서, 학습 곡선, bootstrap 비교는 공통 코드(`experiments/comparison.py`,
`experiments/compare_cli.py`, `evaluation/comparison_report.py`)를 그대로 쓴다.

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

모델을 추가할 때 테스트 파일을 새로 만들 필요는 없다.

- `tests/support.py`의 `MODEL_CASES`에 빠르게 학습할 작은 설정 한 항목을 추가한다.
- `test_retrieval.py`의 공통 테스트가 등록된 모델마다 callback 중단, 이미 방문한
  식당 제외, 후보 개수·중복, 입력 순서를 바꿔도 같은 추천인지 검사한다.
- `test_comparison.py`는 등록된 모델에 대해 CLI·기본 grid, 학습 곡선,
  날짜 cutoff, 재학습·평가·보고서 생성을 함께 검사한다.
- 그래프 gradient, 리뷰 문서 구성 등 알고리즘 고유 검증은
  `test_retrieval.py`에 클래스로 추가한다. 리뷰 파일 검증은 `test_snapshot.py`에 둔다.

선택 의존성이 설치되지 않은 모델의 학습 테스트는 건너뛴다. 등록된 모델에 작은
테스트 설정이 누락되면 registry 검사가 실패한다.

## 이전 평가 방식

2026-09-27까지는 사용자별 leave-last-two-out으로 평가했다. 코드는 제거했고 결과와
재현 방법은 [analysis/archive/primary/](./analysis/archive/primary/README.md)에 있다.
