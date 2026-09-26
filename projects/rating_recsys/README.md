# Rating Recommender System v2

이 프로젝트는 기존의 리뷰 기반 평점 예측 실험을 넘어, 실제 데이터베이스를
사용하는 2-stage 식당 추천 시스템으로 확장한다.

- 전체 계획: [RECOMMENDER_V2_PLAN.md](./RECOMMENDER_V2_PLAN.md)
- Baseline 모델: [BASELINE_MODEL.md](./BASELINE_MODEL.md)
- 이전 버전: [legacy/v1_rating_prediction/](./legacy/v1_rating_prediction/)

현재 Supabase PostgreSQL ingestion, DB-backed modeling dataset과 실행 가능한
2-stage baseline 골격이 구현되어 있다. 동일 split에서 global
candidate component C0(popularity)~C3(RRF union)와 ranker R1(LambdaRank)을
비교하고 immutable snapshot, MLflow와 Streamlit artifact를
생성한다. 날짜는 초기 모델 feature로 사용하지 않고 chronological split과
leakage 방지에만 사용하며, time-aware recommendation은 후속 연구로 둔다.
기존의 노트북, 수집 데이터, 모델 checkpoint 및 예측 결과는 삭제하지 않고
legacy 폴더에 그대로 보존하였다.

## Two-stage baseline 실행

실험 dependency를 설치한 뒤 Supabase snapshot에서 C0 popularity, C1 item-item,
C2 region popularity, C3 quota RRF와 R1 LightGBM LambdaRank를 한 번에 실행한다.
C2를 포함한 C3에서 후보를 선택하고, R1이 그 후보를 재정렬한다.

```bash
pip install -e '.[experiment,dev]'
rating-recsys-experiment
```

실행 결과는 `artifacts/runs/<run_id>/`에 저장된다. MLflow에는 대용량 candidate와
ranking Parquet 전체를 복제하지 않고 핵심 artifact, dataset lineage, 단계별 metric,
추천 결과 table과 pipeline trace를 기록한다.

MLflow UI는 별도 터미널에서 실행한다.

```bash
mlflow ui \
  --backend-store-uri sqlite:///artifacts/mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

`rating-recsys-baseline` experiment에는 다음 구조가 생성된다.

- Parent run: validation/test 핵심 metric, dataset input, chart, table, trace
- `01 · C0 Popularity`: popularity candidate metric과 cutoff curve
- `02 · C1 Item-item CF`: item-item candidate metric과 cutoff curve
- `03 · C2 Region popularity`: 지역 candidate metric과 cutoff curve
- `04 · C3 RRF Candidate Union`: fused candidate metric과 cutoff curve
- `05 · R1 LambdaMART`: C3 후보의 final ranking metric
- Tables: 단계별 metric 및 test 사용자별 top-K 추천 결과
- Traces: split → candidate generation → LambdaMART → 평가 → artifact 기록

개별 추천의 정성 평가에는 다음 로컬 UI를 사용한다. 사용자별 화면에서는 과거
방문, held-out target, 실제 Top-K 추천, candidate 대비 최종 순위 이동과 source
score를 확인할 수 있다. 과거 방문과 target에는 해당 방문에서 작성한 리뷰를
함께 표시한다. 리뷰 본문은 별도 display-only artifact에 저장되며 모델 feature나
MLflow artifact에는 포함하지 않는다. 아이템별 화면에서는 특정 식당이 어떤 사용자에게 몇
순위로 추천됐는지, 실제 target과 일치했는지를 역조회할 수 있다. 기본 화면은
작은 Top-K artifact만 읽으며, 전체 candidate와 feature는 해당 query에서 상세
보기를 켰을 때만 불러온다. Metric은 순위 품질, coverage·discovery, 평가 모수·
latency로 구분하며 stage별 @K 값과 각 지표의 도움말을 제공한다.

```bash
rating-recsys-dashboard --address 127.0.0.1
```

자세한 architecture, layer output과 metric 정의는
[BASELINE_MODEL.md](./BASELINE_MODEL.md)를 참고한다.

지역 후보와 지역 affinity feature를 제거한 비교 실험은 같은 DB snapshot으로
기준선과 무지역 모델을 각각 학습한다. MLflow에 데이터를 올리지 않고 로컬 artifact에
결과를 저장한다.

```bash
python -m rating_recsys.experiments.compare_region
```

[2026-09-23 지역 제거 실험 결과](./analysis/region_ablation_2026-09-23.md)에서
동일 test query의 추천 품질과 지역 다양성 차이를 볼 수 있다.

LightGCN을 세 번째 후보 소스로 추가한 C0+C1+LightGCN RRF 실험은 분기별
과거 그래프로 모델을 학습하고 후보 Recall@20/50/100을 로컬에서 비교한다.

```bash
python -m rating_recsys.experiments.compare_lightgcn
```

[2026-09-23 LightGCN 후보 실험 결과](./analysis/lightgcn_candidate_2026-09-23.md)에
조건, 비교 결과, 재현 아티팩트를 기록했다.

## Supabase PostgreSQL ingestion

현재 5개 legacy CSV를 Supabase PostgreSQL의 private `recsys` schema로 적재하는
첫 번째 파이프라인이 구현되어 있다. 원본 사용자명은 저장하지 않고 고정된
salt를 사용한 SHA-256 pseudonym으로 변환한다. 파일 hash와 review content
hash를 사용하므로 같은 파일을 다시 실행해도 리뷰가 중복 삽입되지 않는다.

### 1. 환경 준비

```bash
cd projects/rating_recsys
conda create --prefix ./.venv python=3.10 pip libgomp -y
conda activate ./.venv
pip install -e '.[dev]'
cp .env.example .env
```

Supabase Dashboard의 `Connect`에서 연결 문자열을 복사하여 `.env`의
`DATABASE_URL`에 설정한다. Migration에는 direct connection이 가장 적합하지만
로컬 네트워크가 IPv4 only라면 port 5432의 Session pooler를 사용할 수 있다.
비밀번호의 예약 문자는 URL encoding하고 `sslmode=require`를 유지한다.

`USER_HASH_SALT`는 한 번 생성한 긴 random 문자열로 설정하고 이후 변경하지
않는다. Salt가 바뀌면 같은 source 사용자가 다른 사용자로 생성된다.

### 2. DB 연결 전 검증

```bash
rating-recsys-ingest --dry-run
```

Dry-run은 Supabase에 연결하거나 데이터를 쓰지 않고 5개 파일의 schema,
필수값, 날짜 parsing, 중복 review hash와 entity 수를 검사한다.

### 3. Migration과 적재

```bash
rating-recsys-migrate
rating-recsys-ingest --skip-migrations
```

또는 migration을 포함해 한 번에 실행한다.

```bash
rating-recsys-ingest
```

적재 결과는 파일별 전체·신규·중복·거부 행 수를 JSON으로 출력한다. 같은
명령을 다시 실행하면 이미 성공한 동일 file hash는 `skipped` 상태가 된다.

검증 query는 [queries/verify_ingestion.sql](./queries/verify_ingestion.sql)에
있다.

2026-09-26 전국 Playwright 크롤링의 현재 `.csv.partial` 내용은 별도 고정
스냅샷으로 적재했다. 기존 5개 파일의 `crawl_runs.source = 'diningcode'`와
새 수집분의 `source = 'diningcode_playwright_national'`로 리뷰를 구분한다.
원본 수집이 완료된 것으로 간주하지 않으며, 재수집 뒤 다시 실행하면 새 스냅샷
run으로 중복을 제외하고 추가할 수 있다.

```bash
python -m rating_recsys.ingestion.import_crawler
```

[적재 결과와 품질 확인](./analysis/crawler_import_2026-09-26.md),
[출처별 SQL](./queries/compare_crawl_sources.sql)을 참고한다. 기본 모델링
데이터셋 조회는 현재 두 출처를 함께 읽는다.

## DB-backed modeling dataset

모델링 코드는 legacy CSV를 읽지 않는다. 아래 명령은 Supabase의
`recsys.reviews`를 직접 읽고, 동일 사용자·식당의 최초 interaction만 남긴 뒤
두 평가 split을 생성하여 JSON audit을 출력한다. DB에는 쓰지 않는 read-only
명령이다.

```bash
rating-recsys-dataset
```

Primary benchmark는 고유 식당 3개 이상 사용자의 마지막 interaction을 test,
마지막에서 두 번째를 validation, 나머지를 train으로 배치한다. train 이력이
1개 이상인 사용자는 하나의 `seen user` 집단으로 평가하며 이력 1~2개와 3개
이상 구간은 진단 지표로만 분리한다.

Secondary benchmark는 전체 interaction의 날짜 분위수로 전역 cutoff를 만들고,
cutoff 이전 이력이 있는 `seen user`와 이력이 없는 `new user`를 별도로
집계한다.

기본 설정을 바꿔 audit할 수도 있다.

```bash
rating-recsys-dataset \
  --minimum-user-items 3 \
  --train-fraction 0.8 \
  --validation-fraction 0.1
```

현재 DB snapshot의 기본 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 최초 user-item interaction | 23,017 |
| Primary seen user | 2,396 |
| Primary train / validation / test | 12,504 / 2,396 / 2,396 |
| train history 1~2개 사용자 | 1,049 |
| train history 3개 이상 사용자 | 1,347 |

### 주요 파일

| 경로 | 역할 |
|---|---|
| `migrations/001_initial_ingestion.sql` | `crawl_runs`, `restaurants`, `app_users`, `reviews` schema |
| `src/rating_recsys/ingestion/transform.py` | CSV 정규화, 날짜 parsing, pseudonym 및 dedup hash |
| `src/rating_recsys/ingestion/loader.py` | PostgreSQL COPY와 set-based upsert |
| `src/rating_recsys/ingestion/cli.py` | dry-run 및 전체 ingestion command |
| `src/rating_recsys/datasets/repository.py` | Supabase에서 최초 user-item interaction 조회 |
| `src/rating_recsys/datasets/split.py` | seen-user 및 global temporal split 생성 |
| `src/rating_recsys/datasets/cli.py` | DB snapshot과 split audit command |
| `tests/test_transform.py` | 변환 규칙과 5개 CSV smoke test |
| `tests/test_dataset_split.py` | split, 중복 방지 및 seen/new cohort test |
