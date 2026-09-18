# Rating Recommender System v2 확장 계획

> 상태: 설계 단계  
> 방향: DB-backed data pipeline → Stage 1 candidate retrieval → Stage 2 learning-to-rank  
> 비용 원칙: 로컬·오픈소스 우선, 관리형 서비스와 유료 API는 기본 구성에서 제외

## 1. 프로젝트 전환 목표

v1은 다이닝코드 리뷰 텍스트와 평점 데이터를 이용하여 Matrix Factorization과
DeepCoNN 계열 모델을 분석한 프로젝트였다. v2는 평점 회귀 실험에서 벗어나,
실제 추천 요청을 처리할 수 있는 2-stage 추천 시스템을 구축한다.

핵심 목표는 다음과 같다.

1. CSV가 아닌 PostgreSQL을 모델링 데이터의 기준 저장소로 사용한다.
2. 데이터 수집부터 학습 데이터 생성까지 재실행 가능하고 중복에 안전한
   파이프라인을 만든다.
3. 시간 기준의 leakage-free 평가 프로토콜을 확립한다.
4. Stage 1 candidate retrieval과 Stage 2 learning-to-rank를 분리하여 각 단계의
   성능을 측정한다.
5. 리뷰·메뉴·지역 정보를 활용한 vector retrieval을 추가한다.
6. 충분한 데이터가 확보되면 semantic ID 기반 generative retrieval을 실험한다.

### 이번 버전의 비목표

- 처음부터 대규모 실시간 트래픽을 처리하는 분산 시스템 구축
- 관리형 Vector DB나 상용 Feature Store 도입
- HSTU·OneRec을 논문과 동일한 산업 규모로 재현
- Graph DB를 도입하는 것 자체를 프로젝트 목표로 삼는 것
- 별도의 학습형 pre-ranker 또는 re-ranker를 추가하여 3단계 이상으로 확장하는 것

## 2. 현재 데이터 진단

현재 보유 CSV를 프로파일링한 결과는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 원본 행 | 31,085 |
| 사용자 | 6,994 |
| 식당 | 약 748 |
| 정규화된 복합키 기준 중복 추정 | 7,870행 |
| 중복 제거 후 행 | 약 23,215 |
| 사용자 interaction 중앙값 | 2 |
| 중복 제거 후 interaction 1회 사용자 | 3,443 |
| interaction 5회 이상 사용자 | 1,357 |
| 평점 4점 이상 | 87.76% |
| 평점 2점 이하 | 2.35% |
| 연도가 없는 날짜 | 1,280건 |
| 상대 날짜 | 118건 |

중복 추정에는 `item_name`, `item_spec_area`, `user_name`, `user_query`, `date`를
공백 정규화한 복합키를 사용했다. 실제 중복 제거 규칙은 원본 행 표본을
검토한 뒤 확정한다.

### 우선 해결할 데이터 위험

#### 2.1 v1 평점 예측과 v2 추천 task의 분리

v1의 DeepCoNN은 작성된 리뷰 텍스트와 `taste`, `price`, `service` 평가로 해당
리뷰의 평점을 예측하는 post-interaction task다. 리뷰 기반 평점 예측이라는
기존 목적에서는 유효하며, v1의 분석 결과로 보존한다.

v2는 사용자의 과거 리뷰와 식당이 기존에 받은 리뷰를 이용하여 아직 평가하지
않은 식당의 순위를 계산하는 pre-interaction 추천 task다. 평가 대상 사용자와
식당 조합의 target 리뷰 및 그 리뷰에 딸린 세부 평가는 입력에서 제외하고
정답 label로만 사용한다. 기존 DeepCoNN도 이 조건에 맞게 user/item document를
다시 구성하여 Stage 1 baseline으로 비교할 수 있다.

#### 2.2 랜덤 분할

사용자별 랜덤 80/20 분할을 제거하고 전역 시간 기준 train/validation/test
분할을 사용한다. 날짜가 불완전한 행은 crawl 시각을 근거로 복원 여부와
신뢰도를 기록한다.

#### 2.3 Positive 편향

수집 데이터는 리뷰가 존재하는 interaction 중심이고 4점 이상 평점이 매우
많다. 관측되지 않은 식당을 단순 negative로 간주하면 exposure bias가 생긴다.
초기에는 같은 지역과 시점에서 popularity-matched negative sampling을 사용하되,
장기적으로 impression/click/save 로그를 직접 수집한다.

## 3. 목표 아키텍처

```text
Crawler / Application events
            │
            ▼
    Raw ingestion + crawl_run
            │
            ▼
PostgreSQL + pgvector + PostGIS
            │
            ├── Temporal dataset builder
            ├── Feature snapshot
            └── Item embeddings
            │
            ▼
Stage 1. Candidate retrieval
  ├── Regional popularity
  ├── Item-item co-occurrence
  ├── BPR / LightGCN
  ├── Two-tower retrieval
  └── Content vector retrieval
            │
            ▼
Candidate union / dedup / hard filters
            │
            ▼
Stage 2. Learning-to-rank
  └── LightGBM LambdaRank → 이후 DCN-V2 비교
            │
            ▼
Recommendation / impression log
```

현재 catalog는 약 748개이므로 전체 catalog scoring도 충분히 빠르다. 2-stage
구조는 당장의 latency 최적화보다는 향후 확장과 단계별 성능 분석을 위해
도입한다. Candidate 모델은 성능을 자동으로 높이지 않으며, candidate recall이
최종 ranker 성능의 상한이 된다.

지역, 영업 상태, 이미 방문한 식당 제외와 같은 hard filter는 Stage 1의 입력과
Stage 2의 출력에 적용하는 결정적 규칙이다. 별도 학습 모델이 아니므로 추천
stage 수에는 포함하지 않는다. 사용자는 중간 candidate를 보지 않고 Stage 2의
최종 Top-K만 받는다.

## 4. 기술 스택과 비용 원칙

| 영역 | 기본 선택 | 비용 |
|---|---|---:|
| 관계형 DB | PostgreSQL | 로컬 무료 |
| Vector 검색 | pgvector | 무료 |
| 공간 검색 | PostGIS | 무료 |
| DB migration | Alembic | 무료 |
| ORM / SQL | SQLAlchemy | 무료 |
| 모델링 | PyTorch, LightGBM | 무료 |
| 실험 관리 | 로컬 MLflow | 무료 |
| API | FastAPI | 무료 |
| Container | Linux Docker Engine / Compose | 무료 |
| Embedding | 로컬 multilingual encoder | 무료 |

관리형 PostgreSQL, 클라우드 GPU, 외부 embedding/LLM API, 지도 API의 무료
한도 초과, 유료 proxy 또는 관리형 Vector DB를 선택할 때만 외부 비용이
발생한다. v2의 기본 구현은 이 서비스들에 의존하지 않는다.

## 5. 데이터베이스 설계

지역별 테이블을 만들지 않고 하나의 정규화된 스키마에 모든 지역을 저장한다.

### 5.1 핵심 테이블

#### `crawl_run`

- `run_id`
- `source`
- `region`
- `started_at`, `completed_at`
- `source_file`
- `file_checksum`
- `row_count`
- `status`

#### `restaurant`

- `restaurant_id`
- `source_restaurant_id`
- `canonical_name`
- `address`
- `latitude`, `longitude`
- `region`, `category`
- `created_at`, `updated_at`

#### `app_user`

- `user_id`
- `source_user_key_hash`
- `created_at`, `updated_at`

원본 사용자명은 공개 모델링 테이블에서 직접 사용하지 않는다.

#### `review`

- `review_id`
- `user_id`, `restaurant_id`
- `rating`, `review_text`
- `taste`, `price`, `service`
- `reviewed_at`, `scraped_at`, `raw_date`
- `date_parse_quality`
- `content_hash`
- `crawl_run_id`

`content_hash`와 source 식별자에 unique constraint를 두어 같은 crawl을 다시
실행해도 데이터가 중복되지 않게 한다.

#### `interaction`

- `interaction_id`
- `user_id`, `restaurant_id`
- `event_type`: `impression`, `click`, `save`, `review`, `rating`
- `event_at`
- `request_id`
- `position`
- `metadata`

#### `restaurant_embedding`

- `restaurant_id`
- `model_name`, `model_version`
- `content_hash`
- `embedding`
- `created_at`

동일 식당도 encoder나 입력 문서가 바뀌면 새로운 version으로 저장한다.

#### `recommendation_log`

- `request_id`
- `user_id`, `restaurant_id`
- `candidate_source`
- `retrieval_score`, `ranking_score`
- `candidate_rank`, `final_rank`
- `model_version`
- `shown_at`

### 5.2 데이터 계층

```text
raw       원본 보존과 crawl provenance
staging   type casting, 날짜 parsing, 문자열 정규화
core      deduplicated users, restaurants, reviews, interactions
features  시점 기준 집계와 embedding metadata
serving   candidate와 recommendation 결과
```

CSV는 raw archive 및 초기 bootstrap 입력으로만 사용한다. DB 적재 후 notebook과
학습 코드는 `read_csv()`를 사용하지 않고 SQL query 또는 repository layer를
통해서만 데이터를 읽는다.

## 6. Stage 1: Candidate retrieval 계획

### 6.1 Baseline

- 지역별 popularity
- 평점 수와 최근성을 결합한 Bayesian popularity
- item-item co-occurrence
- implicit BPR 또는 ALS

### 6.2 Personalized retrieval

- LightGCN
- Two-tower user/item encoder
- in-batch negative와 hard negative 비교
- 전체 catalog exact retrieval과 ANN 결과 비교

### 6.3 Content retrieval

식당명, 메뉴, 지역, 카테고리 및 시점 이전 리뷰를 하나의 item document로
구성하여 multilingual encoder로 embedding한다. 사용자 embedding은 과거에
선호한 식당 문서의 가중 평균 또는 user tower로 생성한다.

초기에는 전체 vector를 exact search하고, catalog가 커지면 pgvector HNSW
index를 추가한다.

### 6.4 Candidate union

각 source의 후보를 합치고 source별 score를 그대로 보존한다.

```text
popularity candidates       20
item-item candidates        30
collaborative candidates    50
content candidates          50
--------------------------------
deduplicated candidate set  최대 100~150
```

수치는 첫 실험의 시작값이며 Recall@K와 latency에 따라 조정한다.

Stage 1의 출력 contract는 query마다 다음 필드를 가진 최대 N개의 행이다.

```text
query_id
user_id
restaurant_id
candidate_sources
source_scores
source_ranks
retrieved_at
candidate_model_version
```

초기값은 `N=100`으로 두고, candidate recall이 부족하면 150 또는 200으로
확장한다.

## 7. Stage 2: Learning-to-rank 계획

첫 ranker는 작은 데이터에서도 안정적이고 해석 가능한 LightGBM LambdaRank를
사용한다.

### 입력 feature

- candidate source별 score와 source 존재 여부
- user/item embedding similarity
- 거리와 지역 일치 여부
- 식당 popularity 및 최근성
- 사용자의 카테고리·가격·맛·서비스 선호 집계
- 최근 interaction과 식당 카테고리의 일치도
- 신규 식당 여부
- 리뷰 및 메뉴 content similarity
- candidate 단계의 rank

### Ranking group과 label

- group: `(user_id, recommendation_cutoff_at)`
- positive: 해당 cutoff 이후 관측된 held-out interaction
- explicit low rating: 낮은 relevance
- unobserved item: sampling된 약한 negative
- impression 이후 무반응: impression 로그가 쌓인 이후 negative 후보

초기 graded relevance 예시는 다음과 같다.

| 사용자 반응 | relevance |
|---|---:|
| 저장·높은 평점·명시적 재방문 | 3 |
| 클릭·평점 4점대 | 2 |
| 약한 interaction·평점 3점대 | 1 |
| 낮은 평점 또는 노출 후 무반응 | 0 |

서로 다른 이벤트를 하나의 label로 합칠 때에는 이벤트 정의와 가중치를
실험별로 versioning한다.

## 8. 평가 계획

### 8.1 데이터 분할

1. 날짜를 절대 시각으로 정규화한다.
2. 전역 시간 기준으로 train/validation/test 구간을 나눈다.
3. 각 query 시점보다 늦은 interaction은 feature 생성에서 제외한다.
4. 동일 사용자·식당의 중복 행이 서로 다른 split에 들어가지 않게 한다.
5. warm user와 cold-start user를 별도 보고한다.

### 8.2 평가 지표

Candidate 단계:

- Recall@20/50/100
- HitRate@20/50/100
- catalog coverage
- candidate source별 unique contribution
- retrieval latency

Ranking 단계:

- NDCG@5/10
- Recall@5/10
- MRR@10
- MAP@10
- coverage, novelty, intra-list diversity
- ranking latency

Rating RMSE는 보조 분석 지표로만 유지하고 최종 추천 모델의 주 지표로
사용하지 않는다.

### 8.3 초기 통과 기준

- Candidate Recall@100 목표: 0.95 이상
- Ranker가 candidate retrieval score 정렬보다 NDCG@10을 개선
- popularity baseline보다 personalized metric을 개선
- 개선 결과에 paired bootstrap confidence interval 보고
- 성능 개선이 coverage와 diversity의 심각한 하락을 동반하지 않을 것
- 재실행 시 동일 데이터 snapshot과 seed에서 결과 재현

목표값은 첫 leakage-free baseline 측정 후 현실적인 값으로 재조정한다.

## 9. Generative recommendation 연구 트랙

첫 generative candidate 모델은 TIGER 계열의 semantic ID generation을
대상으로 한다.

```text
Restaurant content embedding
          │
          ▼
RQ-VAE / residual quantization
          │
          ▼
Restaurant semantic IDs
          │
          ▼
User interaction sequence encoder
          │
          ▼
Autoregressive semantic-ID generation
          │
          ▼
Beam candidates → 동일한 LTR ranker
```

### 진행 조건

- leakage-free sequential dataset builder 완성
- 사용자별 시간순 interaction 보장
- conventional retrieval baseline 완료
- interaction 5회 이상 사용자 규모 확대
- semantic ID 충돌과 invalid ID 처리 방식 확정

### 비교 대상

- Popularity
- BPR / LightGCN
- Two-tower
- SASRec 계열 sequential baseline
- TIGER-style generative retrieval

HSTU와 OneRec은 연구 동향과 구조를 분석하되, 현재 데이터 규모에서는 핵심
구현 목표로 삼지 않는다. 먼저 공개 데이터에서 재현한 뒤 현재 데이터로
전이 가능한지를 판단한다.

## 10. Graph 활용 기준

Graph DB는 v2 초기 범위에서 제외한다. 사용자–식당 interaction graph는
관계형 DB에서 edge list로 추출하여 LightGCN 학습에 사용할 수 있다.

다음 관계가 충분히 축적되고 실제 multi-hop query나 설명 기능이 요구될 때
Neo4j 도입을 검토한다.

```text
사용자 → 방문/저장 → 식당
식당 → 제공 → 메뉴
식당 → 분류 → 음식 카테고리
식당 → 위치 → 상권/역/관광지
사용자 → 선호 → 맛/가격/분위기 속성
```

Graph DB 도입 여부와 knowledge graph 기반 추천 모델 도입 여부는 별개의
결정으로 관리한다.

## 11. 목표 프로젝트 구조

```text
rating_recsys/
├── README.md
├── RECOMMENDER_V2_PLAN.md
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── migrations/
├── src/
│   └── rating_recsys/
│       ├── config/
│       ├── db/
│       ├── ingestion/
│       ├── datasets/
│       ├── features/
│       ├── retrieval/
│       ├── ranking/
│       ├── evaluation/
│       └── serving/
├── tests/
│   ├── data_quality/
│   ├── integration/
│   └── unit/
├── notebooks/
│   └── experiments only
├── artifacts/
│   └── gitignored local outputs
└── legacy/
    └── v1_rating_prediction/
```

Notebook에는 핵심 ingestion, feature generation, metric 구현을 두지 않는다.
Notebook은 `src/`의 versioned 코드를 호출하여 결과를 탐색하고 시각화하는
용도로 제한한다.

## 12. 구현 마일스톤

### M0. v1 보존 및 기준선 기록

- [x] 기존 코드·데이터·모델을 `legacy/v1_rating_prediction/`에 보존
- [x] v2 계획 문서 작성
- [ ] legacy 파일 checksum 및 inventory 생성
- [ ] 기존 결과를 historical result로 명시

### M1. DB 기반 구축

- [ ] PostgreSQL + pgvector + PostGIS Docker Compose
- [ ] Alembic 초기 migration
- [ ] core schema 및 constraint 생성
- [ ] `.env.example` 작성
- [ ] health check와 DB integration test

### M2. 재실행 가능한 ingestion

- [ ] CSV bootstrap importer
- [ ] `crawl_run` provenance 기록
- [ ] text normalization
- [ ] stable content hash와 upsert
- [ ] 중복 검출 report
- [ ] 날짜 parsing 및 quality flag
- [ ] 사용자 식별자 hashing
- [ ] DB row count와 source reconciliation test

### M3. Leakage-free dataset과 baseline

- [ ] temporal split builder
- [ ] feature cutoff enforcement
- [ ] popularity baseline
- [ ] full-catalog evaluation
- [ ] warm/cold segment report
- [ ] MLflow dataset snapshot 및 metric 기록

### M4. Candidate retrieval

- [ ] item-item baseline
- [ ] BPR 또는 LightGCN
- [ ] Two-tower
- [ ] content embedding 생성
- [ ] pgvector retrieval
- [ ] candidate union과 source attribution
- [ ] Recall@K 및 latency 비교

### M5. Learning-to-rank

- [ ] candidate training table 생성
- [ ] negative sampling 전략 비교
- [ ] LightGBM LambdaRank
- [ ] feature ablation
- [ ] NDCG·coverage·diversity 검증
- [ ] full-catalog ranker와 2-stage recommender 비교

### M6. Serving과 피드백 수집

- [ ] FastAPI recommendation endpoint
- [ ] recommendation request log
- [ ] impression/click/save event schema
- [ ] model version과 feature snapshot 기록
- [ ] already-seen filtering
- [ ] 기본 monitoring

### M7. Generative retrieval

- [ ] 공개 데이터 TIGER 재현
- [ ] item semantic ID 생성
- [ ] warm-user subset 실험
- [ ] conventional candidate model과 동일 조건 비교
- [ ] 데이터 확대 여부 및 HSTU 검토

## 13. 핵심 실험표

| ID | Candidate | Ranker | 목적 |
|---|---|---|---|
| E0 | 지역 popularity | score sort | 최소 baseline |
| E1 | BPR/LightGCN | score sort | collaborative baseline |
| E2 | Two-tower | score sort | dense retrieval 기준 |
| E3 | 복수 candidate union | score normalization | recall 개선 측정 |
| E4 | E3 | LambdaRank | LTR의 순수 기여 측정 |
| E5 | E3 + content vector | LambdaRank | 리뷰·메뉴 정보 기여 측정 |
| E6 | TIGER-style GenRec | 동일 LambdaRank | generative candidate 기여 측정 |

모든 실험은 같은 temporal split, candidate evaluation protocol 및 ranking
label 정의를 사용한다.

## 14. 주요 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 불완전한 날짜 | temporal leakage | `raw_date`, `scraped_at`, parsing quality 보존 |
| crawl 중복 | 인기·평점 왜곡 | content hash, unique constraint, reconciliation test |
| positive-only 데이터 | noisy negative | 지역·시점 제약 sampling, impression 로그 도입 |
| 짧은 사용자 이력 | sequential 모델 과적합 | warm/cold 분리, content·popularity fallback |
| 현재 catalog가 작음 | 2-stage 이점 불명확 | full-catalog ranker를 반드시 함께 비교 |
| 사용자명 노출 | 개인정보 위험 | source key hashing 및 원본 접근 제한 |
| embedding 변경 | 재현 불가 | model/version/content hash 기록 |
| GenRec 연산량 | 실험 비용 증가 | 공개 데이터 재현 후 작은 모델부터 진행 |

## 15. 바로 시작할 첫 구현 단위

첫 구현은 M1과 M2의 최소 vertical slice로 제한한다.

1. PostgreSQL, pgvector, PostGIS를 실행하는 `docker-compose.yml`
2. `crawl_run`, `restaurant`, `app_user`, `review` migration
3. legacy CSV 한 개를 staging과 core table로 적재하는 CLI
4. 같은 파일을 두 번 적재해도 row count가 늘지 않는 integration test
5. DB에서 학습 입력을 읽는 repository 함수
6. source row count, inserted, updated, rejected, duplicated를 출력하는 적재 report

이 단위가 통과한 뒤 전체 지역 데이터 migration과 candidate 모델 구현으로
확장한다.

## 16. 완료 정의

v2의 첫 번째 안정 버전은 다음 조건을 모두 만족할 때 완료로 본다.

- 모델링 코드가 CSV를 직접 읽지 않는다.
- 모든 학습 데이터가 재현 가능한 DB snapshot에서 생성된다.
- temporal leakage를 검사하는 자동화 테스트가 있다.
- candidate와 ranking metric이 분리되어 기록된다.
- 추천 경로가 candidate retrieval과 learning-to-rank의 정확히 두 학습 stage로
  구성된다.
- popularity, collaborative, vector, LTR 실험 결과를 동일 조건에서 비교한다.
- 추천 결과에 사용된 model version과 candidate source를 추적할 수 있다.
- 유료 API 없이 로컬에서 전체 파이프라인을 실행할 수 있다.
