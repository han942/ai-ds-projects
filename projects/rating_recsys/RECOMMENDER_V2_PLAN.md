# Rating Recommender System v2 확장 계획

> 상태: M1·M2·M3 완료, M4·M5 초기 C0-C3/R1 baseline vertical slice 구현 완료
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
3. 사용자별 chronological split과 전역 temporal benchmark를 함께 사용하여
   leakage-free 평가 프로토콜을 확립한다.
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

## 2. 초기 데이터 진단 (2026-09-19)

아래 수치는 계획 작성 당시 보유 CSV의 프로파일이다. 2026-09-27 DB 기준
규모와 출처 변화는 [데이터 스냅샷 비교](./analysis/data_snapshot_2026-09-27.md)에 있다.
이 절의 수치를 현재 모델 성능이나 데이터 규모로 사용하지 않는다.

| 항목 | 값 |
|---|---:|
| 원본 행 | 31,085 |
| 익명 사용자 | 6,987 |
| 식당 | 748 |
| 적재 전 중복 행 | 7,866 |
| 중복 제거 후 리뷰 | 23,207 |
| 사용자별 고유 식당 중앙값 | 2 |
| 고유 식당 1개 사용자 | 3,461 |
| 고유 식당 3개 이상 사용자 | 2,396 |
| 고유 식당 5개 이상 사용자 | 1,347 |
| 평점 4점 이상 | 87.76% |
| 평점 2점 이하 | 2.35% |
| 연도가 없는 날짜 | 1,280건 |
| 상대 날짜 | 117건 |
| 날짜 해석 불가 | 1건 |

위 수치는 2026-09-19 Supabase 적재 결과를 기준으로 한다. 원본 31,085행은
리뷰 23,207건, 중복 7,866건, 필수값 누락 12건으로 reconciliation되었다.

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

#### 2.2 희소한 사용자 이력과 데이터 분할

사용자별 랜덤 분할은 미래 interaction이 과거 feature에 섞일 수 있으므로
사용하지 않는다. 반면 전역 시간 분할만 사용하면 test에 처음 등장하는
사용자가 많아 personalized model 평가가 cold-start 성능과 뒤섞인다. 당시
snapshot의 전역 80/10/10 시간 분할에서는 test 사용자 1,269명 중 593명
(46.7%)이 train에 존재하지 않았다.

따라서 다음 두 평가 프로토콜을 함께 사용한다.

1. **Primary seen-user benchmark**: `(user_id, restaurant_id)`별 최초 interaction만
   남기고, 고유 식당이 3개 이상인 사용자의 마지막 interaction을 test,
   마지막에서 두 번째를 validation, 나머지를 train으로 배치한다. train 이력이
   1개 이상인 사용자를 하나의 `seen user` 집단으로 평가한다.
2. **Secondary temporal audit**: 현재 구현된 전역 time cutoff로 seen/new user
   비율만 진단한다. rolling benchmark와 time-aware 모델 비교는 Future Work로
   둔다.

기존의 warm과 few-shot은 별도 모델 경로로 나누지 않는다. 두 집단을
`seen user`로 합쳐 동일한 personalized pipeline을 사용하고, train history
1~2개와 3개 이상 구간의 지표는 성능 진단용 breakdown으로만 유지한다.
`new user`는 popularity와 content/context fallback으로 평가한다. 사용자 정보와
과거 이력이 모두 없는 경우 collaborative personalization이 불가능하다는 점을
명시한다.

날짜가 불완전한 행은 crawl 시각을 근거로 복원 여부와 신뢰도를 기록한다.

#### 2.3 Positive 편향

수집 데이터는 리뷰가 존재하는 interaction 중심이고 4점 이상 평점이 매우
많다. 관측되지 않은 식당을 단순 negative로 간주하면 exposure bias가 생긴다.
초기 baseline은 candidate generator가 반환한 미관측 식당을 약한 negative로
사용한다. 지역·시점 matching과 exposure 보정은 시간·노출 정보를 다루는 후속
실험으로 분리하고, 장기적으로 impression/click/save 로그를 직접 수집한다.

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

초기 계획 당시 catalog는 748곳이었다. 2026-09-27 DB 조회에서는 4,587곳이므로
전체 catalog scoring의 처리 시간은 새 데이터로 재측정해야 한다. 2-stage
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
| DB migration | Ordered SQL migration runner | 무료 |
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

- **C0 global popularity**: interaction 수로 전체 catalog를 정렬한다. 개인화가
  없는 하한선이자 데이터·평가 파이프라인 검증용 candidate source다.
- **C1 item-item co-occurrence**: 사용자의 과거 방문 식당과 함께 등장한 식당을
  cosine-normalized co-occurrence로 검색한다. 작고 희소한 implicit-feedback
  데이터에서 학습 없이 개인화를 검증하기에 적합한 collaborative source다.
- **C2 region popularity**: 사용자의 과거 방문 지역 분포와 지역별 식당
  popularity를 이용해 지역 선호 후보를 만든다.
- **C3 quota RRF union**: C0·C1·C2의 raw score를 직접 더하지 않고 source rank
  기반 RRF로 결합한다. 기존 C0+C1 상위 후보를 일정 수 보존한 뒤 C2를 포함한
  확장 후보로 나머지를 채우며, 각 source score·rank·기여 여부를 보존한다.

Stage 1 평가는 C0, C1, C2, C3를 각각 남겨 source별 기여와 fusion 효과를
분리한다. C3에서 선택한 후보를 평가·저장하고, 동일한 후보를 R1에 전달한다.

BPR/ALS, LightGCN, two-tower와 content retrieval은 위의 비학습·근접 이웃
baseline이 정상 동작한 뒤 비교한다. 초기 계획 당시 catalog는 748곳이었고
2026-09-27에는 4,587곳이다. C0의 full-catalog 비교는 현재 규모의 처리
시간과 함께 평가한다.

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

**R0**는 candidate source 또는 fusion 순서를 그대로 사용하는 identity ranker다.
초기 learned ranker인 **R1**은 C3에서 선택한 후보를 입력으로 받아 source
score와 rank, cutoff 이전 popularity, item-item similarity, 사용자 history 길이
같은 표형 feature로 재정렬한다. 이는 복잡한 neural ranker를 도입하기 전에
candidate 개선과 ranking 개선을 분리해 측정할 수 있고, feature importance를
확인할 수 있다는 점에서 첫 LTR baseline으로 적합하다.

### 입력 feature

초기 R1에서는 candidate source별 score·rank·source 존재 여부, 식당 popularity,
item-item similarity, 사용자 history 길이와 사용자의 과거 지역 분포 대비
candidate 지역 affinity만 사용한다. 모두 query cutoff 이전 interaction에서
집계한다.

user/item embedding similarity, 거리, recency, 카테고리·가격·맛·서비스 선호,
신규 식당 여부와 review/menu content similarity는 각 데이터·모델이 추가될 때
feature ablation으로 도입한다. 특히 recency와 시간 감쇠 feature는 초기
baseline에서 제외하고 8.5절의 Future Work로 관리한다.

### Ranking group과 label

- group: `(user_id, recommendation_cutoff_at)`
- positive: 해당 cutoff 이후 관측된 held-out interaction
- explicit low rating: 낮은 relevance
- unobserved item: sampling된 약한 negative
- impression 이후 무반응: impression 로그가 쌓인 이후 negative 후보

review-only 데이터에서 사용하는 초기 graded relevance는 다음과 같이 고정한다.

| 관측 결과 | relevance |
|---|---:|
| 평점 4.0 이상 | 2 |
| 평점 3.0 이상 4.0 미만 | 1 |
| 평점 3.0 미만 또는 sampled unobserved item | 0 |

서로 다른 이벤트를 하나의 label로 합칠 때에는 이벤트 정의와 가중치를
실험별로 versioning한다. relevance가 모두 0인 query는 ranking metric에서
제외하고 그 수와 비율을 별도 data-quality metric으로 기록한다.

## 8. 평가 계획

### 8.1 데이터 분할

1. 날짜를 절대 시각으로 정규화한다.
2. 아직 방문하지 않은 식당 추천이라는 task에 맞게 동일 사용자·식당의 반복
   리뷰는 최초 interaction 하나로 축약한다.
3. Primary benchmark는 고유 식당이 3개 이상인 사용자에게 chronological
   leave-last-two-out을 적용한다. 마지막 interaction은 test, 마지막에서 두 번째는
   validation, 나머지는 train이다.
4. Primary metric은 train 이력이 1개 이상인 `seen user` 전체를 대상으로 한다.
   train history 1~2개와 3개 이상 결과는 별도의 diagnostic breakdown으로 함께
   기록하되, 모델과 serving 경로를 분리하지 않는다.
5. Secondary temporal audit는 전역 time cutoff에서 seen/new user 비율을
   출력한다. 초기 C0-C3/R1의 성능 비교에는 사용하지 않는다.
6. 각 query 시점보다 늦은 interaction과 target review에서 파생된 정보는 feature
   생성에서 제외한다.
7. 날짜 추론 방식과 dataset snapshot을 기록하여 같은 split을 재현한다.

### 8.2 평가 지표

Candidate 단계:

- **Recall@20/50/100**: held-out relevant item을 후보군이 얼마나 보존하는지
  측정하는 주 지표다.
- **Catalog coverage@100**: 전체 식당 중 candidate로 한 번 이상 등장한 비율이다.
- **Candidate source unique contribution@100**: 특정 source만 찾아낸 relevant
  item과 candidate 수를 기록한다.
- **Retrieval latency p50/p95**와 query당 candidate 수를 기록한다.

현재 primary split은 query당 held-out item이 하나이므로 Recall@K와 HitRate@K가
같다. 둘을 중복된 headline metric으로 보고하지 않고 HitRate는 Recall의 alias로
artifact에만 남긴다.

Ranking 단계:

- **NDCG@5/10**: graded relevance와 상위 순서를 함께 반영하는 주 지표다.
- **Recall@5/10**과 **MRR@10**: relevant item 포함 여부와 첫 relevant item의
  위치를 보여주는 보조 지표다.
- **Catalog coverage@10**, novelty와 intra-list diversity를 guardrail로 둔다.
- **Ranking latency p50/p95**를 기록한다.

query당 relevant item이 하나인 현재 구성에서는 AP가 reciprocal rank와 같아
MAP@10과 MRR@10도 사실상 중복된다. MAP은 impression 로그 또는 multi-positive
evaluation이 생긴 뒤 활성화한다.

같은 이유로 leave-one-out query의 NDCG는 정답 평점의 gain보다 정답이 놓인
순위에 주로 좌우된다. 초기 relevance 1/2는 LambdaRank 학습 가중치와 query
포함 여부에는 사용하지만, 진정한 graded ranking 성능을 주장하지 않는다.
여러 positive가 같은 query에 존재하는 future-window 또는 impression 기반
evaluation을 구축한 뒤 graded NDCG를 본래 의미로 해석한다.

Rating RMSE는 보조 분석 지표로만 유지하고 최종 추천 모델의 주 지표로
사용하지 않는다.

Candidate와 Ranking의 주 지표는 `seen user` 전체에서 계산한다. train history
1~2개와 3개 이상 breakdown은 희소 이력에 따른 성능 저하를 진단하는 용도로
함께 보고한다. `new user`에는 personalized model과 동일한 기준을 강제하지 않고
popularity/content fallback의 HitRate, coverage 및 다양성을 별도로 기록한다.

### 8.3 초기 통과 기준

- Candidate Recall@100 목표: 0.95 이상
- Ranker가 candidate retrieval score 정렬보다 NDCG@10을 개선
- seen user 전체에서 popularity baseline보다 personalized metric을 개선
- train history 1~2개 집단에서 성능이 급락하는지 별도 확인
- new user fallback 결과와 해당 집단의 전체 비중을 별도 보고
- 개선 결과에 paired bootstrap confidence interval 보고
- 성능 개선이 coverage와 diversity의 심각한 하락을 동반하지 않을 것
- 재실행 시 동일 데이터 snapshot과 seed에서 결과 재현

목표값은 첫 leakage-free baseline 측정 후 현실적인 값으로 재조정한다.

### 8.4 재현성 현황과 완료 조건

현재 구현된 장치는 다음과 같다.

- PostgreSQL을 모델링 데이터의 단일 source로 사용한다.
- ingestion file SHA-256, review content hash와 idempotent upsert가 구현되어 있다.
- interaction 정렬과 split tie-break가 결정적이며 split unit test가 존재한다.

다만 현재 `rating-recsys-dataset`은 DB의 최신 상태를 매번 다시 읽고 요약 JSON만
출력한다. 따라서 DB가 변경되면 과거 실험 입력을 완전히 복원할 수 없으며,
seed, dependency version, Git commit, feature schema, 학습 artifact와 MLflow run도
아직 연결되어 있지 않다. 즉 ingestion과 split의 반복 가능성은 확보했지만
모델 실험의 완전한 재현성은 아직 확보되지 않았다.

baseline 구현 시 아래를 완료 조건으로 추가한다.

1. 정렬된 canonical row와 schema version으로 SHA-256 `dataset_snapshot_id`를
   만들고, 실제 interaction snapshot도 immutable artifact로 보존한다. Parquet
   파일 checksum은 row digest와 별도로 기록한다.
2. repository query version, split configuration, relevance mapping, candidate K,
   RRF 상수와 random seed 42를 하나의 versioned experiment config로 저장한다.
3. Python 및 모든 transitive dependency를 lock file로 고정하고 실제 실행 환경의
   package 목록과 OS 정보를 manifest에 기록한다.
4. 모든 score tie는 `restaurant_id`로 결정하고 LightGBM seed와 deterministic
   option을 고정한다. 병렬 실행에서도 순서가 달라지지 않게 테스트한다.
5. Git commit과 feature schema version을 기록하고, 기본 strict mode에서는 dirty
   worktree 실행을 거부한다. 예외 허용 시 diff를 artifact로 함께 저장한다.
6. query, candidate, ranking 결과와 모델을 같은 MLflow run에 저장하고 dataset
   digest를 input metadata로 연결한다. digest만 남기지 않고 실제 snapshot을
   반드시 함께 보존한다.
7. 동일 snapshot·config·code commit에서 candidate 순서와 metric이 같은지
   end-to-end deterministic test로 검증한다.

### 8.5 시간 정보의 현재 범위와 Future Work

현재 시간 정보는 **평가 순서와 leakage 방지에만** 사용한다. `reviewed_at`을
기준으로 최초 user-item interaction을 선택하고 chronological leave-last-two-out과
global temporal audit을 구성한다. 이는 time-aware 추천 모델이 아니라 미래
interaction이 과거 입력에 들어가지 않도록 하는 평가 규칙이다.

현재 global temporal 구현에는 test cohort도 train history만 기준으로 분류하는
제약이 있다. validation에 처음 등장한 사용자가 test에서 다시 등장해도 new로
남을 수 있으므로, 이 audit을 정식 benchmark로 승격하기 전 test history를
`train + validation`으로 확장하는 수정과 회귀 테스트가 필요하다.

초기 C0-C3/R1 모델에는 아래 시간 feature와 알고리즘을 넣지 않는다.

- 최근 interaction에 더 큰 가중치를 주는 recency feature
- time-decay 또는 trend-aware popularity
- 시간 창 기반 item-item co-occurrence와 session model
- 요일·시간대·계절 feature
- rolling-window retraining과 temporal drift monitoring
- SASRec 등 순서·시간 의존 sequential recommender

이 항목들은 baseline 결과와 신규 snapshot이 확보된 뒤 Future Work로 진행한다.
첫 비교는 `static baseline → time-decay popularity/item-item → sequential model`
순서로 하고, 같은 global temporal/rolling benchmark에서 정확도와 drift를 함께
평가한다. 날짜가 `inferred_year`, `relative`, `unknown`인 interaction은 별도
cohort로 보고하여 시간 feature 효과와 날짜 추론 오류를 섞지 않는다.

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

- [x] Supabase PostgreSQL 연결 설정과 private `recsys` schema 설계
- [x] Ordered SQL 초기 migration 구현
- [x] core schema 및 constraint 구현
- [x] `.env.example` 작성
- [x] 실제 Supabase project에 migration 적용
- [ ] pgvector 및 PostGIS extension은 해당 feature 구현 시 활성화
- [ ] health check와 DB integration test

### M2. 재실행 가능한 ingestion

- [x] CSV bootstrap importer 구현
- [x] `crawl_run` provenance 기록 구현
- [x] text normalization
- [x] stable content hash와 set-based upsert
- [x] 중복 검출 report
- [x] 날짜 parsing 및 quality flag
- [x] 사용자 식별자 salted hashing
- [x] 5개 CSV dry-run 및 transformation test
- [x] 실제 Supabase에 5개 CSV 적재
- [x] DB row count와 source reconciliation 확인

### M3. Leakage-free dataset과 baseline

- [x] DB-backed user-item 최초 interaction dataset builder
- [x] seen-user chronological leave-last-two-out builder
- [x] global temporal benchmark와 seen/new cohort audit
- [x] immutable dataset snapshot과 digest
- [x] experiment config, seed, code/environment manifest
- [x] feature cutoff enforcement
- [x] popularity baseline
- [x] full-catalog evaluation
- [x] seen/new user report와 history-depth diagnostic
- [x] MLflow dataset snapshot 및 metric 기록

### M4. Candidate retrieval

- [x] item-item baseline
- [ ] BPR 또는 LightGCN
- [ ] Two-tower
- [ ] content embedding 생성
- [ ] pgvector retrieval
- [x] candidate union과 source attribution
- [x] Recall@K 및 latency 비교

### M5. Learning-to-rank

- [x] candidate training table 생성
- [ ] negative sampling 전략 비교
- [x] LightGBM LambdaRank
- [ ] feature ablation
- [x] NDCG·coverage·diversity evaluator
- [x] MLflow tracking과 Streamlit recommendation explorer
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
- [ ] seen-user subset 실험
- [ ] conventional candidate model과 동일 조건 비교
- [ ] 데이터 확대 여부 및 HSTU 검토

### M8. Time-aware recommendation (Future Work)

- [ ] temporal test cohort를 train+validation history 기준으로 수정
- [ ] time-decay popularity와 item-item 비교
- [ ] rolling-window evaluation과 retraining simulation
- [ ] 날짜 parsing quality별 metric breakdown
- [ ] temporal drift와 catalog freshness monitoring
- [ ] session/sequential baseline과 SASRec 계열 비교

## 13. 컴포넌트 표

Candidate는 `C`, ranker는 `R`로 표시한다.

| ID | Component | 단계 | 목적 |
|---|---|---|---|
| C0 | global popularity/full catalog | Candidate | 비개인화 최소 baseline |
| C1 | item-item co-occurrence | Candidate | 근접 이웃 개인화 기준 |
| C2 | region popularity | Candidate | 지역 선호 source의 단독 기여 측정 |
| C3 | C0+C1+C2 quota RRF union | Candidate fusion | 확장 candidate recall 측정 |
| C4 | BPR/LightGCN | Candidate | 학습형 collaborative retrieval 비교 |
| C5 | Two-tower | Candidate | dense retrieval 기준 |
| C6 | content vector | Candidate | 리뷰·메뉴 정보 기여 측정 |
| C7 | TIGER-style GenRec | Candidate | generative retrieval 기여 측정 |
| R0 | source/fusion score sort | Ranker | candidate 순서를 그대로 쓰는 기준선 |
| R1 | LightGBM LambdaRank | Ranker | LTR의 순수 재정렬 효과 측정 |

현재 실행은 C0~C3 후보를 각각 평가하고, R1은 C3 후보를 재정렬한다.

C0-C3/R1의 성능 비교는 같은 primary seen-user split, candidate evaluation
protocol과 ranking label 정의를 사용한다. Secondary temporal 경로는 초기에는
cohort audit만 수행하며 M8에서 정식 benchmark로 승격한다.

## 14. 주요 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| 불완전한 날짜 | temporal leakage | `raw_date`, `scraped_at`, parsing quality 보존 |
| crawl 중복 | 인기·평점 왜곡 | content hash, unique constraint, reconciliation test |
| positive-only 데이터 | noisy negative | candidate negative로 시작하고 impression 로그 도입 |
| 짧은 사용자 이력 | sequential 모델 과적합 | seen user로 통합하되 history-depth별 진단, content·popularity fallback |
| 현재 catalog가 작음 | 2-stage 이점 불명확 | full-catalog ranker를 반드시 함께 비교 |
| 사용자명 노출 | 개인정보 위험 | source key hashing 및 원본 접근 제한 |
| embedding 변경 | 재현 불가 | model/version/content hash 기록 |
| GenRec 연산량 | 실험 비용 증가 | 공개 데이터 재현 후 작은 모델부터 진행 |

## 15. 바로 시작할 첫 구현 단위

M1과 M2가 완료되었으므로 다음 구현은 M3의 평가 dataset vertical slice다.

1. DB에서 `(user_id, restaurant_id, reviewed_at, rating)` interaction을 읽는
   repository 함수
2. 동일 사용자·식당을 최초 interaction으로 축약하는 dataset builder
3. 고유 식당 3개 이상 사용자의 chronological leave-last-two-out 생성
4. query 시점 이후 interaction과 target-derived feature를 차단하는 leakage test
5. seen user 전체와 history 1~2개/3개 이상 breakdown을 출력하는 evaluator
6. global cutoff에서 seen/new user 비중을 출력하는 temporal audit
7. popularity 및 full-catalog baseline의 Recall@K와 NDCG@K 기록

이 단위가 통과한 뒤 M4 candidate 모델 구현으로 확장한다.

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

## 17. Baseline 선택 근거

- [Amazon.com recommendations: item-to-item collaborative filtering](https://doi.org/10.1109/MIC.2003.1167344)
- [Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [Adapting Boosting for Information Retrieval Measures (LambdaMART)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf)
- [LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [MLflow experiment and dataset tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781)
