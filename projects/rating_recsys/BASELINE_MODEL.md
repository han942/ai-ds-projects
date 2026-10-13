# 기준 모델

문서 정리: **2026-10-13**

기준 실행: **2026-10-04**

관련 코드 수정: **2026-10-12**

현재 baseline은 방문·평점으로 C5 후보 100개를 검색하고, 16개 feature의 LambdaRank로 Top-10을 정하는 Two-stage 모델이다.

- 학습 단위: 3개월 Window
- 정답: 개인별 만족도 등급
- 리뷰·외부 식당 정보: 미사용
- 출력: 예측 별점이 아닌 추천 순위 점수

현재 연구는 [리뷰 텍스트 후보 검색](./reports/review_embeddings.md)에 집중한다. 기존 전체 시스템은 비교 기준으로 보존한다.

2026-10-12에 LambdaRank early stopping의 NDCG 분모를 수정했다. 아래 성능은 수정 전 결과이며 수정 후 재학습 결과는 없다.

## 후보 검색과 재정렬

Two-stage 구조는 전체 식당에서 유망한 후보를 빠르게 찾는 단계와, 소수 후보를 더 자세히 비교하는 단계를 분리한다. C1과 C4는 서로 다른 행동 관계를 사용하고, C5는 점수 척도가 다른 두 검색기를 순위 기반으로 결합한다.

| 구성 | 사용하는 관계·정보 | 출력과 주요 설정 |
|---|---|---|
| C1 item-item | 사용자가 방문한 식당과 함께 방문된 식당의 관계 | 공동 방문 기반 미방문 후보 |
| C4 LightGCN | 사용자–식당 방문 그래프; BPR로 방문을 미관측 식당보다 높게 학습 | 그래프 후보; 3층·64차원·20epoch·L2 `1e-4` |
| C5 RRF | 점수 척도가 다른 C1·C4의 순위 결합 | 방문 식당 제외·최대 100개; RRF 상수 60 |
| R1 LambdaRank | 같은 C5 후보의 방문·평점·지역 등 16개 feature | Top-10 순서; learning rate 0.05·seed 42 |

C1은 개별 방문 식당과의 공동 방문 관계를, C4는 그래프를 거친 관계를 학습한다. 두 검색기가 서로 다른 정답을 회수할 가능성이 있어 순위를 결합하지만, 후보 수를 100개로 고정하면 어떤 식당은 새로 들어오고 다른 식당은 빠진다. 결합 효과는 새 후보 수보다 정답의 유입·탈락과 최종 Recall로 확인한다.

LambdaRank는 이 후보 목록의 순서를 정하는 모델이다. 후보 밖 정답은 복구하지 못하며, 같은 후보에서 인기·평점·지역 등의 신호가 상위 만족도 순서를 개선하는지를 NDCG로 평가한다. 출력 점수는 식당의 예측 별점이 아니다.

C0 인기·C2 지역 인기·과거 C3는 C5 결합에 넣지 않는다. C0 후보의 존재·순위는 LambdaRank feature로만 사용한다.

### LambdaRank feature

LambdaRank는 후보를 새로 만들지 않는다. 아래 feature는 같은 C5 후보 안에서 인기, 공동 방문, 그래프 점수, 사용자 특성을 함께 비교해 순서만 바꾼다.

1. 인기·평점 4개

   - `popularity_score`
   - `item_average_rating`
   - `popularity_source_present`
   - `inverse_popularity_rank`

2. 공동 방문 4개

   - `item_item_sum_similarity`
   - `item_item_max_similarity`
   - `item_item_source_present`
   - `inverse_item_item_rank`

3. LightGCN 3개

   - `lightgcn_score_z`
   - `lightgcn_source_present`
   - `inverse_lightgcn_rank`

4. 후보 결합 2개

   - `rrf_score`
   - `candidate_rank_inverse`

5. 사용자·지역 3개

   - `user_history_length`
   - `user_average_rating`
   - `region_affinity`

LightGCN 점수는 한 query의 후보 안에서 표준화한다. 사용자·식당 평점은 기본적으로 단순 평균을 사용한다. 선택적 shrinkage는 두 평균 feature만 바꾸며 정답과 후보 검색은 바꾸지 않는다.

## 훈련부터 평가까지의 전체 흐름

Validation은 설정 선택에, test는 선택이 끝난 모델의 최종 평가에 사용한다. T2 refit은 validation 기간까지의 이력을 학습에 추가하되, test 결과를 보고 설정을 다시 고르지 않기 위한 단계다.

### 고정 데이터

1. Snapshot

   - `e7896add5b4b5939`
   - 88,554 interactions
   - 사용자 14,008명
   - 식당 4,587곳

2. 시간 경계

   - T1: 2025-12-19
   - T2: 2026-05-04

3. 분할

   - Train: 70,883 interactions
   - Validation: 8,934 interactions
   - Test: 8,737 interactions

4. Test 사용자

   - 과거 이력 사용자: 1,580명
   - 관련 정답 사용자: 1,573명

### 학습·평가 순서

1. Train

   - T1까지 3개월 Window 구성
   - 각 Window 시작 전 이력으로 후보·feature 생성
   - 기간 안의 여러 방문을 함께 정답으로 학습
   - LightGCN도 각 시작 시점의 과거 데이터로 학습

2. Validation

   - T1 이력으로 추천
   - `(T1, T2]` 방문과 비교
   - LambdaRank 설정과 트리 수 선택

3. Refit

   - 선택한 설정·트리 수 고정
   - T2까지 Window와 학습 행을 다시 구성
   - LightGCN과 LambdaRank 재학습

4. Test

   - T2 이력으로 추천
   - 이후 약 145일의 실제 방문과 비교
   - Test에서 설정을 다시 선택하지 않음

- 미래 방문·평점은 정답으로만 사용한다.
- 미래 평점을 평균 feature에 반영하지 않는다.
- 후보 밖 정답을 후보 목록에 삽입하지 않는다.
- 같은 생성 조건의 후보·feature·label·group은 `artifacts/prepared/`에서 재사용한다.

## 학습 정답과 동점 처리

모델은 미래 방문 여부만이 아니라 그 방문의 만족도 강도도 학습한다. 다만 관측되지 않은 식당의 실제 만족도는 알 수 없으므로, 학습 label 0과 실제 저평점을 같은 의미로 해석하지 않는다.

1. 기준 학습 모드

   - Ranker: `window/relevance`
   - 만족도: `history-aware`
   - CLI 기본 `prefix/relevance + absolute`는 과거 재현용

2. 강한 만족 경계

   - 이력 10건 미만: `h = 4`
   - 이력 10건 이상: `h = clip(4 + 0.5 × (과거 평균 − 4), 3.5, 4.5)`

3. Window label

   - `r ≥ h`: label 2, gain 3
   - `3 ≤ r < h`: label 1, gain 1
   - `r < 3`: label 0, gain 0
   - 후보 목록의 미관측 식당: label 0, gain 0

4. 해석 제한

   - 관측 저평점과 미관측 후보가 학습 label에서는 모두 0
   - 미관측 후보를 실제 저평점으로 해석하지 않음
   - 모델은 순위 점수를 출력하며 별점을 예측하지 않음

5. 동점 처리

   - LambdaRank 점수가 같으면 C5 순서 유지

최소 이력 10건의 근거는 [SATISFACTION_BASELINE.md](./SATISFACTION_BASELINE.md)를 따른다.

## 설정과 트리 수 선택

트리가 너무 적으면 관계를 충분히 학습하지 못하고, 너무 많으면 validation에 과적합할 수 있다. Early stopping은 validation 지표가 더 좋아지지 않을 때 학습을 멈추고 트리 수를 선택하는 절차다.

1. 탐색 설정 6개

   - `num_leaves`: 15 / 31 / 63
   - `min_child_samples`: 10 / 100
   - 최대 1,000 trees
   - Validation NDCG@10이 50회 연속 개선되지 않으면 종료

2. 고정 기준 1개

   - leaves 15
   - min child 10
   - early stopping 없이 150 trees

3. 최종 선택

   - 7개 설정의 전체 validation NDCG@10 비교
   - 동점이면 Recall@10
   - 다시 동점이면 grid 순서
   - 선택 후 T2까지 refit

4. 2026-10-12 수정

   - Early stopping·설정 선택·보고가 모두 미래 기간 전체 NDCG 사용
   - 후보 밖 정답까지 IDCG에 포함
   - 관련 정답이 있지만 후보에 정답이 없는 사용자도 0점으로 분모 유지
   - 후보가 비어 있는 사용자도 분모 유지
   - 동점은 C5 순서 유지

기존 실행의 early stopping은 검색된 positive가 있는 group의 후보 내부 NDCG를 사용했다. 저장된 결과와 현재 코드의 학습 절차는 이 부분이 다르며, 수정 후 재학습·성능 비교는 미실행이다.

## 새 기준의 실행 결과

2026-10-04 Window·개인별 만족도, seed 42의 기존 실행이다.

1. 후보 검색

   - Recall@100: **20.1626%**
   - C5 후보 순서 NDCG@10: **0.026590**

2. 최종 추천

   - Graded NDCG@10: **0.025467**
   - Recall@10: **3.6502%**
   - 4점 이상 Recall@10: **3.5105%**
   - 3점 미만 방문의 Top-10 포함: **3 / 43 (6.98%)**

3. 선택 설정

   - leaves 63
   - min child samples 100
   - trees 76
   - T2 refit: 8,043 groups, 804,232 feature rows

4. 만족도 기준 적용

   - 개인 기준: 862명
   - 절대 기준: 718명

5. 재정렬 차이

   - R1−R0 NDCG@10: `−0.001124`
   - 사용자별 paired 95% CI: `[−0.005053, +0.002694]`
   - 구간이 0을 포함해 재정렬 이득 미확인

근거: [전체 실행 보고서](./artifacts/runs/20261004T125314601899Z-e7896add/report.md), [선택 근거](./SATISFACTION_BASELINE.md), [후보·지역 진단](./reports/baseline.md).

## 평가 지표와 집계 범위

Recall은 미래 관련 식당을 추천 목록이 얼마나 회수했는지 측정한다. NDCG는 관련 식당의 만족도 등급과 추천 위치를 함께 반영한다. Precision은 추천 중 정답의 비율, MAP·MRR은 정답이 앞쪽에 배치됐는지, coverage는 얼마나 다양한 식당이 노출됐는지를 본다.

1. 후보 검색

   - Recall@20/50/100
   - 4점 이상 Recall
   - 분모: 해당 정답이 있는 사용자

2. 최종 추천

   - Graded NDCG, Recall, Precision, MAP, MRR@5/10
   - 분모: 관련 정답이 있는 사용자

3. 저평점 진단

   - 실제 3점 미만 방문의 Top-K 포함 수와 비율

4. 추천 분포

   - Coverage
   - Novelty
   - 지역 다양성

노출 로그가 없으므로 결과는 기록된 미래 방문과의 일치다. 실제 노출 후 만족도나 인과 효과를 증명하지 않는다.

## 과거 Prefix 측정 (2026-10-02)

현재 Window·개인별 기준과 직접 비교하지 않는 과거 실행이다.

1. Baseline λ=0

   - 후보 Recall@100: 20.1626%
   - NDCG@10: 0.029360
   - Recall@10: 4.1284%
   - 선택 trees: 1

2. Shrinkage λ=10

   - NDCG@10: 0.028207
   - Recall@10: 4.0369%
   - 선택 trees: 96

3. 차이

   - NDCG: `−0.001153`
   - Paired 95% CI: `[−0.005006, +0.002692]`
   - 구간이 0을 포함해 λ=0 유지

λ와 seed가 각각 하나이므로 모든 평균 보정의 효과를 단정하지 않는다. 1 tree 선택의 원인도 미확정이다.

## 과거 측정 조건과 추가 결과 (2026-10-02)

1. 학습 규모

   - 15,929 groups
   - 1,574,652 rows
   - leaves 15, min child 10, 1 tree

2. 추가 test 지표

   - NDCG@5: 0.023906
   - Precision@10: 1.5639%
   - MAP@10: 0.015684
   - MRR@10: 0.051154

3. 해석

   - 2026-09-30 run과 섞지 않음
   - 2026-10-12 IDCG 수정 후 재학습 결과가 아님

근거는 [과거 기록과 원본 위치](./reports/baseline.md#근거)에 있다.
