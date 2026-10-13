# 실험 계획과 진행 현황

갱신: **2026-10-13 KST**

- 기준 모델: [BASELINE_MODEL.md](./BASELINE_MODEL.md)
- 방법별 결과: [리포트 모음](./reports/README.md)
- 프로젝트 실행: [README_KOR.md](./README_KOR.md)

<a id="지금-어디까지-진행했는가"></a>

## 진행 상황

후보 검색과 재정렬을 분리해 진행한다. Validation 관련 사용자 1,639명 중 909명은 C5 후보에 정답이 하나도 없어, 현재는 고정 후보의 순서를 바꾸는 작업보다 전체 식당에서 새 후보를 찾는 retrieval을 우선한다.

정답 식당이 후보 100개에 없으면 재정렬 모델이 아무리 순서를 잘 정해도 그 식당을 추천할 수 없다. 반대로 후보에 정답이 있는데 상위에 놓지 못한 경우는 재정렬의 문제다. 현재는 전자의 병목이 커 후보 회수율을 먼저 확인하며, 리뷰가 비슷한 식당을 찾는 것을 넘어 미래 만족도와 연결되는 후보를 더 찾는지 검증한다.

1. 기준 모델

   - C1 item-item과 C4 LightGCN으로 후보 검색
   - C1·C4 순위를 RRF로 합쳐 C5 후보 100개 생성
   - 16개 feature의 LambdaRank로 Top-10 재정렬

2. 리뷰 활용

   - BM25, E5, Two-Tower, Transformer, RLMRec, 다중 벡터, 텍스트 그래프 실험 완료
   - 행동 기준을 안정적으로 개선한 조건은 미확인
   - 현재 기준 모델에는 리뷰 미사용

3. 현재 연구 범위

   - 리뷰 텍스트를 이용한 전체 식당 후보 검색 개선
   - 주 지표는 Recall@100
   - 재정렬 후속 연구는 보류

4. 최근 완료

   - 동일 Two-Tower의 텍스트 유무 3seed 비교
   - 텍스트 식당 그래프 3seed 비교
   - 학습형 리뷰 mean/late/shuffle 3seed 비교
   - 확인된 코드 오류 6개 수정
   - 중단 재개의 완료 조건 재사용·실행 소스 보존·progress 원자 저장 보완
   - 의견 추출의 부정어 단어 경계 교정(v4)

5. 현재 판단

   - 기존 C5 유지
   - 수정 후 실데이터 재학습·재평가 필요
   - 반복 사용한 validation 밖의 독립 검증 필요

## 연구 질문과 평가 기준

새 구조를 도입하면 모델 구조와 입력 정보가 동시에 바뀔 수 있다. 그래서 구조 전체의 효과와 리뷰 정보 자체의 추가효과를 서로 다른 연구 질문으로 둔다.

| 연구 질문 | 비교에서 바꾸는 것 | 유지해야 할 것 |
|---|---|---|
| RQ1: 리뷰·식당 정보의 추가효과 | 텍스트 유무 또는 실제/shuffle 대응 | 행동 입력·모델 구조·학습 예산·평가 규칙 |
| RQ2: 추천 구조의 효과 | Two-stage·Two-Tower·그래프·다중 벡터 등 구조 | 이용 가능한 정보의 시간 경계·정답·후보 예산 |

예를 들어 새 Two-Tower가 C5보다 낮다고 해서 리뷰 정보 자체가 쓸모없다고 결론 내릴 수는 없다. 같은 Two-Tower에서 텍스트만 바꾼 대조는 정보의 효과를, C5와 새 모델의 비교는 전체 구성의 효과를 답한다. 현재 계획의 완료 여부와 후속 우선순위는 이 구분에 따라 정리한다.

현재는 리뷰 입력을 비교했으며, 메뉴·외부 식당 속성의 추천 효과는 미검증이다.

### 평가 지표

1. 후보 검색

   - 주 지표: **Recall@100**
   - 보조 지표: Recall@20/50, 후보 순서 Graded NDCG@10
   - 진단: 노출 쏠림, coverage, 실행 비용

2. 전체 Two-stage 시스템

   - 주 지표: **Graded NDCG@10**
   - 보조 지표: Recall@10/100, 4점 이상 Recall, 실제 저평점 포함률

3. 별점 예측

   - 지표: MAE·RMSE
   - 현재 baseline은 별점을 출력하지 않으므로 별도 과제로 평가

Recall@100은 미래 관련 정답 중 후보 100개에 포함된 비율을 사용자별로 평균한다. NDCG는 만족도 등급과 추천 위치를 함께 반영한다. 관측 만족도 쌍 순서 정확도는 보조 진단이며 모델 선택에는 사용하지 않는다.

### 고정 데이터와 시간 경계

모델 차이만 비교하려면 입력 시점과 정답 기간이 같아야 한다. 모든 비교는 같은 snapshot과 T1/T2를 기준으로 하며, cutoff 이후 정보는 입력에서 제외한다.

1. Snapshot

   - ID: `e7896add5b4b5939`
   - 첫 사용자–식당 상호작용: 88,554개
   - 사용자: 14,008명
   - 식당: 4,587곳

2. 시간 분할

   - Train: 70,883개, 2025-12-19(T1)까지
   - Validation: 8,934개, T1 이후 2026-05-04(T2)까지
   - Test: 8,737개, T2 이후 약 145일

3. 정확도 평가 사용자

   - Validation: 1,639명
   - Test: 1,573명
   - 과거 이력과 미래 관련 정답이 모두 있는 사용자

4. 제외한 신규 사용자

   - Validation: 1,139명
   - Test: 1,214명
   - 별도 cold-start 평가는 미실행

수집 리뷰 96,922개와 추천 interaction 88,554개는 서로 다른 단위다. 원문과 모델 입력의 차이는 [텍스트 데이터 조사](./reports/text_data.md)를 따른다.

### 이력 수에 따른 개인별 평점 기준

사용자마다 별점을 사용하는 습관이 다르므로 강한 만족 경계만 과거 평균에 따라 제한적으로 조정한다. 관련 정답의 최소 경계는 3점으로 유지해 후보 Recall의 의미가 바뀌지 않도록 한다.

1. 강한 만족 경계

   - 과거 이력 10건 미만: `h = 4`
   - 과거 이력 10건 이상: `h = clip(4 + 0.5 × (과거 평균 − 4), 3.5, 4.5)`

2. 정답 등급

   - `r ≥ h`: relevance 2, gain 3
   - `3 ≤ r < h`: relevance 1, gain 1
   - `r < 3`: relevance 0, gain 0

3. 입력 경계

   - 과거 평균에는 query cutoff 이후 평점을 넣지 않음
   - 관련 정답 경계는 계속 3점
   - 개인화 기준은 강한 만족과 약한 만족을 구분

10건 선택 근거와 예시는 [SATISFACTION_BASELINE.md](./SATISFACTION_BASELINE.md)에 있다.

### 같은 날짜 방문 처리의 의미

수집 데이터에는 같은 날짜의 방문 순서를 확정할 정보가 없다. 현재 Window 방식은 임의 순서를 만들지 않고 같은 평가 기간의 방문을 함께 정답으로 처리한다.

1. Window

   - 기간 시작 전 이력을 입력으로 사용
   - 이후 기간의 모든 방문을 함께 정답으로 사용
   - 현재 기준 학습 방식
   - 같은 날 방문도 함께 정답으로 보존

2. Prefix

   - 직전 이력으로 다음 방문 하나를 예측
   - 과거 재현용
   - 같은 날 `review_id` 순서를 실제 방문 순서로 가정하는 문제는 미해결

3. 순차 추천

   - 이력 순서와 전이를 encoder로 학습
   - SASRec과 텍스트 순차 모델은 미구현

Window는 3개월 동안 입력 이력을 고정한다. T2 refit에서는 겹친 부분 기간을 덧붙이지 않고 전체 Window를 다시 만든다.

<a id="지금까지-어떤-순서로-시도했는가"></a>

## 실험 이력

날짜는 KST 기준이다. Artifact의 `Z` 시각은 UTC다. 평가 기간·seed·학습 목표가 다른 결과를 하나의 성능 순위로 합치지 않는다.

1. 공동 방문·LightGCN·초기 리뷰 모델 — 9/28~10/2

   - 시도: C1, C4, C5, DeepCoNN, LambdaRank, 평균 shrinkage
   - 결과: C5 채택, DeepCoNN·shrinkage 미채택
   - 한계: 현재 Window·개인별 만족도와 완전히 같은 평가가 아닌 과거 기록

2. Window baseline — 10/4

   - Test C5 Recall@100: **20.1626%**
   - Test C5 후보 NDCG@10: **0.026590**
   - Test LambdaRank NDCG@10: **0.025467**
   - 이후 실험의 기준 조건으로 사용

3. BM25·Liquid 검색 — 10/5~7

   - BM25 test 단독 Recall@100: **5.2968%**
   - Liquid test 단독 Recall@100: **3.2757%**
   - C5 결합도 기준보다 낮음
   - 새 정답 유입보다 기존 정답 탈락이 큼

4. LLM 리뷰 속성·감정 — 10/6

   - 고활동 사용자 15명 진단
   - 같은 리뷰와 평점의 쌍 순서 일치: **74.2%**
   - 미래 추천 정확도는 미검증

5. E5 LTR·초기 Two-Tower — 10/7~8

   - E5 6피처 LambdaRank: 개선 불확실
   - 초기 Two-Tower validation Recall@100: **9.4419%**
   - 초기 Two-Tower test Recall@100: **9.1193%**
   - 구조와 텍스트 효과가 섞여 후속 통제 비교 진행

6. 리뷰 Transformer — 10/9~10

   - 리뷰 블록 attention, Transformer, Top-200 cross-attention
   - 같은 설정으로 10epoch까지 연장
   - 선택 epoch 2 검색 Recall@100: **9.6714%**
   - 선택 재정렬 NDCG@10: **0.007007**
   - 식당 쏠림은 줄었지만 최선 성능은 개선되지 않음

7. RLMRec/E5 정렬 — 10/10

   - LightGCN 행동 학습을 유지하고 E5 평균 표현 정렬 추가
   - Validation graph Recall@100: **19.4880%**
   - C1 결합 Recall@100: **19.6248%**
   - 새 LambdaRank NDCG@10: **0.028022**
   - 차이 구간이 0을 포함해 개선 미확정

8. RQ-VAE — 10/10

   - 리뷰 E5를 3단계 residual code로 압축
   - Payload: **144.63배 축소**
   - 원래 Top-10 이웃 overlap: **10.42%**
   - 압축 효과와 추천 성과를 분리해 해석

9. 동결 리뷰 mean·late 검색 — 10/11

   - 같은 review bank의 평균과 리뷰별 Top-3 matching 비교
   - Validation Recall@100: **4.5032% → 4.7866%**
   - C5 quota 결합은 낮아짐

10. 속성 채널·여러 취향 — 10/11

   - 맛·가격·서비스·분위기·양 × 긍정/부정 10채널
   - 취향 K1/K3 검색
   - 텍스트 단독 K3 Recall@100: **5.5003%**
   - C5 보완 효과는 미확인

11. 동일 Two-Tower 텍스트 대조 — 10/11

   - 조건: 행동만/행동+E5, seed 42/43/44, 고정 12epoch
   - 평균 Recall@100: **9.1229% → 9.3854%**
   - 차이: `+0.2625%p`, 95% 구간 `[-0.2390, +0.7743]`
   - 텍스트 추가효과 미확인

12. 텍스트 그래프·학습형 리뷰 검색 — 10/11

   - 행동 C4 평균 Recall@100: **18.4481%**
   - 실제 텍스트 그래프: **18.2001%**
   - 균형 교정 late: **18.4514%**
   - 실제 텍스트 대응의 추가효과 미확인

13. 코드 오류 수정 — 10/12

   - 의견 부정 범위, E5 token budget, 샘플 검증, BM25 부정어, Two-Tower 미참여 ID, LTR IDCG 수정
   - 전체 테스트: **512 passed, 1 skipped**
   - 수정 후 실데이터 재학습·재평가 미실행

<a id="이-진행에서-무엇을-알게-됐는가"></a>

### 주요 관찰

1. 행동 신호의 강점

   - 공동 방문과 방문 그래프가 텍스트 단독 검색보다 높음
   - 접근성·지역·취향의 개별 기여는 미분리

2. 후보 결합 손실

   - 텍스트가 새 정답을 찾더라도 기존 정답이 후보 밖으로 밀려남
   - 검색 신호와 결합 정책을 별도로 평가해야 함

3. 여러 취향의 가능성

   - K3가 텍스트 단독 검색을 개선
   - 행동 기준의 추가 이득과 일반화는 미확인

4. 표현 집중

   - 일부 학습형 리뷰 표현에서 사용자별 점수 차이가 작음
   - 학습 목표·본문 유무·표현 구조의 원인 비중은 미분리

5. 검증 재사용

   - 같은 validation에서 여러 가설을 반복 탐색
   - 현재 결과는 탐색 근거
   - 새 기간과 seed의 독립 재현 필요

## 다음 작업

아래 항목은 미실행 계획이다. 한 번에 하나의 가설만 변경한다.

1. 수정 코드 영향 검증

   - 고정: Snapshot, 정답, 학습 예산, 대조군
   - 확인: BM25 부정어와 Two-Tower ID 처리의 수정 전후 validation 차이

2. 미래 만족도 목표의 텍스트 학습

   - 고정: 행동 표현과 후보 예산
   - 비교: 실제 텍스트, shuffle, text-only, behavior-only
   - 확인: 과거 방문 BPR와 미래 만족 과제의 차이

3. 독립 시간 검증

   - 과거 rolling validation 또는 새 미래 holdout
   - 모델 선택 규칙과 지표 고정
   - 기존 test를 새 설정 선택에 재사용하지 않음

4. 표현 학습 확장

   - 원문 encoder 추천 목적 미세조정 또는 학습형 텍스트 이웃
   - 집계 구조·CF·예산 고정
   - 한 변경의 효과와 추가 비용 분리

현재 retrieval 범위에서는 LTR 재학습, 잔차 재정렬, 후보 내 cross-attention을 우선하지 않는다.

### 전체 로드맵의 구현 상태

아래 항목은 완료된 범위와 아직 이름만 검토한 범위를 구분한다. 리뷰 코드 생성처럼 일부 구성요소를 구현한 경우에도 전체 생성형 추천기가 완성된 것으로 해석하지 않는다.

1. 행동 후보 모델

   - 완료: item-item, LightGCN, 행동 Two-Tower
   - 남음: 독립 MF·bias MF, SASRec

2. 리뷰 검색

   - 완료: BM25, E5 평균, Transformer 검색, review matching, 취향 군집
   - 남음: 추천 목적 encoder 미세조정, 부정·속성 의미 검증

3. 행동+텍스트

   - 완료: RLMRec/E5, 속성 채널, 고정 텍스트 그래프, 학습 mean/late
   - 남음: 학습 중 그래프 갱신, 미래 목표 학습

4. 재정렬

   - 완료: LambdaRank, E5 6피처, Transformer Top-200
   - 보류: Jev, hard-negative, 검색 점수 잔차

5. 생성형 검색

   - 완료: 리뷰별 RQ-VAE 코드
   - 남음: 식당 Semantic ID와 ID 예측 모델

6. 목록 생성·OneRec

   - 현재 상태: 계획
   - 남음: 통합 검색·순위·선호 정렬

7. 외부 식당 정보·벡터 저장소

   - 완료: 메뉴·속성 수집 및 적재, pgvector·Milvus·Astra 검토
   - 남음: 추천 효과, 실제 운영 검색, 비용 비교

8. 서비스·평가 확대

   - 완료: 로컬 실험, dashboard, MLflow
   - 남음: 추천 API, 노출·클릭·방문 로그, 신규 사용자 평가

### 혼동하기 쉬운 용어

1. Two-stage

   - 후보 검색과 재정렬을 나누는 전체 파이프라인

2. Two-Tower

   - 사용자와 식당을 각각 인코딩하는 모델 구조

3. 시간 분할

   - 과거 입력과 미래 정답을 나누는 평가 규칙

4. 순차 추천

   - 이력 순서와 전이를 학습하는 방식
   - 시간 분할만으로 구현되는 것은 아님

5. 리뷰 다중 벡터

   - 리뷰마다 벡터를 유지해 비교
   - 토큰 단위 ColBERT 재현과 다름

6. 리뷰 코드

   - 리뷰의 압축 표현
   - 고유 식당 ID나 생성형 추천기와 다름

## 실험 공통 절차

반복 실험에서 데이터·정답·후보 예산이 달라지면 모델 효과를 분리하기 어렵다. 다음 절차는 비교 조건을 고정하고, 점 추정치만으로 채택 결론을 내리는 일을 막기 위한 공통 계약이다.

1. Snapshot, 본문 sidecar, cutoff, 정답, 평가 사용자, K를 고정한다.
2. 변경할 입력·구조와 고정할 학습 조건을 기록한다.
3. 과거 이력만으로 프로필·어휘·그래프·후보를 만든다.
4. 모든 사전 지정 조건과 seed를 보고한다.
5. 사용자별 paired 차이와 불확실성, 비용, coverage, 노출 쏠림을 확인한다.
6. 채택 전 새 기간에서 재검증한다.

- 후보 밖 정답과 본문 없는 관련 사용자를 분모에 유지한다.
- 3seed 비교는 사용자별 seed 평균 차이를 bootstrap한다.
- 사용자×seed를 독립 표본으로 합치지 않는다.
- 구조가 다른 모델의 차이는 텍스트만의 효과로 해석하지 않는다.

### 후보 모델 비교

1. 현재 구성 확인

   - C1 / C4 / C1+C4

2. C4 교체

   - C5 / X / C1+X

3. 검색 출처 추가

   - C5 / X / C1+C4+X

4. 입력 효과 분리

   - 같은 X의 행동만 / 행동+텍스트 / 텍스트 대응 shuffle

5. 공통 비교 CLI

   - 등록 모델: `lightgcn`, `deepconn`, `bm25`, `two_tower`
   - Validation Recall@100으로 설정 선택
   - 선택 후 T2 refit·test 수행
   - 새 LambdaRank는 학습하지 않음

6. Validation 전용 CLI

   - `two_tower_ablation_cli`
   - `text_retrieval_comparison_cli`
   - 고정 최종 epoch와 모든 seed 보고
   - 자동 모델 채택 없음

### Baseline 데이터 재사용

후보와 학습 행 생성은 모델 학습보다 오래 걸릴 수 있다. Snapshot과 생성 조건이 같을 때는 저장된 배열을 재사용하고, 정답·cutoff·후보 규칙이 바뀌면 영향을 받는 구간을 다시 만든다.

1. 재사용 자료

   - 원본 interaction·본문: `artifacts/snapshots/`
   - Train/refit 행: `artifacts/prepared/<snapshot-id>/train-*`, `refit-*`
   - 평가 후보: 같은 경로의 `validation-*`, `test-*`
   - 조건·checksum: `manifest.json`
   - 숫자 배열: `data.npz`

2. 후보·학습 행 재사용 가능

   - Ranker tree 수
   - leaves
   - learning rate
   - 새 텍스트 경로만 추가하는 실험

3. 재구성 필요

   - Prefix/Window 변경
   - 학습 label 변경
   - Snapshot, cutoff, 정답, K, RRF, LightGCN, feature, 전처리 변경

4. 캐시 옵션

   - `--no-cache`: 재사용과 저장 생략
   - `--rebuild-cache`: 같은 조건도 다시 생성
   - 현재 Window prepared 배열: 약 45 MB

과거 Prefix 준비의 21.3분→재사용 7.2초 기록은 모델 학습을 포함한 전체 비용이 아니다.

## 초기 비교와 과거 기준

1. 9/28 C5 초기 채택

   - C5 Recall@100: 20.16%
   - 당시 C3: 16.66%
   - 현재 개인별 Window와 완전히 같은 실험은 아님

2. 9/30 DeepCoNN

   - 단독: 8.69%
   - C5 결합: 18.44%
   - 미채택
   - 현재 원 run은 재검산 불가

3. 10/1 동점 처리

   - Ranker 점수가 같으면 C5 순서를 유지하도록 변경

4. 10/2 Prefix shrinkage

   - λ=0 NDCG@10: 0.029360
   - λ=10 NDCG@10: 0.028207
   - 차이 구간: `[−0.005006, +0.002692]`
   - λ=0 유지

5. 10/4 Window baseline

   - 이후 실험 기준
   - 과거 Prefix 점수와의 차이를 모델 개선으로 해석하지 않음

Prefix/Window/Window+raw rating 비교 코드는 있지만, 현재 개인별 정답으로 모든 조건을 재비교한 결과는 없다.

## 실행 방법

환경 준비와 표준 명령은 [한국어 README](./README_KOR.md#실행-방법)에 모았다. 방법별 재현 조건은 [리포트](./reports/README.md), 과거 파일 복원은 [artifacts/README.md](./artifacts/README.md)를 따른다.

1. 표준 baseline

   - `rating_recsys.experiments.cli`

2. 후보 모델 비교

   - `rating_recsys.experiments.compare_cli <model>`

3. 입력·캐시 점검

   - `rating_recsys.experiments.review_ltr --dry-run`

4. 동일 Two-Tower 텍스트 대조

   - `rating_recsys.experiments.two_tower_ablation_cli`

5. 텍스트 그래프·학습형 리뷰 비교

   - `rating_recsys.experiments.text_retrieval_comparison_cli`

6. 기존 구간 순위 재계산

   - `rating_recsys.experiments.segment_diagnostics`

7. 테스트

   - `.venv/bin/python -m pytest -q`

## 데이터 작업

추천 실험의 고정 Snapshot과 새 데이터 수집·적재는 별도 작업이다.

1. Schema 적용

   - 명령: `rating-recsys-migrate`
   - 변경 파일: `migrations/`

2. CSV 적재 점검

   - 명령: `rating-recsys-ingest --dry-run`
   - 기존과 같은 `USER_HASH_SALT` 필요

3. 새 크롤링 자료 적재

   - 진입점: `rating_recsys.ingestion.import_crawler`
   - 최신 전국 `.csv.partial`을 고정 복사해 사용

4. DB interaction 확인

   - 명령: `rating-recsys-dataset`
   - DB 작업에는 `.env`의 `DATABASE_URL` 필요

5. 원본 보존

   - 위치: `artifacts/snapshots/ingestion_sources/`
   - CSV와 `.import.json`의 hash 확인

<details>
<summary>새 크롤링 실행</summary>

```bash
.venv/bin/python -m pip install -r crawler/requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python crawler/diningcode_playwright.py --national-regions --max-restaurants 0
.venv/bin/python crawler/diningcode_playwright.py --national-regions --max-restaurants 0 --resume
```

- 중단 시 `.csv.partial`과 `.checkpoint.json`을 남긴다.
- 대표 음식 목록 수집은 사이트 전체 식당 수집을 뜻하지 않는다.
- `crawl_complete=true`도 모든 리뷰 수집 완료를 보장하지 않는다.
- 접속 제한이 나타나면 수집을 중단한다.

</details>

## 문서와 실행 결과의 역할

1. [README.md](./README.md)

   - 영어 프로젝트 소개와 빠른 실행

2. [README_KOR.md](./README_KOR.md)

   - 한국어 프로젝트 소개와 빠른 실행

3. [BASELINE_MODEL.md](./BASELINE_MODEL.md)

   - 기준 모델의 입력·feature·학습·평가 정의

4. [SATISFACTION_BASELINE.md](./SATISFACTION_BASELINE.md)

   - 개인별 정답과 최소 이력 10건의 근거

5. [reports/README.md](./reports/README.md)

   - 완료한 실험의 입구

6. `artifacts/comparisons/<model>/<run_id>/`

   - 원 설정, 가중치, 순위, 지표, 검산

7. [artifacts/README.md](./artifacts/README.md)

   - 로컬 보존·삭제·ZIP 복원 범위

DeepCoNN·Liquid·RQ-VAE 일부 원 artifact는 현재 접근할 수 없다. 문서 기록과 직접 재검산 가능한 결과를 구분한다.
