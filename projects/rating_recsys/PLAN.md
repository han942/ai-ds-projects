# V2 Plan & Experiment Notes

기준일: 2026-10-04. **전체 범위는 Two-stage baseline → 후보 모델 비교 → 텍스트 결합 → 생성형 추천·OneRec 통합 실험이다.**
현재 구현은 [C5 후보 검색 + R1 LambdaRank](./BASELINE_MODEL.md)이며 텍스트·Two-Tower는 미적용이다.
학습 방향은 **기준 날짜 이전 이력 → 이후 일정 기간의 모든 방문을 한 정답 목록으로 묶는 Window 방식**이다.
개인별 만족도 기준은 구현했다. 최소 이력 10건은 train의 평균 안정성과 적용 범위를 보고 정했다.
규칙·분석 근거·실행 결과는 [SATISFACTION_BASELINE.md](./SATISFACTION_BASELINE.md)에 기록한다.
CLI 기본값은 과거 재현을 위해 Prefix/absolute이며, 이번 Window baseline에는 개인별 기준 옵션을 명시한다.
2026-10-04 전체 실행과 독립 재계산을 완료했다. Test NDCG@10은 0.025467이며, 실행 상세는 위 기록을 따른다.
기존 평가 기준과 새 기준의 점수 차이로 기준의 우열을 판단하는 실험은 진행하지 않는다.
같은 날짜 Prefix 이력 수정은 Prefix를 별도 비교할 때의 과제이며 Window 실행의 선행 조건이 아니다.

## 연구 질문과 평가 기준

**RQ1:** 사용자 과거 리뷰와 식당 정보를 추가하면, 방문·평점 이력만 쓴 baseline보다 사용자가 만족할 만한 식당을 더 잘 추천하는가?
**RQ2:** 이 추천 과제에서 Two-stage, Two-Tower, Generative Retrieval 등 어떤 모델 구성이 가장 효과적인가?

RQ1은 제품 목표다. RQ2는 같은 목표·데이터·평가 규칙 아래 모델 구성을 비교해 답한다. 리뷰나 식당 정보를 넣으면 성능이 좋아진다고 미리 가정하지 않는다.

### 테스트 사례와 평점 처리

- 전역 시간 cutoff 이전 데이터만 입력으로 쓴다. Test에는 cutoff 당시 사용자별 이력에 없던 식당을 이후 방문해 실제 평점을 남긴 사용자–식당 쌍이 들어간다. 사용자는 과거 방문 이력이 있는 개인화 대상이며 target pair는 학습 입력에 없고 test에 실제 평점이 있다. 테스트 행을 평점 기준으로 골라내거나 삭제하지 않는다.
- 현재 주 test는 T2 이후 145일 동안의 여러 방문을 대상으로 한다. 바로 다음 방문 하나를 맞히는 평가가 아니다. 다음 방문 성능은 별도 event-level 평가로 보고한다.
- 사용자 데모그래픽 정보는 현재 데이터에 없으며 필수 입력으로 가정하지 않는다. 사용자 취향은 cutoff 이전 본인 리뷰·평점에서, 식당 쪽 정보는 cutoff 이전 리뷰와 그 시점에 확보한 메뉴·식당 속성에서 만든다. 입력별 데이터 커버리지와 결측 fallback을 기록한다.
- 식당 메뉴·리뷰·속성도 추천 cutoff 당시 알 수 있었던 값만 쓴다. 현재 수집한 메뉴를 과거 시점에 알려져 있던 정보처럼 사용하지 않는다.
- 실제 미래 평점을 사용해 만족도 순위 정답을 만든다. 과거 이력이 10건 미만이면 기존 절대 기준을 쓰고, 10건 이상이면 강한 만족 경계에 과거 사용자 평균을 반영한다. `--satisfaction-mode history-aware`로 적용한다. 고평점만 모아 테스트셋을 만들지 않는다.
- 모든 모델은 선택한 동일 만족도 정답으로 NDCG·Recall을 평가한다. 4점 이상 방문 Recall과 저평점 방문의 추천 포함 수·비율은 보조 진단이며, 별도 기준의 점수 크기로 평가 기준 자체의 우열을 판단하지 않는다. 저평점 방문도 테스트에 남긴다.
- 별점 예측은 실제 테스트 평점이 있는 모든 사용자–식당 쌍(저평점 포함)에 대해 MAE/RMSE를 계산한다. 이 지표는 방문했다는 조건 아래의 별점 예측을 평가하며, 추천 순위 지표를 대신하지 않는다.

### 이력 수에 따른 개인별 평점 기준

**선택한 공통 평가 방향:** 방문·평점 이력이 적으면 기존 절대 평점 기준을 유지하고, 이력이 충분하면
사용자의 과거 평균 평점을 활용해 평소보다 높게 평가한 식당인지 함께 판단한다.
개인별 정답은 구현되어 있으며 label 0/1/2와 gain 0/1/3을 학습·평가에서 함께 사용한다.

| 추천 시점의 과거 이력 | 만족도 정답 구성 방향 |
|---|---|
| 10건 미만 | 기존 기준 유지: 4점 이상 강한 정답, 3점대 약한 정답 |
| 10건 이상 | 강한 만족 경계 `clip(4 + 0.5 × (과거 평균 − 4), 3.5, 4.5)`; 약한 경계는 3점 유지 |

예를 들어 같은 4점이라도 과거 평균이 3점인 사용자에게는 평소보다 1점 높은 평가이고,
평균이 4.6점인 사용자에게는 평소보다 0.6점 낮은 평가다. 평균 대비 차이는 개인의
평가 성향을 참고하는 신호이며, 그 자체를 만족·불만족의 확정 판정으로 쓰지는 않는다.

- 평균과 이력 수는 각 추천 시점 이전 평점으로만 계산한다. 미래 평가 평점을 평균에 넣지 않는다.
- 최소 관측 수는 train 분석으로 10건을 선택했다. 단순 과거 평균을 사용하고 강한 경계의 이동을
  ±0.5점으로 제한한다. 분산으로 나누지 않아 상수 평점도 처리하며, 강한 경계 상한은 4.5점이다.
  세부 근거와 예시는 [개인별 만족도 설명](./SATISFACTION_BASELINE.md)에 있다.
- 현재 ranker에는 사용자 평균 평점 feature가 이미 있다. 이를 정답 기준에 쓰는 것은 별도 변경이다.
  사용자 평균을 모든 후보 점수에서 똑같이 빼기만 하면 후보 순위는 변하지 않는다.
- 평가 기준은 과제의 정의이며 점수가 더 높은 기준을 고르는 실험 대상이 아니다.
  선택한 만족도 정답을 구현·고정한 뒤 모든 모델을 같은 정답·사용자 범위·분모·cutoff에서 평가한다.
  기존 기준으로 계산한 과거 점수는 기록으로 보존하며 새 결과와 직접 비교하지 않는다.
  이전 모델을 비교할 때도 같은 새 기준으로 다시 평가하고, 모델 선택은 같은 기준의 validation을 쓴다.

### 첫 실행의 순서

1. 개인별 정답 규칙을 구현했다. 최소 이력 10건·평균 가중치 0.5·최대 이동 0.5점을 고정한다.
2. Window의 미래 방문 목록에 그 규칙으로 정답을 만들고, 학습 label·validation 선택·test 평가가
   같은 만족도 정의를 쓰는지 확인한다. 평균은 각 query의 과거 이력으로만 계산한다.
3. 현재 C5 + LambdaRank 구조를 Window 방식으로 학습·평가해 새 과제의 baseline을 측정한다.
4. 이후 리뷰·식당 정보 추가와 모델 구조 변경을 같은 고정 평가 기준에서 비교한다.

기존 `window/relevance` 명령만 실행하면 개인별 만족도 기준은 적용되지 않는다.
`--satisfaction-mode history-aware`를 함께 지정한다.
기존 기준의 Window를 먼저 측정하거나 Prefix와 비교하는 것은 이번 첫 실행의 목표가 아니다.

### 지표가 답하는 질문

| 평가 층위 | 지표 | 해석 |
|---|---|---|
| 후보 검색 | Recall@20/50/100, 우선 4점 이상 target Recall | 미래에 만족한 식당을 재정렬 전에 후보로 찾았는가? |
| 전체 추천 | Graded NDCG@5/10, Recall@5/10 및 4점 이상 Recall | 만족한 식당을 최종 목록의 상위에 배치했는가? |
| 별점 예측 | MAE/RMSE, 실제 평점이 있는 모든 test 쌍 | 실제 방문한 식당에 사용자가 줄 평점을 얼마나 정확히 예측했는가? |
| 저평점 점검 | Top-K에 든 실제 3점 미만 test 방문의 수·비율 | 나중에 낮은 점수를 준 식당에 높은 만족 점수를 주는 경향이 있는가? |

RQ1/RQ2의 주 비교 지표는 end-to-end Graded NDCG@10이다. Validation에서 이 지표로 설정을 고르고, test는 최종 비교에만 쓴다. 후보 Recall@100과 4점 이상 Recall@10은 검색·고평점 적중을 진단하며, 별점 MAE/RMSE는 보조 성과로 보고한다.
별점 예측 점수로 후보를 정렬할 수는 있지만, 낮은 MAE가 좋은 Top-K 순위를 보장하지 않는다.
테스트 기간에 4점 이상 방문이 없는 사용자는 테스트에서 삭제하지 않는다. 별점·저평점 분석에는 포함하고, 4점 이상 정답을 전제로 하는 Recall의 대상 사용자 수와 제외 이유를 함께 기록한다.

미래 방문이 기록되지 않은 다른 후보에는 실제 평점이 없다. 이를 사용자가 싫어한 식당으로 해석하지 않는다. 오프라인 순위 지표에서 그 항목에 gain을 주지 않는 경우에도, 관측된 저평점과 미관측 후보는 원자료와 진단 결과에서 구분한다.

### 모델 비교 순서

1. **정보 효과를 분리한다.** 같은 Two-stage 구조·cutoff·후보 조건에서 평점·방문 이력 baseline, 사용자 과거 리뷰 추가, 식당 정보 추가, 둘 다 추가를 비교한다.
2. **구조 효과를 비교한다.** 유효한 정보 입력과 동일한 시간 분할·정답 정의를 사용해 Two-stage, Two-Tower, Generative Retrieval 등을 비교한다. 후보 생성 모델은 Recall, 최종 시스템은 end-to-end 순위와 별점 지표를 보고한다.
3. **재정렬 효과를 분리한다.** Retriever 후보를 고정해 R0와 LambdaRank 등 재정렬기를 비교한다. 이 조건은 RQ1의 전체 추천 성과를 설명하는 구성 요소이지, 후보 검색 성능을 포함한 전체 결과와 혼동하지 않는다.

모든 feature와 텍스트는 cutoff 이전 정보에서 만든다. 앞으로의 학습은 기준 날짜까지의 이력과
그 이후 정해진 기간의 방문 목록을 분리하는 Window 방식으로 진행한다. 같은 날짜의 방문은
같은 정답 목록에 두며, 방문 순서를 정하거나 원본 행을 삭제할 필요가 없다.

### 같은 날짜 방문 처리의 의미

**Window 방식에서는 날짜 경계로 입력과 정답을 나누므로 당일 방문 순서 처리가 필요 없다.**
예를 들어 9월 30일까지의 이력으로 후보를 만들고 10월 1일~12월 31일 방문을 정답 목록에
넣으면, 10월 1일에 방문한 A·B는 모두 그 목록에 들어간다. A·B의 평점은 정답의 만족도
등급에 사용하고, 해당 query의 후보·feature·사용자 평균을 만드는 데는 쓰지 않는다.
후속 기간의 query에서는 그때까지 관측된 A·B를 과거 이력으로 사용할 수 있다.

이전에 설명한 날짜 처리는 **다음 방문 하나를 학습하는 Prefix에만 해당하는 문제**다.
방문 데이터에는 날짜가 있지만 당일의 정확한 방문 순서는 없다. 현재 Prefix 코드는
날짜가 같으면 `review_id`로 정렬해 앞선 행을 과거 이력에 넣는다. 이 ID 순서가 실제
방문 순서라는 보장은 없어서, 당일의 나중 방문을 입력에 사용했을 가능성을 배제할 수 없다.

예를 들어 10월 1일 A·B를 방문했다면 B를 학습 정답으로 삼을 때 A가 B보다 먼저였다고
가정하지 않고 9월 30일까지의 이력만 사용한다. A·B 원본 행은 유지하며 각각 정답으로
쓸 수 있고, 10월 2일부터는 둘 다 과거 이력에 포함된다. 전날까지 이력이 없는 행은
개인화 학습 query를 만들지 못할 수 있으므로 query 수 변화도 기록한다.

Prefix를 비교 실험으로 유지한다면 입력 이력과 후보 생성 context를 함께 수정해야 한다.
이 수정은 아직 미구현이며, Window로 진행하는 데 필요하지 않다. 전역 날짜로 나눈
validation/test 방문을 삭제하거나 같은 날짜 방문 전체를 버리는 작업도 아니다.

## 전체 프로젝트 범위

각 확장의 추가 효과를 분리해 비교하고, 순위·별점 성능과 학습 비용·추론 비용·재현 조건을 기록한다.

| 단계 | 비교할 구성 | 핵심 질문 | 상태 |
|---|---|---|---|
| 0. 새 과제의 baseline | 개인별 만족도 정답 + Window + C5/LambdaRank | 선택한 과제에서의 기준 성능은 무엇인가? | 2026-10-04 구현·전체 실행·독립 지표 검증 완료 |
| 1. 텍스트 없는 후보 모델 | Item-item, MF, Two-Tower, LightGCN, SASRec | 협업 관계·그래프·방문 순서 중 어떤 신호가 검색에 도움이 되는가? | Item-item·LightGCN 구현, 나머지 미구현 |
| 2. 리뷰 텍스트 후보 검색 | BM25, 리뷰 임베딩 검색, C5와의 결합 | 어휘 검색·의미 검색이 C5의 후보 Recall을 보완하는가? | BM25·임베딩 미실험, 과거 DeepCoNN 결과만 있음 |
| 3. 텍스트 학습·재랭킹 | 텍스트 feature·Two-Tower, LambdaRank와 Jev 비교 | 학습형 ranker와 decision-model 재랭커가 순위를 개선하는가? | 후속 설계 |
| 4. Generative Retrieval | TIGER 계열의 Semantic ID 후보 생성 + 공통 ranker | 벡터 검색을 ID 생성으로 바꾸면 후보 품질·비용이 달라지는가? | 미구현 |
| 5. 리스트 생성형 추천·OneRec | 추천 목록 생성, 검색·순위 통합, 선호 정렬 | 분리된 Two-stage와 통합된 생성형 모델은 어떻게 다른가? | 미구현 |

이 순서는 실험의 확장 경로이며 성능 향상을 전제하지 않는다.
OneRec은 생성형 추천의 한 방식이다. 단순 후보 생성과 검색·순위 통합의 차이를 확인하도록 단계를 나눴다.
입력 확장은 우선 행동·평점 + 리뷰 텍스트까지이며, 이미지·음성은 현재 범위에 추가하지 않는다.

### 시간 분할·순차 추천·모델 구조의 구분

| 실험 축 | 의미 | 현재 baseline |
|---|---|---|
| 시간 분할 | 과거 정보로 미래를 평가하는 데이터 구성 | 전역 T1/T2 사용 |
| 순차 모델링 | 방문 순서·전이·최근 행동을 모델 입력으로 학습 | 방문 집합·정적 그래프·집계 feature 사용 |
| Two-stage | 후보 검색과 최종 순위를 별도 단계로 계산 | 적용 |
| Two-Tower | 사용자·식당 encoder로 벡터를 따로 계산하고 매칭 | 미적용 |
| 입력 정보 | 행동·평점·텍스트 등 사용할 신호 | 행동·평점만 사용 |

다음 방문을 정답으로 학습한다고 방문 순서까지 학습하는 것은 아니다.
현재 LightGCN은 방문 edge를, ranker는 평균·검색 점수 등 집계를 사용한다. SASRec처럼 순서를 입력받는 모델과 구분한다.
그래프·콘텐츠 모델에도 순차 encoder를 결합할 수 있고, Two-Tower도 사용자 tower에 순차 encoder를 넣을 수 있다.
순차 방식은 [SASRec 논문](https://arxiv.org/abs/1808.09781), Two-Tower는 [공식 retrieval 설명](https://www.tensorflow.org/recommenders/examples/basic_retrieval)을 참고한다.

### 리뷰 텍스트 후보 검색과 임베딩은 나눠 비교한다

| 조건 | 변경 | 확인할 효과 |
|---|---|---|
| C4 LightGCN / C5 | LightGCN 단독과 item-item + LightGCN RRF 비교 | C5 결합의 효과; C5가 현재 후보 baseline |
| BM25 단독 / C5 + BM25 | 사용자 선호 리뷰와 cutoff 이전 식당 리뷰의 단어 검색 | lexical 신호의 독립 Recall과 C5 보완 효과 |
| 임베딩 단독 / C5 + 임베딩 | 사용자·식당 리뷰 프로필의 의미 벡터 검색 | 의미 검색의 독립 Recall과 C5 보완 효과 |
| BM25 + 임베딩 / C5 + BM25 + 임베딩 | lexical·dense 검색 결과를 RRF 등으로 결합 | 두 텍스트 검색 신호의 추가 효과 |
| 고정 후보 + 텍스트 ranker feature | 동일 후보에서 사용자–식당 텍스트 유사도만 추가 | 재랭킹에서 텍스트의 효과 |
| Two-Tower: 텍스트 없음 / 있음 | 같은 구조·목표에서 리뷰 임베딩 입력만 변경 | 추천 학습과 텍스트를 함께 쓸 때의 효과 |

BM25는 임베딩 없이 단어 일치로 후보를 찾는다. 임베딩 검색은 표현이 달라도 의미가 가까운 리뷰를 찾는다.
OpenRouter는 임베딩을 만드는 API 선택지이고, pgvector·Milvus·Astra DB는 벡터를 저장·검색하는 기반이다.
기존 Two-stage에도 텍스트를 넣을 수 있으므로, 텍스트 검색·텍스트 ranker feature·Two-Tower를 별도 변경으로 비교한다.
후보를 바꾸는 조건은 새 후보의 학습 행·feature를 만들고 ranker도 재학습한다.

### LambdaRank와 Jev 재랭킹 비교

| 방식 | 학습 여부 | 입력·출력 | 이 프로젝트에서의 비교 |
|---|---|---|---|
| R0 | 학습 없음 | 기존 후보 순서 유지 | 재랭킹 전 기준 |
| R1 LambdaRank | 프로젝트 query·label로 학습 | 후보별 수치 feature → 순위 점수 | 현재 학습형 LTR 기준 |
| Jev (TypeSafe) | 이 프로젝트의 label로 재학습하지 않음 | 사용자 이력과 후보 설명 → Choice 확률 | 같은 후보를 재정렬하는 zero-shot decision model |

Jev의 Choice는 고정된 선택지별 확률을 반환하며 한 질문에 최대 255개 옵션을 받는다. 현재 K=100에서는 후보별 확률을 순위 점수로 쓸 수 있다. Jev는 후보 검색 인덱스나 임베딩 모델이 아니며, shortlist에 없는 식당을 추가하지 않는다. [공식 Choice 문서](https://docs.typesafe.ai/primitives/choice)

Jev는 LambdaRank와 같은 의미의 학습형 LTR이 아니다. 같은 고정 후보에서 R0·R1·Jev를 비교하고, 후보 생성 단계의 Recall은 별도로 측정한다. 먼저 C5 후보에서 비교한 뒤 BM25·임베딩 결합 후보에서도 확인한다. Jev 요청 수·입력 토큰·API 비용·지연시간·사용 model ID를 기록한다.

TypeSafe의 예제도 BM25로 shortlist를 만든 뒤 후보를 재랭킹한다. 최근 추천 연구는 SASRec 후보 안에서 Jev를 평가했으나, 정답이 검색된 사용자만 선택하고 평가 목록에 정답을 포함한 통제 실험이다. 따라서 재랭킹 비교를 뒷받침하지만 Jev의 end-to-end 후보 Recall을 증명하지는 않는다. [BM25→재랭킹 예제](https://docs.typesafe.ai/cookbooks/rerank_typesafe), [추천 재랭킹 연구](https://arxiv.org/abs/2609.40241)

### 생성형 후보 검색에서 OneRec 통합까지

**Generative Retrieval**은 사용자 이력으로 식당 식별 코드를 생성해 후보를 찾는 방식이다.
[TIGER](https://arxiv.org/abs/2305.05065)를 참고해, 식당 의미 표현을 이산 코드인 **Semantic ID**로 바꾸고 이력에서 다음 ID를 예측한다.

1. Semantic ID 생성과 유효 식당 ID로의 복원부터 구현한다. 코드북·식당 표현은 해당 학습 cutoff 이전 정보로 구성한다.
2. 생성 후보를 공통 ranker로 재정렬해, 기존 벡터 retrieval과 후보 품질·최종 순위를 비교한다.
3. 후보 생성과 별도 ranker를 분리한 구성을 기준으로, 추천 목록을 직접 생성하는 모델을 비교한다.
4. [OneRec](https://arxiv.org/abs/2502.18965)의 검색·순위 통합, session-wise 목록 생성, 선호 정렬(IPA/DPO)을 단계별로 검토·구현한다.

OneRec 실험은 원논문 재현 범위, 모델 크기·학습 예산, 식당 데이터에 맞춘 변경을 명시한다.
현재 데이터에는 세션·노출·클릭·watch-time이 없으므로 식당 방문 기간을 세션 대용으로 쓰는 변경과 관측 평점 기반 reward를 별도 조건으로 기록한다.
Reward·선호 정렬은 관측 데이터로 확인할 수 있는 범위에서 평가하며, 목록 생성만 구현한 모델을 OneRec 전체 재현으로 부르지 않는다.

### 전체 실험의 공통 평가

- 같은 snapshot·cutoff·평가 사용자·eligible catalog 조건을 사용하고 모델 설정은 validation에서 선택한다. Test의 정답은 cutoff 뒤 방문해 실제 평점을 남긴 사용자–식당 쌍이다.
- 후보 생성은 C4 LightGCN, 현재 C5, BM25, dense embedding, 검증된 조합을 고정 K에서 비교한다. Recall@K로 후보 효과를 먼저 보고 최종 NDCG 등으로 재랭킹 효과를 본다.
- R0·R1·Jev는 같은 사용자·같은 후보 목록에 적용한다. retrieval Recall은 후보 단계 지표로 따로 둔다. 정답을 후보에 강제로 넣지 않는다.
- 현재 평가는 사용자당 T2 고정 cutoff 하나와 이후 기간의 여러 미래 방문을 쓴다. 이 snapshot에서는 T2=2026-05-04, test 방문 범위=2026-05-05~2026-09-26(145일)이다. 이 결과를 “다음 방문 예측”으로 해석하지 않는다.
- 다음 방문 성능을 주장하려면 query마다 방문 직전 이력과 바로 다음 방문 하나로 만든 별도 event-level 평가를 추가한다. 장기 후보 추천을 목표로 하면 현재의 다중 방문 기간 지표를 유지하고 평가 horizon을 명시한다.
- 후보가 있는 모델은 Recall@20/50/100, 모든 모델은 최종 NDCG·Recall·Precision·MAP·MRR@5/10을 비교한다.
- 현재 순위 지표는 positive가 하나 이상인 사용자별 평균이다. 기존 설정에서 positive는 평점 3점 이상을 뜻한다. 새 만족도 주 지표에서 positive를 4점 이상으로 정하면 사용자 수와 지표 분모를 다시 계산한다.
- 현재 test 사용자 1,580명 중 기존 3점 이상 positive 사용자는 1,573명이며, 이력 없는 신규 사용자 1,214명은 개인화 순위 지표에서 제외된다. 신규 사용자 비율과 인기 fallback을 별도로 보고한다.
- cutoff 뒤 실제 방문·평점이 있는 pair는 미방문 정답이 아니라 held-out test 정답이다. 별점 오차에는 test 평점 전체를 쓰며, 만족도 순위에는 평점별 relevance를 적용한다.
- 관측 방문 기반 offline 지표를 실제 추천 노출 후 만족도나 인과 효과로 해석하지 않는다. 노출·클릭·저장·방문 로그가 생기면 별도 사용자 평가나 온라인 평가를 추가한다.
- 목록 직접 생성 모델에 존재하지 않는 후보 단계를 억지로 가정하지 않는다. 생성 모델은 유효 ID 비율·중복·방문 식당 제외 여부도 확인한다.
- Coverage·노출 쏠림·사용자 이력 길이별 성능, 학습·추론 시간, API 토큰 비용·저장 용량을 함께 기록한다.
- 텍스트 입력 유무·순서 입력 유무·검색/재정렬 통합·선호 정렬의 추가 효과는 각각 통제 비교한다.
- 학습은 날짜 경계로 입력 이력을 고정하고 이후 기간의 모든 방문을 묶는 Window 방향으로 진행한다. Prefix를 별도 비교한다면 같은 날 방문 순서를 `review_id`로 가정하는 부분을 수정하고 해당 prepared 자료를 다시 만든다.
- 동일 test를 반복 확인한 결과는 탐색으로 기록하고, 최종 비교에는 새 미래 holdout·seed 반복을 사용한다.

### 2026-10-04 평가 점검

저장된 query·후보·정답과 snapshot에서 baseline 및 shrinkage의 ranking 지표, 후보 일치, 이력 식당 제외, label window, 동점 순서를 독립 재계산했다. 계산은 일치한다. 기존 3점 이상 relevance 기준의 C5 test Recall@100은 20.16%이며, 그 positive 5,622개 중 5,514개(98.08%)는 cutoff 당시 catalog에 이미 있어 미등장 식당만으로 낮아진 결과는 아니다.

| 점검 항목 | 확인 결과와 해석 |
|---|---|
| 같은 날 순서 위험 | T1/T2 validation/test split은 날짜 경계로 과거·미래를 나눈다. 다만 prefix 학습 query의 23.6% (train), 24.2% (T2까지 refit)가 같은 날짜의 이전 interaction을 `review_id` 순서로 이력에 포함한다. timestamp가 없어서 이 순서는 실제 시간 순서로 검증되지 않았다. |
| 평가 대상 | 개인별 과거 이력에는 없고 cutoff 뒤 145일 안에 실제 방문·평점이 기록된 사용자–식당 쌍을 정답으로 하는 다중 정답 평가다. 이는 다음 방문 하나를 맞히는 과제가 아니다. |
| 사용자 범위 | accuracy는 양의 미래 방문이 있는 이력 사용자 1,573명의 macro 평균이다. 신규 사용자 1,214명은 제외되므로 전체 서비스 성능을 대표하지 않는다. |
| 불확실성 | Test는 한 snapshot·한 기간이다. paired bootstrap은 해당 사용자 표본 안 차이의 불확실성을 보지만 새 기간·seed에서의 일반화를 보장하지 않는다. |

**판정:** 저장된 과거 test pair의 ranking 계산은 재현 확인했다. 개인별 만족도 정답을 구현했으며 Window baseline을 학습·평가한다. 기존 relevance 기준의 점수와 새 기준의 점수 차이는 성능 개선의 근거로 쓰지 않는다. 같은 날짜 Prefix 이력 수정은 Prefix 비교를 수행할 때 별도로 다룬다.

## 다음 실행

| 순서 | 작업 | 현재 상태 | 판단 기준 |
|---|---|---|---|
| 0 | Baseline 후보·학습 행 재사용 | 구현·실제 snapshot 검증 완료 | 같은 조건에서 자동 재사용 |
| 1 | 선택한 개인별 만족도 정답 구현 | 구현·검증 완료, 최소 이력 10건 | 학습·validation·test의 정답 정의 일치 |
| 2 | 새 기준으로 Window baseline 측정 | 전체 실행 완료: NDCG@10 0.025467 | 선택한 과제의 기준 성능 기록 |
| 3 | BM25 식당 리뷰 검색 / C5+BM25 | 설계, 미구현 | BM25 단독 Recall 및 C5 보완 Recall |
| 4 | Jev vs LambdaRank 같은 후보 재랭킹 | 설계, 미구현 | 공통 후보 NDCG·Recall과 API 비용·지연 |
| 5 | 리뷰 임베딩·dense/hybrid 검색·텍스트 Two-Tower | 설계, 모델·차원·저장소 미선택 | lexical과 분리한 검색·순위 효과 |
| 6 | Generative Retrieval + 공통 ranker | 미구현 | 벡터 검색과 생성 후보 비교 |
| 7 | 목록 생성·OneRec 통합·선호 정렬 | 미구현 | 구조 통합과 선호 정렬 효과 분리 |
| 8 | 새 미래 holdout·seed 반복 | 미실행 | 반복 확인한 test의 탐색 결과 검증 |

개인별 평점 성향은 [이력 수에 따른 기준](#이력-수에-따른-개인별-평점-기준)을 따른다.
이력이 적으면 기존 기준을 유지하고, 충분하면 과거 사용자 평균을 활용하는 정답 구성을
모든 모델의 공통 기준으로 사용한다. 관측 선호 쌍은 보조 진단으로 두며, 평가 기준의
점수 차이를 모델 개선으로 해석하지 않는다.

실험 실행 개선도 남아 있다. 비교 조건·학습 예산을 지정하고 validation에서 종료하는 통합 설정은 **미구현**이다.
현재 가능한 설정과 비용은 [후보 모델 비교](#후보-모델-비교)에 정리했다.

후속 작업:

- 추천 API, 후보·모델 버전 추적, 노출·클릭·재방문 로그
- DB integration test, 날짜 정밀도·수집 완전성 확인, cold-start(이력이 부족한 사용자·식당) 평가
- 후보 K=200/300 실험 전 평가 cutoff 수정: 현재 20/50/100·5/10 중 K 이하만 생성해 다른 K에서 실패할 수 있음

## LTR 실험: P/W/WR

이 절은 기존 공통 relevance로 작성한 비교 설계를 보존한다. 현재의 기준 실행은
[선택한 만족도 기준을 구현한 Window baseline](./SATISFACTION_BASELINE.md#재현)이며, 이 절의 기존 명령은
개인별 만족도 기준을 적용하지 않는다. 이후 학습 구성을 비교한다면 모든 조건을 동일한
새 평가 기준으로 측정하며, 과거 점수와 직접 비교하지 않는다.

### 질문과 변경 변수

**기간 안의 여러 방문을 함께 학습하고 평점 차이를 보존하면, 최종 추천 순위가 좋아지는가?**
LTR(Learning to Rank)은 후보 100개의 순서를 학습하는 단계다. 이번 실험은 후보 모델을 유지하고 **정답을 묶는 방식과 학습 label**을 바꾼다.

학습 **group**은 한 사용자·한 시점의 후보 목록과 정답이다.
**Label**은 정답을 표시한 코드, **gain**은 LambdaRank가 그 정답에 부여하는 가치다.

| 조건 | 학습 정답 구성 | Label / gain | 확인할 효과 |
|---|---|---|---|
| P: Prefix, 과거 baseline | 방문 직전 이력 → 다음 방문 하나 | 평점 세 등급 0/1/2 → gain 0/1/3 | 과거 기준 |
| W: Window | 기간 시작 이력 → 기간 안 여러 방문 | P와 동일 | 단일 방문 대비 여러 방문 학습 |
| WR: Window + Rating | W와 같은 기간·이력 | 평점 × 2 → gain 원래 평점 | 평점 해상도와 저평점 방문 취급 변화 |

평가에서는 세 조건 모두 이미 미래 기간의 여러 방문을 정답으로 사용한다.
학습 방식이 바뀌어도 출력은 순위 점수이며, 예측 별점이나 개인별 정규화 평점이 아니다.

### 같은 사용자로 보는 차이

가상 예시: 민수는 10월 전 A·B를 방문했고, 10~12월에 C(3.5점)·D(4.5점)·E(5점)를 방문했다.
F는 미관측 식당이다. C·D·E·F가 해당 시점에 존재하고 실제 C5 후보에 검색됐다고 가정한다.

| 조건 | 입력 이력 | 정답과 학습 차이 |
|---|---|---|
| P | C 직전 A·B → D 직전 A·B·C → E 직전 A·B·C·D | 각각 C만 label 1, D만 2, E만 2. 해당 group의 다른 미방문 후보는 0 |
| W | 10월 1일의 A·B로 고정 | 한 group에서 C=1, D=2, E=2, F=0. D·E를 C보다 위에 두되 D와 E는 구분하지 않음 |
| WR | W와 동일 | Label C=7, D=9, E=10, F=0 / gain 3.5, 4.5, 5, 0. E > D > C를 학습 가능 |

P에서는 D와 E가 같은 group의 관측 정답으로 비교되지 않는다. WR의 label 9는 9점 평가가 아니라 4.5점을 정수로 표시한 값이다.
현재 WR은 반점 단위의 1~5점만 허용하며 4.1점을 임의로 반올림하지 않는다.

**WR의 남는 가정:** 관측 1점의 gain도 미관측 0보다 높다. 싫어한 식당을 미관측 식당보다 위에 두도록 학습할 수 있다.
W는 3점 미만을 0으로 처리하므로 W → WR은 평점 해상도만 바꾸는 비교가 아니다. 미관측 0은 실제 0점 평가를 뜻하지 않는다.

### 통제 조건과 평가

| 항목 | 공통 조건 또는 확인 사항 |
|---|---|
| 입력·검색 | 같은 snapshot, 전역 T1/T2, C5 규칙, feature 정의, seed |
| 학습 예산 | 같은 ranker 탐색 범위·validation 선택 규칙 |
| 평가 후보 | 같은 cutoff·이력으로 같은 C5 후보 사용 → 후보 Recall이 같아야 함 |
| 최종 평가 | 공통 relevance 0/1/2·gain 0/1/3, 후보 밖 정답 포함한 NDCG·Recall·Precision·MAP·MRR |
| 동점 | C5 순위 유지 |
| 학습 진단 | 사용 group 수, positive가 여러 개인 group 수, 비교 가능한 관측 평점 쌍 수 |

- P와 W는 학습 시점·이력이 달라 학습 후보가 같지 않을 수 있다. W/WR도 label에 따른 group 선별 결과는 달라질 수 있다.
- 후보 밖 정답을 삽입하지 않는다. 여러 정답 방식이라도 검색된 positive가 하나면 실제 학습 정답도 하나다.
- Early stopping은 후보 내부 NDCG, 설정 선택은 전체 validation NDCG를 쓴다. 선택 후 T2까지 refit한다.
- 이미 확인한 test 비교는 탐색 결과다. 최종 일반화 확인에는 새로운 미래 holdout과 seed 반복이 필요하다.

### 미래 정보를 막는 기간 구성

W/WR의 기본 학습 기간은 3개월이다. 10~12월 group의 후보·평균·인기도·LightGCN은 **10월 1일 이전** 정보로 만든다.
기간 안의 방문·평점은 label에만 사용하고, 입력 이력이나 평균에 미리 반영하지 않는다.

- Train 마지막 기간은 T1, 최종 학습 마지막 기간은 T2에서 자른다.
- Window refit은 T2까지 기간을 다시 구성한다. T1에서 자른 group과 확장 group을 중복으로 붙이지 않는다.
- Validation/test는 약 4개월이므로 3개월 학습 기간과의 차이를 실험 조건에 기록한다.

### 실행과 결과 상태

이 절의 절대 등급 W/WR 비교는 작은 데이터 검증을 통과했지만 **전체 snapshot 비교는 아직 없다.**
개인별 기준의 Window baseline은 [별도 실행 기록](./SATISFACTION_BASELINE.md#실행-결과)으로 구분한다.
2026-10-02 shrinkage 비교는 P의 평균 feature만 바꾼 실험이므로 W/WR 결과로 해석하지 않는다.
프로젝트 루트에서 아래 옵션으로 학습 정답 구성을 지정한다. 다음 작업은 W/WR이며,
Window prepared 자료는 해당 설정으로 별도 생성한다. Prefix의 같은 날짜 이력 수정은
W/WR 실행에 필요하지 않다. 아래 P는 현재 구현의 재현 명령이며, 보수적인 Prefix 비교를
하려면 같은 날짜 이력과 후보 생성 context를 먼저 수정하고 그 조건을 별도로 기록한다.

```bash
# P: 다음 방문 하나
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode prefix --ranker-label-mode relevance --no-mlflow --label prefix-control

# W: 기간 안 여러 방문, 기존 등급
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode relevance --no-mlflow --label window-control

# WR: 같은 기간, 원래 평점 gain
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode rating --no-mlflow --label window-rating
```

## 리뷰+평점 후보 설계

**질문:** 방문·평점에 리뷰 표현을 더하면 후보 검색이 개선되는가?
먼저 Stage 1 후보에서 효과를 확인하고, 후보를 확정한 뒤 LTR을 재학습한다.

| 비교 후보 | 방법 | 확인할 효과 |
|---|---|---|
| 평점 MF | 사용자·식당 잠재 벡터의 내적으로 원래 평점 학습 | 방문 기반 C5 대비 평점 정보 |
| MF + 사용자·식당 bias | 평가 성향·식당 수준의 편차를 규제로 제어 | 개인별 평가 성향 보정 |
| 텍스트 프로필 단독 | 과거 리뷰의 표현을 사용자·식당 프로필로 구성 | 텍스트 자체의 참고 성능 |
| MF + 텍스트 / bias MF + 텍스트 | 평점과 텍스트 신호 결합 | 텍스트의 추가 효과 |

MF(Matrix Factorization)의 기본 관계는 `μ + b_u + b_i + p_u·q_i`다.
μ는 전체 평균, b는 사용자·식당 편차, p·q는 잠재 벡터다. 현재 LightGCN에는 명시적 평점 회귀 bias가 없다.

텍스트 표현과 결합 원칙:

- OpenRouter 임베딩 API·외부 사전학습 encoder를 비교 범위에 포함한다. 사용 모델·차원·호출 조건은 아직 선택하지 않았다.
- Cutoff 이전 리뷰로 학습하는 글자 n-gram TF-IDF 또는 자체 학습 표현도 비교한다. TF-IDF는 표현의 겹침을 측정하며 문장 의미를 이해하는 모델로 해석하지 않는다.
- 좋아한/싫어한 식당의 프로필을 구분한다. 본인 리뷰와 방문 식당의 다른 사용자 리뷰를 비교하며, 이력이 부족하면 공통 프로필을 쓴다.
- 어휘·IDF·문서·프로필은 cutoff 이전 리뷰로만 만든다. Target의 리뷰·평점·맛/가격/서비스는 label 외 입력에서 제외한다.
- 후보 100개·방문 제외·학습 예산을 맞춘다. 다른 척도의 점수를 그대로 더하지 않고 validation에서 가중치를 선택하거나 RRF로 순위를 결합한다.

임베딩 저장·검색 설계:

- 원본 리뷰·평점은 Supabase에 유지한다. 임베딩 생성과 검색 저장소는 별도 모듈로 구성한다.
- 리뷰별 벡터와 `review_id`·본문 hash·모델 ID·차원·전처리 버전을 저장해 반복 API 호출을 줄인다. 사용자·식당 프로필은 cutoff별로 구성한다.
- 저장소는 [Supabase pgvector](https://supabase.com/docs/guides/ai/hybrid-search), [Milvus](https://milvus.io/docs/multi-vector-search.md), [Astra DB](https://docs.datastax.com/en/astra-db-serverless/databases/hybrid-search.html)를 비교 검토한다. 아직 연결하거나 배포하지 않았다.
- 의미 벡터 검색과 키워드 검색을 분리해 검증한다. Supabase의 기본 전문 검색과 Milvus BM25를 같은 알고리즘으로 취급하지 않는다.
- 한국어 토큰화와 Astra 내장 reranking 등 저장소별 차이는 검색 조건에 명시한다. VDB 자체의 변경과 임베딩 모델·검색 알고리즘 변경을 구분한다.

판단 기준은 후보 Recall@20/50/100·전체 방문 Recall이다. 평점 예측 오차(RMSE/MAE)·관측 평점 쌍 순서 일치도도 확인한다.
사용자 이력 수·평점 구간·식당 리뷰 수별 성능, coverage·노출 쏠림·실행 비용을 함께 기록한다.

해석 시 주의할 점:

- 문체를 취향으로 오인하는 영향은 확인할 가설이다. 관측되지 않은 식당의 실제 만족도는 이 평가로 알 수 없다.
- RMSE 개선이 후보 Recall 개선을 보장하지 않는다.
- 사용자별 상수 bias를 모든 후보에서 빼기만 하면 순위는 바뀌지 않는다. 전체 평균 shrinkage도 후한 평가와 좋은 식당만 방문한 효과를 완전히 구분하지 못한다.
- V1 원본의 μ는 식당 평균의 평균, 수정본은 관측 평점 전체 평균이다. 같은 추정량으로 취급하지 않는다.

## 후보 모델 비교

### 어떤 조건부터 비교하는가

현재 C5는 C1 item-item과 C4 LightGCN의 RRF 결합이다.
새 모델 X는 **후보 검색부터 비교**하고, 개선된 조건에 대해 LTR을 다시 학습한다.

| 목적 | 우선 비교할 조건 |
|---|---|
| 현재 구성의 역할 확인 | C1 / LightGCN / C1+LightGCN |
| LightGCN 교체 | 현재 C1+LightGCN / X / C1+X |
| 새 모델 추가 | 현재 C1+LightGCN / X / C1+LightGCN+X |
| 교체·추가 모두 확인 | 현재 baseline / X / C1+X / C1+LightGCN+X |

모든 비어 있지 않은 조합 7개를 처음부터 실행할 필요는 없다.
같은 cutoff·설정·seed의 X 후보는 결합 조건끼리 공유한다. RRF·지표 계산·bootstrap은 추가 모델 학습이 아니다.

### 현재 compare 명령의 동작과 한계

`rating-recsys-compare <model>`은 **후보만 비교하며 LTR을 학습하지 않는다.**

1. 같은 snapshot·T1/T2로 query를 만들고 저장된 validation baseline 후보를 읽는다. 없으면 T1까지 고정 LightGCN과 C0~C5를 생성한다.
2. 새 모델의 grid를 T1까지 학습하고 **단독 validation Recall@100**으로 epoch·설정을 선택한다.
3. 선택한 설정의 단독 / C1+X / C1+C4+X 중 validation Recall@100이 가장 높은 결합을 고른다.
4. 새 모델을 T2까지 한 번 refit한다. Test baseline 후보도 읽거나 생성한 뒤 같은 기간에서 평가한다.

| 기본 실행 | 새 모델 설정 수 | 학습 + 최종 refit | Baseline LightGCN 학습 | LTR |
|---|---:|---:|---:|---:|
| `compare lightgcn` | Layers 3종 × L2 2종 = 6 | 6 + 1 | 첫 생성 2, 재사용 0 | 0 |
| `compare deepconn` | Objective/activation 3종 | 3 + 1 | 첫 생성 2, 재사용 0 | 0 |
| Grid 1개로 제한 | 1 | 1 + 1 | 첫 생성 2, 재사용 0 | 0 |

- 설정을 단독 성능으로 먼저 고른다. 특정 결합 기준으로 설정까지 최적화하는 기능은 없다.
- LightGCN 비교는 최대 200 epochs, 5 epochs마다 평가, 개선 없는 평가 6회면 종료다. C5 내부의 고정 20 epochs와 예산이 다르다.
- 결과 9행은 baseline·참고 6조건과 새 모델 관련 3조건이다. 학습 호출 수나 epoch별 평가 횟수와 구분한다.
- Test의 선택되지 않은 결합은 진단용이다. 최종 정책은 validation에서 선택한다.
- `--baseline-run`은 저장된 R1 추천을 읽는다. X 후보로 LTR을 재학습한 성능이 아니며 `prepared/` 재사용과도 별개다.
- 등록 모델은 `lightgcn`, `deepconn`이다. 다른 X는 모델 구현과 공통 adapter 등록이 필요하다.

### 학습 예산을 줄이는 실행

아래는 LightGCN 후보 grid를 1개로 제한한다. C5 내부의 고정 LightGCN과 비교하며 자동 결합·refit/test는 계속 실행한다.

```bash
rating-recsys-compare lightgcn \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --layers-grid 3 --regularization-grid 0.0001 \
  --max-epochs 20 --eval-every 5 --patience 3
```

후보 CLI는 grid·epochs·평가 간격·patience·seed를 지원하지만 validation 전용 옵션은 없다.
전체 실험의 `--num-leaves-grid 15 --min-child-samples-grid 10`은 고정 150-tree 기준도 남겨 **총 2개 설정**을 비교한다.
Shrinkage CLI는 snapshot·강도·출력·cache 옵션만 받고 튜닝 grid 변경은 지원하지 않는다.

### 추가할 실행 설정

| 선택 항목 | 현재 상태 | 목표 |
|---|---|---|
| 비교 조건 | 자동 결합 목록 고정 | Baseline / X / C1+X 등 직접 선택 |
| 평가 단계 | 후보·LTR runner 분리 | 임의 X를 LTR까지 연결 |
| 학습 예산 | 모델별 CLI 지원 | 통합 설정 파일에서 고정 설정 또는 grid 지정 |
| 종료 단계 | Refit/test 자동 실행 | Validation에서 종료 후 선택 조건만 test |
| 재사용 | `prepared/` 구현 | 동일 조건의 후보·학습 행 유지 |
| 실행 전 확인 | 공통 dry-run 없음 | 조건 수·학습 호출 수·epoch 상한 표시 |

후보 비교에서는 snapshot·cutoff·평가 사용자·정답·후보 수·방문 제외 규칙을 맞춘다.
정답 겹침·coverage·시간도 기록하고, 동일 예산 비교인지 모델별 튜닝인지 명시한다.
후보가 바뀌면 LTR 학습 행·score feature도 다시 구성한다. LightGCN 전용 feature를 X에 그대로 대응한다고 가정하지 않는다.

## Baseline 데이터 재사용

### 무엇을 저장하는가

원본 snapshot은 `artifacts/snapshots/`, 계산한 학습 행·후보는 `artifacts/prepared/<snapshot-id>/`에 둔다.
표준 실험·후보 비교·shrinkage는 기본으로 자동 재사용하고, 없거나 손상된 구간만 생성한다.

| 저장 구간 | 내용 |
|---|---|
| `train-<key>/` | T1까지 feature·label·group 크기·과거 평점 prior |
| `refit-<key>/` | Prefix: T1 이후 추가 학습 행 / window: T2까지 전체 학습 행 |
| `validation-<key>/` | T1 시점 C0~C5 순위·feature·공통 평가 label |
| `test-<key>/` | T2 시점 C0~C5 순위·feature·공통 평가 label |

각 구간의 `data.npz`는 자료형·순서를 보존하는 압축 숫자 배열이다.
`manifest.json`은 후보 목록·행 수·graph 과거 범위·생성 조건·checksum을 기록한다.
과거 feature와 미래 label은 분리하고, test label은 학습·선택에 사용하지 않는다.

실행 manifest에서 `prepared_data`의 key·경로·`hit`/`built`로 재사용 여부를 확인한다.
실제 실행 시간은 `timings_seconds`다. Graph의 `fit_seconds`는 저장 파일을 처음 생성한 시간이다.
구간별 lock·완료 manifest로 불완전한 파일의 동시 읽기를 막는다.

### 무엇을 바꾸면 다시 생성하는가

| 변경 | 처리 |
|---|---|
| Ranker 트리 수·leaves·learning rate·bootstrap 횟수 | 학습 행·평가 후보 모두 재사용 |
| 새 X의 구조·파라미터·리뷰 입력 | Baseline 재사용, X만 별도 계산 |
| Prefix/window 또는 relevance/rating | Train/refit 새 생성, baseline 평가 후보 재사용 |
| Baseline LightGCN·seed·RRF·K·feature·평균 보정 | 영향받는 baseline 구간 새 생성 |
| Snapshot·cutoff·평가 정답·전처리 코드/NumPy/SciPy 버전 | 조건에 맞게 새 생성 |

Key는 보수적으로 구성한다. Snapshot 전체가 바뀌면 과거 내용이 같아도 새 key를 쓴다.
문서·ranker 튜닝 코드만 바뀌면 전처리 key는 유지한다. Graph 가중치 자체는 영구 저장하지 않는다.

```bash
# LTR 학습·성능 평가 없이 데이터만 준비
rating-recsys-prepare --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl

# 후보 비교용: train/refit 학습 행 생략
rating-recsys-prepare --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl --scope candidates
```

실험 명령에 `--no-cache`로 사용·저장을 생략하거나 `--rebuild-cache`로 다시 생성할 수 있다. 두 옵션은 함께 쓰지 않는다.

### 실제 재사용 검증 결과

2026-10-04 snapshot `e7896add5b4b5939`에서 측정했다.

| 항목 | 결과 |
|---|---:|
| 최초 데이터 준비 | 1,277.010초, 약 21.3분 |
| 같은 조건의 재실행 | 7.155초, 4구간 모두 `hit` |
| Train / 추가 refit groups | 14,018 / 1,911 |
| Train / 추가 refit rows | 1,383,764 / 190,888 |
| Validation / test query | 1,645 / 1,580 |
| 저장 크기 | 약 77MB |

**해석:** 이 시간은 준비·읽기만 측정하며 LambdaRank를 포함한 전체 실험 시간이 아니다.
Train/refit summary·모든 C5 평가 후보가 2026-10-02 baseline과 일치했고, 평가 label·feature 행 순서도 확인했다.
기록은 해당 prepared 폴더의 `preparation_initial.json`, `preparation_reuse.json`, `verification.json`이다. 새 성능·MLflow run은 생성하지 않았다.

<details>
<summary>재사용 도입 전 전체 실행 비용</summary>

2026-09-30 manifest의 총 시간은 약 23.6분이다.

| 단계 | 시간 |
|---|---:|
| Train 과거 후보·feature·학습 행 생성 | 16.4분 |
| Validation LightGCN·후보·feature | 1.0분 |
| LambdaRank 7개 설정 탐색 | 41초 |
| 추가 학습 행·최종 refit | 4.1분 |
| Test 후보 생성·평가 | 1.3분 |

과거 후보 생성이 가장 큰 비용이었다. Graph 기록 41개는 3개월 시간 구간이며 첫 구간은 데이터 부족으로 skip했다.
하이퍼파라미터 41개를 탐색한 것이 아니다. Ranker 학습은 7개 설정 + 최종 refit 1회다.
2026-10-02 shrinkage는 실행 내부 공유만 적용해 약 20.6분이었다. 현재 두 runner 모두 저장 데이터를 읽어 과거 graph·학습 행 재생성을 생략한다.

</details>

## 지난 실험과 판단

| 시점 | 실험 | 관측 결과 | 결정 |
|---|---|---|---|
| 2026-09-28 | LightGCN + item-item | 같은 global split의 Recall@100: 20.16%, 당시 C3 16.66% 대비 +3.50%p | C5 채택 |
| 2026-09-30 | DeepCoNN: 리뷰 CNN·FM 후보 | 단독 8.69%, C5 결합 18.44% < C5 20.16% | 채택 안 함 |
| 2026-09-30 | C5 + LambdaRank | ID 동점 정렬 당시 R0/R1 NDCG@10 약 0.0276 | 후보 밖 정답 삽입 run 대체 |
| 2026-10-01 | Ranker 동점 규칙 | 점수가 같은 후보의 순서 변경 | C5 순위 유지 |
| 2026-10-02 | 평균 feature shrinkage λ=10 | Test NDCG@10: 0.029360 → 0.028207 | 기본 λ=0 유지 |

[최신 shrinkage 보고서](./artifacts/comparisons/shrinkage/20261002T062004863532Z-e7896add/report.md)의 validation NDCG@10은 0.025151 → 0.027881였다.
Test 차이 −0.001153의 사용자별 paired bootstrap 2,000회 95% 신뢰구간은 [−0.005006, +0.002692]다.

- **실험 범위:** λ·seed 각 1개, λ는 평가 전 고정. 평균 feature만 변경하고 후보·group·label은 공유했다. 같은 validation 규칙으로 baseline 1 tree, shrinkage 96 trees를 선택했다.
- **해석:** Test 개선이 이어지지 않아 채택하지 않았다. 모든 shrinkage 방식의 실패, 평점 label 정규화, MF bias 학습 결과로 해석하지 않는다.
- **비교 한계:** 2026-09-27 이전 leave-last-two-out(사용자별 마지막 두 방문을 분리) 결과는 현재 global split과 데이터·모수가 달라 직접 비교하지 않는다. DeepCoNN 결과도 텍스트 전체가 무효라는 뜻은 아니다.

## 실행 방법

### 현재 컴퓨터에서 바로 실행

2026-10-04 기준 `~/.local/bin/`의 dashboard·mlflow·prepare·experiment·compare는 프로젝트 `.venv/bin/`에 연결돼 있다.
환경 활성화 없이 `(base)`에서도 실행할 수 있다. 프로젝트 가상환경은 유지해야 한다.
기존 터미널은 `rehash` 후 실행한다.

Dashboard와 MLflow는 **각각 별도 터미널**에서 실행한다. Dashboard는 프로젝트 artifacts를 기본으로 읽고, MLflow는 아래 DB 경로를 사용한다.

```bash
rating-recsys-dashboard --address 127.0.0.1
mlflow ui --backend-store-uri sqlite:////home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

학습·비교 명령은 프로젝트 루트에서 실행한다. P/W/WR·제한 grid·준비 명령은 각 실험 절에 있다.

```bash
cd /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --satisfaction-mode history-aware \
  --label window-personal-satisfaction-n10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python -m rating_recsys.experiments.shrinkage --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl --strength 10
rating-recsys-compare lightgcn --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
rating-recsys-compare deepconn --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

위 shrinkage의 `python -m` 실행은 아래 개발 환경을 활성화한 뒤 사용한다.
Dashboard/MLflow는 `runs/`의 표준 실행용이며 shrinkage는 별도 report·metrics로 확인한다.
전체 옵션은 `rating-recsys-experiment --help`, `rating-recsys-compare <model> --help`에 있다.

### 개발·학습용 환경

프로젝트의 `python`·`pip`를 쓰려면 기존 Conda prefix를 활성화한다. 필요한 패키지는 이미 설치돼 있다.

```bash
conda activate /home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv
rehash
command -v python rating-recsys-dashboard mlflow
```

세 경로가 모두 `rating_recsys/.venv/bin/`인지 확인한다.
저장소 루트의 Python venv `ai-ds-projects/.venv`에는 dashboard·mlflow가 없다. 그 환경이 활성화돼 있으면 먼저 `deactivate`한다.
프로젝트 환경에는 `source .venv/bin/activate` 대신 `conda activate`를 쓴다.

<details>
<summary>새 컴퓨터의 환경 설치·명령 등록</summary>

프로젝트 루트에서 새 환경을 만들 때만 실행한다.

```bash
conda create --prefix ./.venv python=3.10 pip libgomp -y
conda activate ./.venv
pip install -e '.[experiment,dev]'
cp .env.example .env
```

DB 읽기에는 `.env`의 `DATABASE_URL`, 적재에는 고정 `USER_HASH_SALT`도 필요하다. Snapshot 실험에는 DB 연결이 필요 없다.
DeepCoNN은 추가로 설치한다. 리뷰 본문은 snapshot 옆 `.reviews.jsonl`을 읽으며 없으면 DB에서 만든다.

```bash
pip install -e '.[deepconn]' --extra-index-url https://download.pytorch.org/whl/cpu
```

환경 활성화 없이 명령을 쓰려면 아래 경로를 해당 컴퓨터에 맞춰 한 번 등록한다. 기존 연결은 재등록하지 않는다.

```bash
mkdir -p "$HOME/.local/bin"
for task_command in rating-recsys-dashboard mlflow rating-recsys-prepare rating-recsys-experiment rating-recsys-compare; do
  ln -s "/home/hananthony1/codes/ai-ds-projects/projects/rating_recsys/.venv/bin/$task_command" "$HOME/.local/bin/$task_command"
done
rehash
command -v rating-recsys-dashboard mlflow rating-recsys-prepare rating-recsys-experiment rating-recsys-compare
```

`~/.local/bin/`이 PATH에 있어야 한다. 다른 환경에 같은 명령이 있으면 PATH 순서에 따라 먼저 선택될 수 있다.

</details>

## 데이터·크롤링

모델은 Supabase DB 또는 고정 snapshot을 읽는다. 수집 CSV는 적재 단계에서만 사용한다. DB 검증 SQL은 `queries/`에 있다.

| 2026-09-27 수집 기록 | 값 |
|---|---:|
| 원본 리뷰 | 96,922개: 기존 23,207 / 전국 73,715 |
| 최초 interaction | 88,554개 |
| 중간 전국 CSV | 82,966행: 필수값 누락 77 / 확실한 중복 9,174 |

전국 수집은 미완료다. 날짜 표시 변경에 따른 중복·본문 누락·날짜 정밀도 차이가 남아 있다.
고정 salt로 사용자명을 가명화하고 content hash로 중복을 제외하지만, 같은 표시 이름을 같은 사용자로 매핑하는 한계가 있다.

<details>
<summary>적재·크롤링 명령과 운영 주의점</summary>

```bash
rating-recsys-migrate
rating-recsys-ingest --dry-run
rating-recsys-ingest --skip-migrations
python -m rating_recsys.ingestion.import_crawler
rating-recsys-dataset
```

전국 import는 `crawler/data/`의 최신 전국 `.csv.partial`을 고정 복사한다. 완료 `.csv`는 자동 선택하지 않는다.
CSV·`.import.json`은 `artifacts/snapshots/ingestion_sources/`에 보존한다. JSON의 `source_snapshot`은 적재 당시 경로이며 현재 파일은 같은 이름·hash로 이 폴더에서 찾는다.

```bash
python -m pip install -r crawler/requirements.txt
python -m playwright install chromium
python crawler/diningcode_playwright.py --national-regions --max-restaurants 0
python crawler/diningcode_playwright.py --national-regions --max-restaurants 0 --resume
```

- WSL 라이브러리 설치: `sudo .venv/bin/python -m playwright install-deps chromium`
- 기본은 창 표시. `--headless`로 숨기고 `--profile-url`로 단일 식당을 지정한다.
- 중단하면 `.csv.partial`·`.checkpoint.json`을 남긴다. `--skip-incomplete`는 미완료 식당을 미루고 대기 식당을 처리한다.
- 이 옵션의 `crawl_complete=true`도 모든 리뷰의 수집 완료를 보장하지 않는다. `skip_incomplete_urls`·식당 상태를 함께 확인한다.
- 대표 음식 목록 수집은 사이트 식당 전수 수집이 아니다. 접근 제한 시 중단한다.

</details>

## 파일 관리

문서 역할은 [README](./README.md)의 안내, [BASELINE_MODEL](./BASELINE_MODEL.md)의 현재 모델·수치, 이 PLAN의 실험·운영으로 나눈다.
작성 기준은 [docs_style.md](./docs_style.md), 프로젝트 맥락은 [AGENTS.md](./AGENTS.md)에 있다. 자동 생성 보고서는 각 실행 결과와 함께 둔다.

| 경로 | 내용 |
|---|---|
| `artifacts/snapshots/` | Interaction·리뷰 본문·적재 원본 |
| `artifacts/prepared/` | 조건별 학습 행·baseline 후보 |
| `artifacts/runs/` | 표준 baseline 실행: 2026-09-30 run 보존 |
| `artifacts/comparisons/<model>/<run_id>/` | 비교 실행: 2026-10-02 shrinkage 보존 |
| `artifacts/mlflow.db` | 로컬 지표 저장소 |
| `legacy/v1_rating_prediction/` | V1 notebook·데이터·모델·결과·개발 기록, [안내](./legacy/v1_rating_prediction/README.md) |
| `legacy/v2_experiments.zip` | 과거 V2 결과·분석·로그·문서 140개, 로컬 전용 |

결과 폴더는 `report.md`·`manifest.json`·`metrics.json`·모델·추천 파일을 가진다.
Git에는 결과 report·후보 비교 learning_curve만 올리고 snapshot·모델·리뷰 본문은 로컬에 둔다.

과거 기록은 필요할 때 임시 폴더에 푼다. 보관 MLflow 경로는 당시 위치이므로 원본을 확인하고, 재현 시 당시 commit·평가 프로토콜을 맞춘다.

```bash
python -m zipfile -l legacy/v2_experiments.zip
python -m zipfile -e legacy/v2_experiments.zip /tmp/rating_recsys_history
```

## 테스트 관리

프로젝트 환경에서 실행한다.

```bash
python -m pytest -q tests
```

| 파일 | 검증 범위 |
|---|---|
| `test_data.py` | 설정·적재 변환·전역 분할·snapshot |
| `test_retrieval.py` | 후보 제외·결정성·LightGCN·텍스트 모델 |
| `test_ranking.py` | 지표 수식·window label·shrinkage |
| `test_pipeline.py` | 학습·선택·평가·누수·CLI·저장 데이터 복구 |
| `test_comparison.py` | 공통 후보 비교·학습 곡선·CLI·baseline 재사용 |
| `test_observability.py` | MLflow·Streamlit 결과 조회 |
| `test_satisfaction.py` | 개인별 경계·미래 정보 누수 방지·낮은 평점 보존·cache 무효화 |

공통 fixture는 `support.py`다. 새 모델은 `MODEL_CASES`에 작은 설정을 추가하고 기존 retrieval/comparison 테스트를 사용한다.
V1 CSV 5개·31,085행을 통째로 읽는 검증은 V2에서 제거했다.

**검증 기록:** 2026-10-04 개인별 만족도 기준 포함 114개 통과 (56.91초).
재사용 전후 모델·추천 일치, 전처리 생략, window/rating·shrinkage 재사용, 데이터·seed·전처리 변경 시 무효화, 손상 복구·baseline 공유를 확인했다.
개인 기준은 9/10건 경계·상수 평점·저평점 보존을 검증했고, test 평점을 바꿔도 학습 모델과 추천이 같음을 확인했다.
