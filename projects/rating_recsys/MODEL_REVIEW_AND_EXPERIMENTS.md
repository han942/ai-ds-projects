# 모델 구현 점검과 LTR·리뷰 후보 비교

기준일: 2026-10-02. 기존 구현, 이번 비교를 위한 코드, 아직 실행하지 않은 설계를 구분한다.
기존 기준선 수치는 snapshot `e7896add5b4b5939`의
[2026-09-30 run](./artifacts/runs/20260930T135424227862Z-e7896add/report.md)에서 가져왔다.

## 1. 무엇을 바꾸려는가

정답 식당이 하나라는 문제는 **Stage 2 LTR의 학습 query**에 관한 것이다. 평가 query는
이미 사용자별 미래 window의 여러 방문을 정답으로 사용한다. Stage 1 후보 생성과는
별도 문제이므로, 먼저 C5 후보·feature·평가 정의를 고정하고 LTR 학습 구조를 비교한다.

리뷰 텍스트는 사용자 요청에 따라 **Stage 1 후보 생성의 추가 feature**로 설계한다.
처음부터 LTR에 텍스트 feature를 함께 추가하면 학습 query 변경과 텍스트 추가 효과를
구분할 수 없으므로 실험을 나눈다. 리뷰 후보 실험에서는 LTR을 다시 학습하지 않고,
후보 방식이 정해진 다음 그 후보에 맞춰 LTR을 재학습한다.

## 2. 기본적인 사용자·식당 관계를 구현하지 않았던 것인가

아니다. 다음 구성은 새로운 모델 제안이 아니라 기존 MF의 기본 구조다.

\[
\widehat r_{ui}=\mu+b_u+b_i+p_u^\top q_i
\]

| 구현 | 사용자·식당 관계 | 학습 목표와 현재 사용 여부 |
|---|---|---|
| v1 원본 `diningcode_analysis.ipynb` | `Study/RecSys/matrixfactorization/matfac.py`에서 기준 평균 + 사용자 bias + 식당 bias + ID embedding 내적을 구현 | 원래 평점의 오차로 P, Q, b_u, b_i를 정규화 SGD로 update. 노트북과 repository의 실제 모듈까지 확인 |
| v1 수정본 `diningcode_revision.ipynb`의 `MF` | 전체 평균 + 사용자 bias + 식당 bias + ID embedding 내적을 명시적으로 구현 | 원래 평점에 대한 회귀 |
| v1 수정본 `TextRatingModel(use_id=True)` | 리뷰 CNN/FM 항에 사용자·식당 ID bias와 ID embedding 내적을 추가 | 텍스트와 협업 필터링을 결합한 평점 회귀 |
| 현재 v2 C4 LightGCN | 그래프 전파 후 사용자·식당 embedding 내적 | 방문 식당을 미방문 식당보다 높게 하는 BPR. 별도 사용자·식당 bias와 평점 회귀는 없음 |
| 현재 v2 C1 | 과거 방문 식당과 후보 식당의 co-occurrence 유사도 | 학습 없는 이력 기반 후보 생성 |
| v2 별도 DeepCoNN 실험 | 텍스트 CNN/FM의 점수는 사용자 항 + 식당 항 + 텍스트 기반 사용자·식당 내적으로 분해 가능 | MSE/BPR 모두 비교. 현재 C5에는 채택하지 않음 |
| 현재 v2 R1 | 사용자 평균 평점·이력 길이와 식당 평균 평점·retrieval score 등으로 트리 학습 | LambdaRank. 명시적 MF 수식이나 학습된 사용자 ID embedding을 별도로 넣은 모델은 아님 |

확인한 코드:
[v1 원본](./legacy/v1_rating_prediction/diningcode_analysis.ipynb),
[v1 수정본](./legacy/v1_rating_prediction/diningcode_revision.ipynb),
[원본 MF 모듈](../../Study/RecSys/matrixfactorization/matfac.py),
[LightGCN](./src/rating_recsys/retrieval/lightgcn.py),
[DeepCoNN](./src/rating_recsys/retrieval/deepconn.py),
[LTR feature](./src/rating_recsys/ranking/features.py).

따라서 “개인화의 기본 관계를 새로 도입한다”는 설명은 부정확하다. 이번에 확인하려는
차이는 **방문 여부를 학습한 관계와 원래 평점을 학습한 관계**, 그리고 텍스트가 그
관계에 추가하는 정보다. v1 모델을 그대로 옮겨도 snapshot·시간 분할·목표가 달라
기존 평점 RMSE로 v2 후보 성능을 예상할 수 없다.

원본 MF의 `global_mean`은 식당별 평균의 평균이고 수정본은 관측 평점 전체의 평균이다.
같은 식 형태라도 정확히 동일한 추정값·학습 절차라고 표현하지 않는다.

VBPR도 협업 필터링 관계에 이미지 기반 개인화 항을 추가하는 방향이다. 다만 VBPR은
implicit feedback의 pairwise ranking 모델이므로 원래 평점을 MSE로 예측하는 모델과
동일한 학습 목표는 아니다. 사용자별 상수인 전체 평균·사용자 bias는 같은 사용자 내
두 식당의 점수 차이에서 상쇄된다.
[VBPR 원 논문](https://arxiv.org/abs/1510.01784).

## 3. Shrinkage의 실제 적용 상태

현재 v2의 사용자 평균과 식당 평균은 단순 평균이다. 사용자 평균에 shrinkage를
적용한 상태라고 설명하면 안 된다. 예시 식은 다음과 같다.

\[
\widetilde\mu_u=\frac{n_u\overline r_u+\lambda\mu}{n_u+\lambda}
\]

리뷰가 적을 때 전체 평균을 더 믿고, 많을 때 사용자 평균을 더 믿는다. 이는 평균의
불확실성을 줄이는 가정이지 “후하게 점수를 주는 사용자”와 “좋은 식당만 방문한 사용자”를
완전히 구분하는 방법은 아니다. 보정 강도는 validation에서 선택한다.

v1 수정본에는 별도로 `fit_bias`가 있다. 식당 bias를 계산한 뒤 사용자 bias를 계산하는
update를 15회 반복하며, 잔차 합을 `평가 개수 + 정규화 강도`로 나눈다.

\[
b_u=\frac{\sum_i(r_{ui}-\mu-b_i)}{n_u+\lambda_u},\qquad
b_i=\frac{\sum_u(r_{ui}-\mu-b_u)}{n_i+\lambda_i}
\]

기존 v1 validation 선택은 사용자 3, 식당 25였다. v2에 그대로 사용할 근거는 없다.
일반적인 MF의 L2 정규화와 이 사후 평균 보정은 관련 있지만 같은 코드나 같은
하이퍼파라미터로 취급하지 않는다.

한 사용자의 모든 후보에서 같은 평균을 빼면 순위는 변하지 않는다. 실제 취향 차이는
사용자×식당 항과 사용자×텍스트 특성의 관계가 만들어야 한다. 식당 평균까지 제거한
잔차만으로 추천하면 공통적인 식당 품질을 지울 수 있으므로, 평점 회귀에서는 bias와
취향 항을 합친 최종 평점도 함께 평가한다.

## 4. LTR 비교: 정답 1개 → 여러 정답 → 원래 평점

기존 prefix 학습은 한 번의 미래 방문을 정답으로 둔다. 3점 미만은 0, 3점 이상 4점
미만은 1, 4점 이상은 2이며, 후보의 나머지는 0이다. 한 그룹 안에서 여러 관측 식당의
만족도를 직접 비교하지 못한다. 단일 positive query의 NDCG에서는 positive gain이
ideal DCG로 정규화될 때 상쇄되므로, 1/2 등급을 단순히 강약 가중치로 설명하는 것도
충분하지 않다. 평가의 여러 positive 사이에서는 gain 차이가 의미를 갖는다.

비교용으로 두 설정을 추가했다. 기본값은 과거 기준선을 재현하는 `prefix/relevance`로
유지하고, 새 방식은 명시적인 옵션으로 실행한다. 아직 전체 snapshot의 새 성능을
측정했다고 해석하지 않는다.

| 조건 | 학습 query | 학습 label/gain | 비교 목적 |
|---|---|---|---|
| P | 기존 prefix, 정답 1개 | relevance 0/1/2, gain 0/1/3 | 기존 방식 |
| W | 사용자 × calendar window | relevance 0/1/2, gain 0/1/3 | 여러 정답 그룹을 구성한 효과 |
| WR | W와 동일 | 관측 평점 그대로의 선형 gain | 4·4.5·5점 등 세밀한 선호 차이를 보존한 효과 |

### Window 구성과 누수 경계

1. 기본 3개월 calendar window마다 시작일 직전까지의 사용자 이력·retrieval context를
   고정한다. LightGCN도 그 시작일 **이전** interaction만으로 학습한다.
2. Window 안에서 방문한 식당 여러 개를 같은 사용자의 outcome으로 둔다. 시작일
   이후의 방문·평점은 label에만 쓰며 후보·feature에는 쓰지 않는다.
3. C5가 실제 검색한 100개 후보에만 label을 붙인다. 후보 밖 정답을 끼워 넣지 않는다.
4. 관측 positive가 검색되고 서로 다른 label이 있는 그룹을 학습한다. 정답이 여러 개여도
   후보에 하나만 포함되면 실제 여러 positive의 선호 비교가 생기지 않는다.
5. 튜닝 학습의 마지막 window는 T1에서 자른다. 최종 refit에서는 T2까지 모든 window를
   재구성한다. T1의 부분 window와 확장된 window를 중복으로 붙이지 않는다.
6. 학습 window는 기본 3개월, 기존 validation/test window는 약 4개월이다. 이 길이 차이는
   실험 조건과 한계로 남기며, 처음 비교에서 평가 window 자체를 바꾸지 않는다.

학습 요약에는 전체 그룹 수, 검색된 positive가 여러 개인 그룹 수, 서로 다른 관측
label이 있는 그룹 수, 실제 평점이 다른 관측 식당 쌍 수, 그룹당 검색 positive 평균을
기록한다. Window 구조만 바꿨다고 만족도 순서 학습이 충분해졌다고 결론 내리지 않는다.

### “Relevance 제거”의 정확한 의미

WR은 평점을 0/1/2의 세 등급으로 묶는 것을 제거한다. LambdaRank의 정수 label 요구에
맞춰 평점 × 2를 index로 전달하고 gain은 index / 2, 즉 원래 평점으로 설정한다.
예를 들어 4.5점은 index 9, gain 4.5다. 개인별 평점을 정규화한 예측값은 아니다.
반점 단위 1~5점 이외의 데이터는 조용히 반올림하지 않고 거절한다.
[LightGBM LambdaRank 문서](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html).

WR에서도 미관측 후보는 gain 0인 약한 negative다. **실제 0점 평가라는 뜻은 아니다.**
따라서 관측 1점 식당도 미관측 후보보다 높은 gain을 갖는다는 가정이 남는다. 이 실험은
relevance 개념과 미관측 비교를 전부 제거한 모델이 아니다. 추후에는 실제 관측 평점이
다른 쌍만 학습하는 pairwise objective와 미관측 비교를 분리하고, 비교 가능한 사용자·쌍
수를 함께 보고한다. 평점이 같은 쌍은 선호 순서를 부여하지 않는다.

### 평가와 실행

P/W/WR 모두 평가 사용자의 정의, C5 후보, feature, relevance threshold와 평가 NDCG
gain을 동일하게 유지한다. WR 학습 gain을 바꿨다는 이유로 평가 NDCG 정의까지 바꾸지
않는다. Validation에서 설정·트리 수를 선택하고 고른 설정으로 T2까지 refit한다.
Early stopping도 공통 0/1/3 gain의 NDCG를 사용하도록 했다. 검색 positive가 있는
validation 그룹의 후보 내부 ideal DCG로 정규화하므로, 후보 밖 정답까지 포함한 전체
window 평가와는 구분한다. 동점은 validation에서도 후보 순서로 처리한다.

```bash
# 프로젝트 가상환경을 활성화한 상태에서 실행
rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode relevance --label window-control

rating-recsys-experiment --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window --ranker-label-mode rating --label window-rating
```

기존 2026-09-30 run은 식당 ID로 동점을 정렬했고 현재 코드는 C5 순서를 유지한다.
학습 구조의 효과만 분리하려면 P도 현재 동점 규칙으로 재측정한다. 기존 수치를 그대로
옆에 놓을 경우 이 차이를 표시한다. R0도 validation에서의 비교 대상으로 보고,
재정렬을 기본적으로 이득이라고 가정하지 않는다.

공통 지표는 NDCG·Recall·Precision·MAP·MRR@5/10, coverage·novelty·지역 다양성이다.
후보 Recall@100이 세 조건에서 같아야 한다. 사용자별 paired bootstrap을 사용하고
작은 차이는 seed 3~5회로 확인한다. 같은 test를 반복 확인한 비교는 탐색 결과이며
최종 모델의 일반화 확인에는 새로운 미래 holdout이 필요하다.

## 5. 리뷰 텍스트 + 평점 후보 생성 설계

이 절은 설계다. 새 리뷰 후보 모델의 구현·전체 실행 결과를 뜻하지 않는다.
**Sentence Transformers 패키지와 해당 사전학습 모델을 사용하는 방식은 사용자 요청으로
보류했다(2026-10-02).** 설치·모델 다운로드·추론 실험을 진행하지 않는다. 기존 LightGCN,
MF와 자체 데이터로 학습하는 텍스트 표현까지 보류한 것은 아니다.

우선 외부 사전학습 모델을 사용하지 않는 대안으로, train 리뷰만으로 어휘와 IDF를
학습한 글자 n-gram TF-IDF를 검토할 수 있다. 이 방식은 단어·표현의 겹침을 중심으로
비교하며 문장 의미를 잘 이해하는 모델이라고 해석하지 않는다. 현재 자체 CNN도 후보
대안이지만, 기존 DeepCoNN 결과를 고려해 처음부터 복잡한 모델을 전제로 삼지 않는다.
표현 방식의 선택과 평점·텍스트 결합 설계는 구분하고, 텍스트 후보의 효과는 실험으로
확인한다.

기본 협업 필터링에 텍스트 항을 추가한다.

\[
\widehat r_{ui}=\mu+b_u+b_i+p_u^\top q_i+f_\theta(\text{사용자 취향 프로필},\text{식당 리뷰 특성})
\]

평점은 원래 값으로 회귀 학습한다. 아직 방문하지 않은 식당에 대한 사용자 평점은
입력할 수 없으므로, 과거 평점이 취향 프로필을 구성하고 학습 정답을 제공하는 역할이다.

### 텍스트 feature

- 과거 리뷰를 벡터화한다. TF-IDF를 사용할 경우 어휘·IDF는 각 학습 cutoff 이전 리뷰로만
  학습하고 validation/test 리뷰는 fit에 사용하지 않는다. 프로필에 미래 리뷰를 넣지
  않는 것뿐 아니라 표현을 학습하는 데이터에도 같은 경계를 적용한다.
- 보류한 사전학습 선택지는 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`다.
  추후 재개한다면 고정 384차원 encoder의 revision·파일 hash와 truncation 조건을 기록한다.
  [제공자 모델 카드](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
- 사용자 프로필은 과거 좋아했던 식당의 특성과 싫어했던 식당의 특성을 구분한다.
  본인 리뷰의 임베딩을 사용하는 방식과, 과거 방문 식당에 대한 다른 사용자의 리뷰를
  사용하는 방식을 별도 비교한다. 후자는 본인의 문체를 선호로 오인하는 영향을 줄이려는
  가설이며 효과가 확인된 사실은 아니다.
- 평점 가중치는 원래 평점 기준과 regularized 사용자 bias 보정 기준을 나누어 비교한다.
  평가 개수가 적거나 선호·불호가 구분되지 않으면 공통 프로필·평점 모델로 fallback한다.
- 후보 식당에는 추천 시점 이전 리뷰만 사용한다. 학습 target 리뷰는 사용자·식당
  프로필에서 모두 제외한다. Target의 평점·본문·맛/가격/서비스 점수는 label 외 입력으로
  쓰지 않는다. 과거 profile 생성에서도 당시 미래 리뷰를 읽지 않도록 시점을 고정한다.
- 선호 유사도, 불호 유사도, 리뷰 수, 텍스트 보유 비율과 이력 수를 feature로 검토한다.
  단순한 문장 유사도는 긍정·부정을 충분히 구분하지 못하므로 평점과 함께 학습한다.

### 비교 조건

| 조건 | 평점 학습 | 사용자·식당 bias | 텍스트 | 목적 |
|---|---|---|---|---|
| C5 | 기존 방문 기반 C1+LightGCN | 기존 구성 | 없음 | 추천 후보 기준선 |
| A | MF, 원래 평점 | 없음 | 없음 | 평점 기반 후보의 기여 |
| B | A와 동일 | 정규화 bias | 없음 | 기본 평가 성향 보정의 기여 |
| C | 없음 | 없음 | 동일 가중치의 텍스트 프로필 유사도 | 텍스트 단독 참고 |
| D | A와 동일 | 없음 | 평점과 함께 학습한 텍스트 항 | 텍스트 추가의 기여 |
| E | B와 동일 | 정규화 bias | D와 동일한 텍스트 항 | 보정과 텍스트의 결합 |

A/B/D/E는 같은 latent dimension·최적화 조건·학습 예산으로 비교한다. 먼저 단독
Top-100, 그다음 validation에서 고른 후보 소스와 C5의 결합을 평가한다. 후보 수는
100개로 고정하고 이미 방문한 식당은 제외한다. 평점 점수와 cosine 유사도를 척도
조정 없이 더하지 않는다. Validation에서 결합 가중치를 선택하거나 source별 순위를
RRF로 결합한다. RRF도 새로운 후보가 기존 좋은 후보를 밀어낼 수 있어 검증이 필요하다.

### 지표와 판단

- 공통 후보 Recall@20/50/100: 기존 relevance 평가 기준으로 C5와 직접 비교한다.
- 별도 방문 Recall: 저평점 방문도 포함한 전체 관측 방문을 대상으로 추가 측정하고
  기존 positive Recall과 구분한다.
- 원래 평점 RMSE/MAE: 실제 평가가 존재하는 방문에서만 계산한다. 점수를 낼 수 있는
  방문의 수·비율을 함께 보고, 모델마다 다른 평가 집합이면 공통 집합에서도 비교한다.
- 관측 선호 순서 일치도: 같은 사용자의 서로 다른 실제 평점 쌍을 대상으로 계산한다.
  미관측 식당의 실제 만족도를 측정했다고 해석하지 않는다.
- 사용자 이력 1~2/3~9/10개 이상, 과거 평균 평점 구간, 식당 리뷰 수와 텍스트 유무별
  결과를 보고 cold-start는 기존 평가 대상과 별도 모수로 측정한다.
- Coverage, 식당 노출 쏠림, 학습·추론 시간과 사용자별 paired bootstrap을 기록한다.

RMSE가 좋아져도 후보 Recall이 좋아진다고 보장하지 않는다. 두 지표를 함께 보고
신호의 역할을 판단한다. 원래 평가 기준이 개인 취향을 충분히 표현하는지 검토하는
추가 평가와 모델 학습 변경은 결과표에서 구분한다.

## 6. 기존 리뷰 후보 실험과의 관계

[DeepCoNN 보고서](./artifacts/comparisons/deepconn/20260930T111734850000Z-e7896add/report.md):

| 동일 snapshot의 기존 test | Recall@100 |
|---|---:|
| C5 | 20.16% |
| DeepCoNN 단독 | 8.69% |
| C5 + DeepCoNN | 18.44% |

당시 구현은 글자 임베딩을 처음부터 학습하고 문서를 400자로 제한한 CNN/FM이었다.
보고서의 분해 분석에서는 식당 항과 인기도의 높은 상관이 관찰됐고 개인화 기여가
작았다. 이 구현의 부진을 텍스트 전반의 무효성으로 일반화하지 않는다.

과거 보고서는 텍스트를 ranker feature로 먼저 시험하자고 제안했다. 그 문장은 당시
판단 기록으로 보존한다. 2026-10-02 사용자 결정은 **텍스트를 후보 생성에서 별도
feature로 평점과 결합해 비교**하는 것으로, 이후 실험 방향은 이 문서를 따른다.

## 7. 현재 실행 환경과 보류한 리뷰 encoder의 자원 예상

2026-10-02 프로젝트 실행 환경에서 확인했다. WSL/가상화 환경에 보이는 자원이며
물리 PC 전체 RAM이나 Windows 쪽 GPU 유무를 단정하지 않는다.

| 항목 | 확인 결과 |
|---|---|
| CPU | Intel Core i5-10400, 6 core / 12 thread |
| 실행 환경 RAM | 7.7 GiB, 당시 available 약 5.1 GiB |
| Swap | 2.0 GiB |
| 프로젝트 파일시스템 여유 | 약 932 GiB. 가상 디스크 표시이므로 물리 Windows 디스크 여유와는 다를 수 있음 |
| 설치된 PyTorch | 2.5.1+cpu, 현재 환경에서 CUDA 사용 불가 |

선택한 multilingual MiniLM의 `model.safetensors`는 제공자 파일 목록 기준 약 471 MB다.
토크나이저와 설정 파일을 포함하면 한 가지 PyTorch 형식의 모델 파일은 대략 0.5 GB다.
저장소 전체 4.62 GB는 ONNX/OpenVINO/TensorFlow 등 여러 형식이 함께 있는 크기이며
전부 내려받을 필요는 없다.
[제공자 파일 목록](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main).

88,554개 리뷰를 384차원 float32로 저장하면 벡터 배열 자체는
`88,554 × 384 × 4 = 136,018,944 bytes`, 약 130 MiB다. 사용자 14,008명과 식당
4,587개의 평균 프로필 배열을 각각 저장해도 합계 약 27 MiB다. ID·metadata·중간
배열·모델 실행 메모리는 별도로 필요하다.

아래는 보류한 모델의 자원 검토 기록이며 실행 승인을 뜻하지 않는다.
현재 환경에서는 CPU로 고정 encoder 추론을 작은 batch(8~16), 최대 128 token으로
진행하고 임베딩을 파일에 캐시하는 방식이 가능할 것으로 판단한다. 약 1~3 GiB의
추가 RAM을 계획값으로 잡되, 실제 peak RAM을 측정한 값은 아니다. Transformer를
fine-tuning하는 계획과 구분한다. PyTorch는 이미 설치돼 있어 CPU 실험을 위해
CUDA/PyTorch 전체를 다시 설치할 필요는 없다.

전체 소요 시간은 아직 측정하지 않았다. 추후 사용자가 사용을 재개하면 우선 리뷰
1,000개로 속도와 peak RAM을 측정하고, 결과를 바탕으로 전체 예상 시간과 batch를
결정한다. 임베딩 생성과 다른 모델의 학습을 동시에 하지 않아 메모리 경합을 줄인다.
모델·패키지·캐시의 총 추가 디스크 예산은 여유 있게 1~2 GB부터 잡고 설치 계획을
확인한다. 현재 비교 코드는 임베딩 모델의 다운로드나 리뷰 전송을 수행하지 않는다.

## 8. 검증 상태

LTR 비교 옵션의 누수 경계, 부분 window refit, 여러 관측 평점의 label, 후보 밖 정답을
추가하지 않는 조건과 공통 validation 지표를 synthetic 데이터로 검증했다. 기존 후보
비교·CLI·MLflow·대시보드 테스트를 포함해 프로젝트 테스트 89개가 통과했다.

```bash
python -m pytest -q tests
```

새 LTR 설정과 새 텍스트 후보의 전체 snapshot 성능 비교는 아직 실행하지 않았다.
기존 기본값 `prefix/relevance`를 새 기준선으로 교체하지 않았으며, 실제 측정 후
판단한다. 설치된 패키지나 다운로드한 외부 모델이 추가된 상태도 아니다.
