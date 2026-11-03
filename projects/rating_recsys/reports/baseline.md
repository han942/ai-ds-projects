# Baseline: 후보 검색과 재정렬

**연구 질문**

- 현재 성능을 만드는 행동 신호
- 추천의 지역 집중 정도

문서 정리: 2026-10-13. 기준 실행: 2026-10-04~05; 학습·분할 재검증: 2026-10-13.

**현재 판단: C1과 LightGCN의 결합은 후보 회수를 보완했다. LambdaRank의 test 재정렬 이득과 전국 취향 추천은 확인하지 못했다.**

이 문서는 텍스트 실험의 기준이 되는 행동 신호와 병목을 설명한다. C1은 함께 방문된 식당을, LightGCN은 방문 그래프를 거친 관계를 이용한다. 두 검색기의 결합이 정답을 더 회수했는지와, 같은 후보의 재정렬이 만족도 순서를 개선했는지는 서로 다른 비교다. 세부 모델 정의는 [BASELINE_MODEL](../BASELINE_MODEL.md)에 둔다.

## 비교 조건

1. 실행

   - 2026-10-04~05
   - seed 42
   - Snapshot `e7896add5b4b5939`

2. 학습·정답

   - 3개월 Window
   - History-aware 만족도
   - 개인 기준 적용 최소 이력 10건

3. 후보·평가

   - 후보 100개
   - Test 관련 정답 사용자 1,573명

4. 입력

   - 방문·평점
   - 리뷰 텍스트 없음

입력·정답·16개 feature의 정의는 [BASELINE_MODEL](../BASELINE_MODEL.md)을 따른다. **기존 test 점수는 2026-10-12 LTR 전체 정답 IDCG 수정 전 실행이다.** 10월 13일에는 아래와 같이 제한된 validation 학습을 재생했다. 전체 grid 재선택·T2 refit·새 test 평가는 하지 않았다.

## 학습·시간 분할 재검증 (2026-10-13)

**학습 실패나 확인된 미래 입력 누수는 없었다. LambdaRank는 더 오래 학습할 때 과적합 징후가 있지만, 기존 76트리가 과도했다고 단정할 근거는 부족하다.** 같은 설정을 300트리까지 재생한 validation 곡선에서도 최고 지점은 76트리였다. LightGCN은 20epoch까지 학습 손실이 줄면서 validation 후보 회수율도 높아졌다. 이는 한 seed와 반복 사용한 validation의 탐색 결과다.

이번에는 snapshot과 저장 배열·모델을 독립 검산하고, 역사적 manifest 설정을 고정해 T1 데이터로 LambdaRank 300트리와 LightGCN 20epoch를 각각 한 번 학습했다. 기존 run·모델·prepared 파일을 변경하지 않았고 test 예측과 모델 채택은 없었다. 현재 CLI 기본값인 `prefix/absolute`가 아니라 기준 실행의 `window/history-aware/min_history=10`을 사용했다.

### Validation 동작

Validation은 미래 데이터로 gradient를 학습하는 단계가 아니라, 과거로 학습한 모델의 설정과 학습량을 선택하는 단계다. 추천을 만들 때와 정답을 확인할 때 이용 가능한 정보를 구분한다.

**3개월 분기는 학습 사례를 만드는 단위이며, 분기별 validation fold가 아니다.** 기준 Train에서는 38개 분기의 사용자×기간 query 19,158개를 만들고, 학습 가능한 7,123개 후보 그룹을 하나의 배열로 합쳐 ranker 하나를 학습한다. LambdaRank의 `group`은 같은 사용자가 같은 시점에 받은 후보 목록이므로, group마다 별도의 최상 모델을 고르는 절차가 아니다.

Validation은 아래의 미래 구간 하나다. 각 설정으로 사용자별 NDCG를 계산한 뒤 관련 사용자 1,639명의 **평균**을 비교해 설정 하나를 선택한다. 저장 추천으로 직접 재계산한 평균 0.0276943799는 기존 보고와 일치했다. 분기별·사용자별 최고 점수를 모으거나 가장 좋은 분기의 점수를 전체 성능으로 보고하지 않는다. 현재 코드에는 분기별 rolling validation이 구현돼 있지 않다.

| 단계 | 모델 입력·학습 | 정답과 사용 목적 |
|---|---|---|
| Train | 2025-12-19(T1)까지의 데이터. 각 3개월 학습 Window 시작 전 이력·그래프·feature 사용 | 해당 Window의 방문·평점을 학습 label로 사용 |
| Validation | T1까지의 이력으로 한 번 추천. 이후 방문·리뷰·평점은 입력에서 제외 | 2025-12-20~2026-05-04의 결과로 설정·트리 수 선택 |
| Refit | 선택을 마친 설정으로 2026-05-04(T2)까지 다시 학습 | 기존 validation 기간도 이제 학습 데이터에 포함 |
| Test | T2까지의 이력으로 추천 | 2026-05-05~2026-09-26의 결과. 이번 감사는 분할 일관성만 확인 |

Validation 기간 중 이력이 매 방문마다 갱신되는 방식은 아니다. T1에서 만든 목록을 136일 동안 관측된 정답 전체와 비교한다. Test도 T2의 입력을 고정한 145일 Window다. Train은 3개월 달력 Window라 평가 기간과 길이가 다르다. 따라서 train/validation/test 점수 차이에는 학습량뿐 아니라 기간·사용자·정답 분포 차이도 포함된다.

후보 검색의 Recall@100은 사용자별로 `후보 100개에 있는 3점 이상 미래 식당 수 / 3점 이상 미래 식당 수`를 계산해 평균한다. 재정렬은 같은 후보 안의 Top-10을 만족도 등급과 위치를 반영한 NDCG@10으로 채점한다. 후보 밖 정답을 정답 분모와 이상적 순위(IDCG)에서 빼지 않는다. 신규 사용자는 별도 집계하며 현재 개인화 정확도 평가에는 포함하지 않는다.

예를 들어 T1까지 A·B·C를 방문한 사용자가 이후 D에 5점, E에 4점, F에 2점을 줬다면 모델 입력은 A·B·C뿐이다. 정답은 D·E이고, 후보에 D만 있으면 Recall@100은 50%다. F는 관련 정답이 아니지만, 미관측 식당을 실제 불만족 식당으로 해석하지 않는다. E가 후보 밖에 있어도 NDCG의 이상적 순위에는 D·E가 모두 남는다.

### 분할·학습 입력 검산

| 검사 | 직접 확인한 결과 |
|---|---|
| Snapshot·prepared checksum | 기준 manifest와 일치 |
| 사용자–식당 쌍·review_id 중복 | 각각 0 |
| 저장 validation/test query와 snapshot 이력·정답 | 불일치 0 |
| 평가 이력의 cutoff 이후 사건 | 0 |
| 과거 LightGCN checkpoint의 edge 수 | 독립 재구성과 불일치 0 |
| Train/refit Window 정답 중복 | 각 phase 내부 0; 마지막 부분 Window 중복 추가 없음 |
| 학습 배열 | 712,232행·7,123그룹·16피처; 유한값·그룹 합계 일치 |
| 저장 validation 모델 | 76트리이며 기존 validation NDCG 0.0276943799 재현 |

Train의 19,158 query 중 실제 ranker 학습에 쓰인 것은 7,123그룹이다. 후보에 양성이 없거나 모든 label이 같으면 순위 학습 gradient를 만들 수 없어 제외한다. 이 설계 자체는 오류가 아니지만 학습 집합은 전체 평가 사용자보다 후보 회수가 쉬운 집합이다. 학습 행의 98.58%는 label 0이며 대부분 미관측 후보다. 행 수를 만족도 비교 예시 수로 해석하면 안 된다.

### 날짜 정밀도의 한계

| 날짜 정밀도 | Train 70,883건 | Validation 8,934건 | Test 8,737건 |
|---|---:|---:|---:|
| 연·월·일 명시 (`exact`) | 70,434 | 653 | 0 |
| 월·일에서 연도 추정 (`inferred_year`) | 431 | 8,199 | 8,427 |
| 상대 날짜 해석 (`relative`) | 18 | 81 | 308 |
| 알 수 없어 수집일 대체 (`unknown`) | 0 | 1 | 2 |

월·일만 있으면 ingestion은 수집 연도를 붙이고 수집일보다 미래가 되면 전년으로 바꾼다. 최근 날짜의 사이트 표기와 맞을 수 있으므로 추정 자체를 오류나 누수라고 단정하지 않는다. 다만 사이트가 연도를 생략하는 조건을 원자료로 확인해야 시간 분할을 더 신뢰할 수 있다. `unknown` 3건 제외 후에도 T1/T2는 같았지만, 이것으로 추정 연도의 정확성까지 검증한 것은 아니다. `exact`만 남기면 test가 0건이어서 독립 검증 대안이 되지 않는다.

### 학습량별 결과

LambdaRank는 기존 선택 구조인 leaves 63·min child 100·seed 42를 고정하고 300트리까지 재생했다. 아래 두 conditional 지표는 **회수 양성이 있는 그룹에 대한 후보 내부 IDCG**로 계산했다. Train과 validation의 정의는 같지만 모집단·Window 길이는 다르며, 전체 시스템 NDCG와 숫자를 직접 비교하지 않는다.

| 트리 수 | Train conditional NDCG@10 | Validation conditional NDCG@10 | Validation 전체 정답 NDCG@10 |
|---:|---:|---:|---:|
| 20 | 0.224877 | 0.091816 | 0.024033 |
| 40 | 0.283657 | 0.101592 | 0.026082 |
| 76 | 0.355943 | 0.105481 | **0.027694** |
| 100 | 0.389749 | 0.098058 | 0.025661 |
| 150 | 0.446805 | 0.097300 | 0.025075 |
| 300 | 0.557130 | 0.091337 | 0.023945 |

76 이후 train 점수는 계속 높아지지만 validation은 전반적으로 낮아져 추가 학습의 과적합 징후가 있다. 그러나 1~300트리의 전체 validation 곡선 최고점도 기존 76트리여서, 기존 선택 모델을 과도한 학습의 결과로 확정하지 않는다. 다른 구조·seed·시간 기간까지 검증한 결과는 아니다.

기존 early stopping은 회수 양성이 있는 730명·후보 내부 IDCG를 사용했고 보고는 관련 정답 사용자 1,639명·전체 IDCG를 사용했다. 같은 76트리의 0.105481과 0.027694는 서로 다른 정의의 점수다. 10월 12일 구현 교정은 필요했지만, 이번 고정 구조에서는 올바른 지표로 봐도 최고 트리 수가 같았다. 전체 7조건 grid의 수정 후 재선택 효과는 미검증이다.

| LightGCN epoch | Train BPR loss | Validation Recall@100 |
|---:|---:|---:|
| 1 | 0.664722 | 10.5791% |
| 5 | 0.252530 | 13.2850% |
| 10 | 0.165348 | 16.2071% |
| 15 | 0.117444 | 18.2510% |
| 20 | 0.087704 | **19.2869%** |

20epoch의 Recall은 기존 C4 validation을 재현했다. 검사한 5개 checkpoint에서는 주 지표가 계속 좋아져, 현재 20epoch 예산에서 후보 회수의 과적합 징후는 관찰되지 않았다. 손실 감소만으로 일반화를 판단하지 않았으며 20epoch 이후는 검사하지 않았다.

### Retrieval 설정의 선택 근거

Git의 과거 상세 보고서를 복원해 9월 28일 선택 범위를 확인했다. **Retrieval은 일부 설정과 source 조합만 validation으로 고르고, 나머지 설정은 고정한 제한된 탐색이었다.** 10월 4일 전체 baseline 실행에서는 이를 그대로 사용하고 LambdaRank만 탐색했다.

| 설정 | 현재 값 | 과거 선택·고정 근거 |
|---|---|---|
| LightGCN 층 수 | 3 | 1/2/3 비교 |
| LightGCN L2 | `1e-4` | `1e-4`/`1e-2` 비교 |
| LightGCN epoch | 20 | 5epoch마다 validation Recall@100 평가, 최고 지점 선택 |
| LightGCN 차원·학습률·batch | 64 / 0.005 / 2,048 | 위 6조건에서 고정; 최적화 근거 미확인 |
| 후보 source | C1+C4 | C4 단독·여러 source RRF 결합의 validation Recall@100 비교 |
| source별 검색 깊이·최종 후보 예산 | 각각 최대 100 / 100 | 고정; 두 예산이 같은 `candidate_k`에 연결됨 |
| RRF 상수·source 가중치 | 60 / 동일 | 고정; 비교 탐색 기록 미확인 |
| C1 유사도·이력 집계 | 공동 방문 cosine·전체 과거 이력의 합 | 현재 구현 고정; 감쇠·이력 길이·다른 유사도 대조 미확인 |

6조건의 best Recall@100은 layers 1의 L2별 16.90%/17.95%, layers 2의 18.43%/18.82%, layers 3의 19.29%/18.78%였다. 먼저 C4 단독 기준으로 설정을 고른 후 source 결합을 비교했다. 선택한 C4와 C1을 결합한 validation Recall@100은 19.48%였다. 결합 기준의 모든 파라미터나 최종 LTR NDCG를 공동 최적화한 결과는 아니다.

새로 복원한 과거 곡선에는 선택 조건의 epoch 20 이후도 기록돼 있다. Recall@100은 20epoch 19.29%에서 50epoch 18.30%로 낮아졌고 BPR loss는 계속 줄었다. 이 기록은 이번 20epoch까지의 직접 재생과 구분한다. 과거 원 metrics/checkpoint는 없어 복원 보고서의 모든 수치를 독립 재검산하지는 못했다. 또한 단일 seed·동일 validation 탐색이며 현재 개인별 만족도 이전 NDCG 수치를 혼합하지 않는다.

따라서 현재 설정을 retrieval 전체의 최적값으로 해석하지 않는다. 후속 검증은 최종 후보 예산을 고정한 상태에서 source별 검색 깊이, RRF 설정, C4 차원·학습률·이력 정책을 제한된 사전 지정 범위로 비교하고, 시간 기간별 재현을 확인하는 방식이 적절하다. 후보가 바뀌면 LambdaRank 학습 후보·피처도 다시 만들어야 한다.

근거: [복원한 9월 28일 보고서](../artifacts/diagnostics/baseline_revalidation_20261013/historical_lightgcn_report.md), [출처·선택 범위](../artifacts/diagnostics/baseline_revalidation_20261013/retrieval_parameter_provenance.json), [현재 C5 결합 구현](../src/rating_recsys/retrieval/hybrid.py), [현재 설정](../src/rating_recsys/experiments/config.py).

### 판단과 다음 기준

현재 C5 후보 기준을 유지하되 LambdaRank의 독립적인 추가 이득은 계속 미확인으로 둔다. 다음에는 날짜 표기·연도 추정을 확인하고, 동일한 예측 기간의 rolling validation에서 설정을 고정해 재검증한다. 기존 test를 보고 설정을 바꾸지 않는다. 실제 노출 로그 없이 미관측 식당의 만족도를 알 수 없으므로 현재 과제는 관측 미래 방문·평점과의 일치라는 한계도 유지한다.

근거: [분할 감사](../artifacts/diagnostics/baseline_revalidation_20261013/split_audit.json), [저장 모델 감사](../artifacts/diagnostics/baseline_revalidation_20261013/saved_model_audit.json), [LTR 학습 곡선](../artifacts/diagnostics/baseline_revalidation_20261013/learning.json), [C4 학습 곡선](../artifacts/diagnostics/baseline_revalidation_20261013/graph_learning.json). 재현 진입점은 [감사 스크립트](../scripts/audit_baseline_learning.py)이며 실행 소스와 정확한 범위는 같은 artifact 폴더에 보존한다.

## 결과

| 구성 | Test Recall@100 | NDCG@10 |
|---|---:|---:|
| C1 공동 방문 후보 순서 | 17.8127% | 0.023961 |
| C4 LightGCN 후보 순서 | 18.7510% | 0.021563 |
| C5: C1+C4 RRF 후보 순서 | 20.1626% | 0.026590 |
| 같은 C5 후보 → LambdaRank | 20.1626% | 0.025467 |

Recall@100의 20.1626%는 사용자의 미래 관련 식당 중 후보 100개에 담긴 비율을 사용자별로 평균한 값이다. 만족한 사용자 비율이나 전국 모든 식당의 정확도를 뜻하지 않는다. 같은 C5를 재정렬하면 후보 집합이 같아 Recall@100도 같고, 상위 순서가 달라지는 효과는 NDCG@10으로 확인한다.

## 해석

같은 사용자의 두 결과를 함께 재표본추출한 paired bootstrap으로 차이를 확인했다.

| Test 비교 | 관측된 차이 | 판단 |
|---|---|---|
| C5−C1 Recall@100 | +2.35%p; 95% CI [+1.22, +3.43]%p | 이 실행에서 후보 회수 보완 확인 |
| C5−C4 Recall@100 | +1.41%p; 95% CI [+0.27, +2.52]%p | 이 실행에서 후보 회수 보완 확인 |
| C4−C1 Recall@100 | 차이 구간에 0 포함 | 단독 모델의 우열 미확인 |
| LambdaRank−C5 NDCG@10 | −0.001124; 95% CI [−0.005053, +0.002694] | 재정렬 이득 미확인 |

지역 집중과 만족도 구별 능력은 별도 진단했다.

| 진단 | 결과 | 해석의 범위 |
|---|---|---|
| 지역 feature 제거 재학습 | NDCG@10 0.025467→0.024286; 차이 구간에 0 포함 | 지역 feature 이득 미확인. Gain 비중 7.25%는 성능 기여율이 아님 |
| Top-10의 주 활동 지역 비중 | C5 82.24%, LambdaRank 84.14%, 지역 feature 제거 78.57% | 제거 후에도 집중됨. 방문·공동 방문 신호에도 지역 정보가 담길 수 있음 |
| 만족도 차이가 있는 식당쌍의 순서 | 사용자 평균 50.88%; 95% CI [42.30%, 60.18%] | 후보 안의 91명·234쌍 진단. 무작위 기대값 50%를 포함하며 전체 추천 정답률이 아님 |

과거 Prefix·절대 등급에서 평균 평점 shrinkage λ=10도 비교했다. Validation NDCG@10은 0.025151→0.027881, test는 0.029360→0.028207로 test 개선을 확인하지 못했다. 정답·학습이 달라 현재 Window 결과와 직접 비교하지 않는다.

<a id="왜-리뷰-모델보다-강하게-나타나는가"></a>

## 행동 모델의 상대적 강점

2026-10-10 코드와 저장된 validation 순위를 다시 점검했다. **기준 모델은 공동 방문 관계를 활용하지만 당시 리뷰 Transformer는 그 점수·학습된 그래프 표현을 이어받지 않았다.** 성능 격차의 원인을 분리한 대조 학습은 아니므로, 이 비교만으로 텍스트 효과를 판단하지 않는다.

저장된 validation 사용자 1,645명 중 relevance 정답이 있는 1,639명을 재계산했다. 이번 점검에서 새 학습·예측·test 분석·API 호출은 수행하지 않았다.

| 같은 validation | Recall@100 | NDCG@10 |
|---|---:|---:|
| C1 공동 방문 유사도 | 16.5149% | 0.021895 |
| C4 LightGCN | 19.2869% | 0.020708 |
| C5: C1+C4 순위 결합 | 19.4769% | 0.023311 |
| 같은 C5 → 기존 16피처 LambdaRank | 19.4769% | 0.027694 |

LambdaRank의 0.027694는 같은 validation에서 7개 설정을 비교한 선택 결과다(`leaves=63`, `min_child_samples=100`, 76 trees; 미조정 reference 0.024387). 새 Transformer의 고정 설정 하나와 탐색 예산이 같지 않았고, 독립 holdout 점수도 아니다. [선택 기록](../artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add/selection.json).

| 모델 | 실제 활용한 신호 | 비교에 영향을 주는 차이 |
|---|---|---|
| C1 | `공통 방문 사용자 수 / sqrt(각 식당 방문 사용자 수 곱)`을 과거 방문 식당별로 합산 | 저평점 방문도 관계에 포함. 문장·맛·서비스의 유사성을 직접 측정하지 않음 |
| C4 LightGCN | Cutoff 이전 고유 사용자–식당 방문 그래프의 3층 전파 | 직접 이웃 수 외에 그래프를 거친 관계를 학습 |
| 당시 리뷰 Transformer | 별도로 무작위 초기화한 ID embedding + 리뷰 표현 | LightGCN 초기화·전파·C1 점수 없음. 적격 후속 양성으로 학습된 사용자는 전체 ID 11,655명 중 4,705명 |

LightGCN의 일반 전파 구조는 [원 논문](https://arxiv.org/abs/2002.02126)을 따른다. 현재 정답은 미래 방문 중 개인별 만족도 기준을 넘은 식당이므로 공동 방문의 지역·접근성·선호 신호가 목표와 맞을 가능성이 있다. 각 요인의 실제 기여는 분리하지 않았다.

| Validation 후보 이동 | 관측 결과 |
|---|---|
| C1→C5 Top-100 | 기존 정답 195쌍 탈락, 새 정답 341쌍 유입 |
| C5−C1 Recall@100 | +2.9620%p; paired 95% CI [+1.9910, +3.9052]%p |
| C5−C4 Recall@100 | +0.1900%p; paired 95% CI [−0.9199, +1.2980]%p |

C1 보완은 관측됐지만 LightGCN 단독보다 결합이 항상 좋다고 단정하지 않는다. C5의 후보 100개 회수율도 약 19.5%여서 검색 개선 여지가 남아 있다.

## 후속 비교와 현재 과제

| 상태 | 내용 |
|---|---|
| 완료·미채택 | 같은 방문 그래프에 E5 텍스트 정렬을 더한 RLMRec 비교. 가중치·shuffle·전체 ranker 재학습까지 일관된 개선 미확인 |
| 완료·미채택 | 동일 Two-Tower의 텍스트 유무 및 텍스트 그래프·학습형 다중 벡터의 3seed retrieval 대조. 최신 결과는 [리뷰 임베딩 리포트](./review_embeddings.md)를 따름 |
| 과거 설계·미실행 | 고정 행동 후보·점수에 본문 잔차를 더하는 [추가효과 설계](./review_embeddings.md#베이스라인을-유지한-텍스트-추가효과-검증). 현재 후속 범위에서 제외 |
| 현재 과제 | 전체 cutoff catalog에서 새 후보를 찾는 retrieval 개선. Recall@100으로 평가하며 미래 방문·만족도에 맞춘 텍스트 학습을 새 holdout 또는 과거 rolling validation에서 검증 |

후속 3seed 결과와 위 단일 seed의 과거 test/validation을 같은 표의 직접 비교로 합치지 않는다.

## 근거

| 근거 | 위치 |
|---|---|
| 기준 실행·test | [실행 보고서](../artifacts/runs/20261004T125314601899Z-e7896add/report.md) |
| 지역·만족도 진단 | Run `20261005T140007795282Z-e7896add`; `baseline_features.zip` |
| Prefix shrinkage 비교 | Run `20261002T062004863532Z-e7896add`; `shrinkage.zip` |
| Validation 저장 순위 재검산 | [baseline audit](../artifacts/diagnostics/baseline_text_context_audit.json), [paired 비교·RRF 감사](../artifacts/diagnostics/validation_rrf_signal_audit.json) |
| 과거 상세 리포트 | `detailed_reports.zip`; [보관·복원 안내](../artifacts/README.md) |
