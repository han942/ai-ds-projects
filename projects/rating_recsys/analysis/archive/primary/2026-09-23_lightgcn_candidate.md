# LightGCN 후보 생성 실험: C0+C1+LightGCN RRF

> 보관 문서 · primary leave-last-two-out 프로토콜. 현재 코드에는 이 평가 경로가 없다. 본문의 명령과 모듈은 commit `ae3f4ea` 기준이며, 재현 방법은 [보관 안내](./README.md)를 따른다.

2026-09-23에 현재 DB snapshot의 동일 validation/test query로 후보 생성 방식을 비교했다. 현재 baseline C3와 달리 이 실험은 **후보 생성까지만** 평가한다. 기존 R1 LambdaRank의 학습·재정렬 결과를 이 수치로 추정하지 않는다.

## 구현과 조건

- C0: 전체 인기, C1: cosine item co-occurrence, 새 소스: [LightGCN](https://arxiv.org/abs/2002.02126). 각 소스의 상위 100개를 `1 / (60 + source rank)`로 합산하고 Top-100을 선택한다. 새 RRF에는 지역 소스와 기존 C3의 50개 보존 quota가 없다.
- LightGCN은 사용자–식당 이분 그래프의 정규화 인접 행렬을 2회 전파하고 0~2층 임베딩을 평균한다. 사용자–positive item–sampled negative item의 BPR 손실로 초기 임베딩을 학습한다. 차원 32, 80 epoch, Adam learning rate 0.03, L2 계수 0.0001, seed 42. 모델은 NumPy/SciPy로 구현했다.
- 질의가 속한 **분기의 시작일보다 앞선** 참조 interaction만 LightGCN 학습에 사용한다. Validation은 train interaction만, test는 train+validation interaction만 모델과 C0/C1/C3의 참조 데이터로 사용한다. C0/C1/C3는 기존과 같이 각 질의보다 앞선 interaction으로 갱신한다. 미래 interaction은 그래프 학습에도 후보 생성에도 사용하지 않는다.
- 원래 seen-user split 23,017건, 적격 사용자 2,396명, train/validation/test 12,504/2,396/2,396건. Test에서 relevance > 0인 query는 2,350개다. Snapshot ID는 `03a763252e30c5516dd4720c9c107e7dbfb41fbadda696caccd29afe106b9785`로 [이전 지역 제거 실험](./2026-09-23_region_ablation.md)과 같다.
- 이전 비교값을 복사하지 않고 이 실행에서 C0, C1, 지역 포함 C3 후보를 모두 다시 계산했다. C3 Test Recall@100 51.53%, C0+C1 46.72%는 이전 실행값과 일치한다. 질의별 C3 후보 순서도 이전 지역 포함 run의 validation/test 각 2,396개와 모두 일치하고, C0+C1 후보 순서는 이전 무지역 test run의 2,396개와 모두 일치한다. 모든 결과는 MLflow 전송 없이 로컬에 저장했다.

## 후보 결과

하나의 query에 held-out 식당 하나를 두고, relevance > 0인 query에서 후보 목록에 포함된 비율이다. % 단위다.

| 후보 생성 | Validation @20 | Validation @50 | Validation @100 | Test @20 | Test @50 | Test @100 |
|---|---:|---:|---:|---:|---:|---:|
| C0+C1 RRF | 16.74 | 31.84 | 46.13 | 16.89 | 31.11 | 46.72 |
| **C0+C1+LightGCN RRF** | **18.27** | **32.85** | **47.99** | **17.91** | **31.40** | **48.26** |
| 현재 C3: C0+C1+C2 지역 quota RRF | 16.74 | 31.84 | **50.74** | 16.89 | 31.11 | **51.53** |

Test에서 LightGCN 추가로 C0+C1 대비 Recall@20은 **+1.02%p**, Recall@100은 **+1.53%p** 높았다. 같은 2,350개 relevant query를 짝지어 5,000회 bootstrap한 Recall@100 차이의 95% 구간은 **[+0.17, +2.89]%p**다. LightGCN RRF만 target을 찾은 query는 147개, C0+C1만 찾은 query는 111개였다.

현재 지역 포함 C3와 비교하면 LightGCN RRF는 Test Recall@20에서 **+1.02%p**, Recall@100에서 **−3.28%p**다. Recall@100 차이의 paired bootstrap 95% 구간은 **[−4.47, −2.09]%p**이며, 현재 C3만 target을 찾은 query는 145개, LightGCN RRF만 찾은 query는 68개였다. 후보 NDCG@20은 Test에서 C0+C1 0.0696, LightGCN RRF 0.0721, C3 0.0696이다. Test catalog coverage@100은 세 방식 모두 100%였다.

LightGCN 단독 Test Recall@100은 **40.34%**다. 분기 모델이 해당 사용자를 알고 있어 graph 점수를 낸 query는 **2,186/2,396개**다. 나머지는 C0+C1 RRF로 후보를 만든다. LightGCN 단독 후보가 최종 RRF에 기여한 Test target은 841건이지만, 그중 LightGCN에만 속한 target은 23건이다. 다른 소스와 겹치는 후보가 많다.

## 판단과 범위

LightGCN을 세 번째 후보 소스로 구현할 수 있고, C0+C1 조합의 후보 품질은 이 snapshot에서 개선된다. **현재 지역 포함 C3를 대체할 근거는 아직 없다.** 특히 운영상 중요한 Recall@100은 C3가 3.28%p 높다. 새 방식의 Top-20 품질은 좋아졌으므로 작은 후보 예산을 목표로 한 후속 비교에는 가치가 있다.

현재 비교는 지역 소스를 LightGCN으로 바꾸는 동시에 기존 C3의 50개 보존 quota를 제거한다. C3와의 차이는 소스와 결합 정책이 모두 영향을 줄 수 있다. C0+C1 대비 차이는 동일한 순수 RRF에서 LightGCN 소스 추가 효과다. 분기별 모델은 그 분기 중 새로 생긴 사용자·식당을 학습하지 못해 해당 질의에는 graph 점수가 없을 수 있다. 2개 source fallback을 적용했다. 새 R1을 학습하지 않았으므로 최종 Top-10 추천 품질에 대한 결론은 없다. 학습 시간은 validation 약 47초, test 약 51초이며 후보 평가 루프는 각각 약 30초였으나, 이는 단일 로컬 실행의 총 처리 시간이고 서비스 요청 지연시간이 아니다.

## 재실행과 아티팩트

```bash
python -m rating_recsys.experiments.compare_lightgcn
```

- [실험 manifest](../../../artifacts/archive/primary/comparisons/lightgcn_20260923T045007231844Z-03a76325/manifest.json)
- [설정](../../../artifacts/archive/primary/comparisons/lightgcn_20260923T045007231844Z-03a76325/config.json)
- [Validation 상세 지표](../../../artifacts/archive/primary/comparisons/lightgcn_20260923T045007231844Z-03a76325/metrics_validation.json)
- [Test 상세 지표](../../../artifacts/archive/primary/comparisons/lightgcn_20260923T045007231844Z-03a76325/metrics_test.json)
- [Test 질의별 후보 목록](../../../artifacts/archive/primary/comparisons/lightgcn_20260923T045007231844Z-03a76325/candidates_test.jsonl)
