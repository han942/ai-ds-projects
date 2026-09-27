# Region ablation: C3→R1 vs C0+C1→R1

> 보관 문서 · primary leave-last-two-out 프로토콜. 현재 코드에는 이 평가 경로가 없다. 본문의 명령과 모듈은 commit `ae3f4ea` 기준이며, 재현 방법은 [보관 안내](./README.md)를 따른다.

2026-09-23에 같은 데이터와 평가 규칙으로 두 모델을 새로 학습·평가했다. 이전 C0+C1→R1 run의 수치는 비교에 사용하지 않았다.

## 실험 조건

- 지역 포함 기준선: C0 popularity, C1 item-item, C2 region popularity를 quota RRF로 결합해 Top-100 후보를 만들고 R1 LambdaRank가 재정렬한다.
- 무지역 실험: C2 후보와 R1의 `region_affinity` feature를 제거한다. 식당 지역 metadata는 지역 다양성 **평가에만** 사용한다.
- 두 run 모두 첫 user-item interaction 23,017건, seen user 2,396명, train/validation/test 12,504/2,396/2,396건을 사용한다. Test에서 relevance > 0인 query는 2,350개다.
- 동일한 snapshot ID: `03a763252e30c5516dd4720c9c107e7dbfb41fbadda696caccd29afe106b9785`. Seed 42, 후보 K=100, 추천 K=10, split 및 기타 설정이 동일하다.
- MLflow 업로드 없이 로컬 artifact로 실행했다.

## Test 결과

| 지표 | 지역 포함 | 무지역 | 무지역 − 지역 포함 |
|---|---:|---:|---:|
| 후보 Recall@20 | 16.89% | 16.89% | 0.00%p |
| 후보 Recall@50 | 31.11% | 31.11% | 0.00%p |
| 후보 Recall@100 | 51.53% | 46.72% | **−4.81%p** |
| R1 Recall@5 | 7.45% | 5.70% | −1.74%p |
| R1 Recall@10 | 11.62% | 9.70% | **−1.91%p** |
| R1 NDCG@10 | 0.0600 | 0.0485 | **−0.0116** |
| R1 catalog coverage@10 | 93.67% | 91.39% | −2.29%p |
| R1 지역 다양성@10 | 32.79% | 52.49% | +19.70%p |
| 후보 생성 p95 지연시간 | 9.49 ms | 8.55 ms | −0.94 ms |

후보 Recall@20/50이 같은 이유는 C3가 C0+C1 상위 50개를 먼저 보존하기 때문이다. 지역 후보를 제거하면 남은 50개에서 relevant target을 더 자주 놓친다.

같은 test query를 짝지어 10,000회 bootstrap한 무지역 − 지역 포함 차이의 95% 구간은 후보 Recall@100 **[−6.17, −3.45]%p**, R1 Recall@10 **[−3.15, −0.68]%p**, R1 NDCG@10 **[−0.0181, −0.0053]**이다. 후보 target을 지역 포함에서만 찾은 query는 192개, 무지역에서만 찾은 query는 79개였다. R1 Top-10 target은 각각 135개와 90개였다.

Validation에서는 R1 Recall@10이 10.82% → 10.23%(−0.59%p), NDCG@10이 0.0540 → 0.0529(−0.0012)였다.

## 해석

이 snapshot에서는 지역 입력을 유지하는 편이 후보 발견과 최종 추천 정확도에 유리하다. 무지역 모델은 추천 목록의 지역 다양성이 높고 후보 생성이 약간 빠르다. 따라서 정확도를 우선하는 현재 baseline에서는 지역 포함 구성을 유지한다.

이 실험은 **C2 후보와 region affinity feature를 함께 제거**했으므로 어느 쪽이 정확도 차이를 만들었는지는 분리할 수 없다. 지연시간은 단일 실행의 측정값이며 안정적인 속도 우위로 해석하지 않는다. 새 snapshot이나 다른 seed에 대한 일반화도 아직 검증하지 않았다.

## Artifact

- [지역 포함 run](../../../artifacts/archive/primary/runs/20260923T024727601730Z-03a76325/manifest.json)
- [무지역 run](../../../artifacts/archive/primary/runs/20260923T025110890068Z-03a76325/manifest.json)
- [수치 비교 JSON](../../../artifacts/archive/primary/comparisons/region_ablation_20260923T024727601730Z-03a76325_vs_20260923T025110890068Z-03a76325.json)

재실행: `python -m rating_recsys.experiments.compare_region` (프로젝트 디렉터리의 `.venv` 환경에서 실행).
