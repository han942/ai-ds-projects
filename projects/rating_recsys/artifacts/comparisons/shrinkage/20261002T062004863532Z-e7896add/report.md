# V2 평점 평균 shrinkage 비교 · 20261002T062004863532Z-e7896add

- Snapshot `e7896add5b4b5939` · 88,554 interactions
- 사용자 14,008명 · 식당 4,587개 · test 평가 query 1,573개
- Train ≤ 2025-12-19, validation ≤ 2026-05-04, test는 그 이후
- 실행 단계 합계 20.6분 · validation/test 기간에 처음 등장한 사용자는 정확도 평가에서 제외
- 조건: C5 Top-100, prefix/relevance, seed 42, 동일 ranker grid와 동점 처리
- 처리: 사용자·식당 평균 두 feature에만 `(평점 합 + λ × 과거 전체 평균)/(평가 수 + λ)`, λ=10
- λ는 결과를 보기 전에 고정했다. 모델 종류·텍스트·label·window는 유지했다. V1은 사용하지 않음.
- 후보 추출·학습 group·label은 공유하고 두 ranker의 설정과 트리 수는 같은 규칙으로 validation에서 각각 선택.

## 결과

공통 C5 Recall@100: 20.1626%

| 지표 | Baseline | Shrinkage | 차이 (shrinkage − baseline) |
|---|---:|---:|---:|
| ndcg@5 | 0.023906 | 0.021840 | -0.002066 |
| recall@5 | 0.025630 | 0.021597 | -0.004033 |
| precision@5 | 0.019072 | 0.015766 | -0.003306 |
| map@5 | 0.014825 | 0.013636 | -0.001189 |
| mrr@5 | 0.045200 | 0.038271 | -0.006929 |
| catalog_coverage@5 | 0.395092 | 0.450365 | +0.055273 |
| novelty@5 | 10.951365 | 11.230480 | +0.279115 |
| intra_list_region_diversity@5 | 0.139747 | 0.227658 | +0.087911 |
| ndcg@10 | 0.029360 | 0.028207 | -0.001153 |
| recall@10 | 0.041284 | 0.040369 | -0.000915 |
| precision@10 | 0.015639 | 0.014240 | -0.001399 |
| map@10 | 0.015684 | 0.014994 | -0.000690 |
| mrr@10 | 0.051154 | 0.044807 | -0.006346 |
| catalog_coverage@10 | 0.583683 | 0.606898 | +0.023215 |
| novelty@10 | 11.178054 | 11.151536 | -0.026518 |
| intra_list_region_diversity@10 | 0.136582 | 0.220858 | +0.084276 |

NDCG@10 Δ=-0.001153, paired bootstrap 95% CI [-0.005006, +0.002692]. **NDCG 차이를 확정할 근거 부족**.

## Paired bootstrap (shrinkage − baseline)

주지표는 NDCG@10이며 아래는 사용자별 2,000회 재표집한 95% 신뢰구간이다. 단위는 점수 차이다.

| 지표 | 차이 | 95% CI |
|---|---:|---|
| ndcg@5 | -0.002066 | [-0.006208, +0.002081] |
| recall@5 | -0.004033 | [-0.009445, +0.001368] |
| precision@5 | -0.003306 | [-0.005976, -0.000636] |
| map@5 | -0.001189 | [-0.004370, +0.002055] |
| ndcg@10 | -0.001153 | [-0.005006, +0.002692] |
| recall@10 | -0.000915 | [-0.007519, +0.005788] |
| precision@10 | -0.001399 | [-0.002924, +0.000191] |
| map@10 | -0.000690 | [-0.003679, +0.002306] |

Precision@5는 −0.3306%p이며 명목 95% CI [−0.5976, −0.0636]%p가 0 아래다. NDCG@10의 차이는 불확실하다. 여러 보조 지표를 함께 확인했고 다중 비교 보정을 하지 않은 탐색 결과로 해석한다.

## Validation 선택

| 조건 | leaves | min_child_samples | trees | Val NDCG |
|---|---:|---:|---:|---:|
| baseline | 15 | 10 | 1 | 0.025151 |
| shrinkage | 63 | 10 | 96 | 0.027881 |

## 해석과 한계

- 기본값은 λ=0으로 유지한다. 이 한 번의 비교로 shrinkage를 기본 모델에 채택하지 않는다.
- 이 실험은 평점 평균 feature 두 개의 shrinkage만 측정한다. 사용자별 label 보정이나 MF bias 모델의 효과로 확대 해석하지 않는다.
- 후보 Recall은 같은 추출 결과이므로 동일하다. LTR 이후의 순위 성능으로 판단한다.
- Seed 1회, 보정 강도 1개. 이미 확인한 test window의 탐색적 비교이며 최종 일반화 확인에는 새 미래 holdout이 필요하다.
- 과거 2026-09-30 run과 달리 현재 동점 처리와 공통 validation metric을 양쪽에 적용해 baseline을 다시 학습했다.

## 재현

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python -m rating_recsys.experiments.shrinkage \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --strength 10
```

원본: manifest.json, metrics.json, recommendations_validation/test.jsonl(공유 C5 후보·정답·양쪽 순위), model_baseline/shrinkage.txt, source.diff, runner_source.py.

## 이번 실행의 판단과 검증

Validation NDCG@10은 개선됐지만 test에서는 NDCG·Recall·Precision·MAP·MRR@5/10의 측정값이 모두 낮았다. NDCG@10 차이의 신뢰구간은 0을 포함한다. 이번 비교에서는 λ=0 baseline을 유지한다. 이는 모든 shrinkage 방식이 나쁘다는 결론은 아니다.

최종 shrinkage 모델에서 사용자 평균·식당 평균은 각각 514회·618회 트리 분할에 사용됐다. 이 숫자는 gain 중요도가 아닌 분할 횟수다. Baseline은 각각 0회·1회다. 양쪽 설정을 validation에서 고르는 절차까지 포함한 비교이며, 고정된 동일 트리 구조의 두 평균만 바꿔 추론한 실험은 아니다.

프로젝트 테스트 92개가 통과했다. 저장된 추천을 독립 재계산해 validation 1,645명(positive 1,639명), test 1,580명(positive 1,573명)의 NDCG·Recall·Precision·MAP·MRR@5/10과 test paired bootstrap 신뢰구간의 일치를 확인했다. 모든 query의 공유 후보·이력 제외·정답 window·동점 순서도 확인했다. `verification_source.py`와 `verification.json`에 남겼다. 표 표시를 확장한 보고서 생성 코드는 `report_renderer_source.py`, 실제 학습 당시 코드는 `runner_source.py`다. 모델 학습을 추가로 실행하지 않았다.
