# 실험 보고서 · 20260930T043202619555Z-e7896add

- 실행: 2026-09-30 04:32 UTC · label `c5-baseline`
- 데이터: snapshot `e7896add5b4b5939` · 88,554 interactions · 사용자 14,008 · 식당 4,587 · 2016-04-19 ~ 2026-09-26
- 코드: commit `1dfeb828a7` (미커밋 변경 포함, `source.diff`)
- 후속(2026-09-30): 이 run의 R1은 학습 정답을 후보 밖 순위 101로 끼워 넣은 행에 의존했다.
  학습 정답 끼워넣기를 없애고 다시 실행한
  [`20260930T135424227862Z-e7896add`](../20260930T135424227862Z-e7896add/report.md)가 현재 기준선이다.
  이 run의 `source.diff`에는 새로 추가한 미추적 파일(`retrieval/hybrid.py` 등)이 빠져 있다.

## 1. 요약

- 후보 생성: C5 Recall@100 20.16% (C1 단독 17.81%, C4 LightGCN 단독 18.75%; 참고 C3 16.66%, C5 − C3 +3.50%p [+2.30, +4.70])
- LTR: NDCG@10 0.0276 → 0.0253 (-0.0023 [-0.0067, +0.0020]), Recall@10 4.03% → 3.41% (-0.61%p [-1.26, +0.06])
- Validation 선택: LambdaRank num_leaves 63 · min_child_samples 10 · trees 54. LightGCN은 고정 설정(3층 · 64차원 · L2 0.0001 · 20 epoch)

## 2. 데이터와 조건

| 구간 | 기간 | interactions | 평가 사용자 | 사용자당 정답 | 정답이 catalog에 있음 |
|---|---|---:|---:|---:|---:|
| Train | ~ 2025-12-19 | 70,883 | – | – | – |
| Validation | 2025-12-19 이후 ~ 2026-05-04 | 8,934 | 1,639 | 3.79 | 96.24% |
| Test | 2026-05-04 이후 | 8,737 | 1,573 | 3.57 | 98.08% |

평가 사용자는 cutoff 이전 이력이 있고 window에 평점 3.0 이상 방문이 있는 사용자다. 이력이 없는 사용자(validation 1,139명, test 1,214명)는 평가하지 않는다. Ranker 학습 정답은 튜닝 시 2025-12-19, 최종 재학습 시 2026-05-04까지다.

주요 조건: `candidate_k` 100, `ranking_k` 10, `region_mode` with_region, `relevance_high_threshold` 4.0, `relevance_low_threshold` 3.0, `train_fraction` 0.8, `validation_fraction` 0.1, `random_seed` 42

기본값과 다른 조건: 없음

## 3. 설정 선택

C4 LightGCN: 3층 · 64차원 · L2 0.0001 · learning rate 0.005 · batch 2048 · 20 epoch (config 고정값, 이 run에서 고르지 않음). Validation window 모델은 train 70,883 edges, test window 모델은 train + validation 79,817 edges로 학습했다.

Ranker 학습 query는 3개월 단위 구간 시작일 이전 interaction으로 다시 학습한 LightGCN을 쓴다. 모델 40개 (학습 247s), LightGCN 후보가 있었던 학습 query 56,088/67,023 (83.68%). 나머지는 그 시점 그래프에 없던 사용자라 C1 순서만 쓴다.

LambdaRank: 기준은 validation R1 NDCG@10 최대 (동률이면 Recall@10, 그다음 grid 순서). `reference_untuned`은 early stopping 없이 고정 설정으로 학습한 비교 기준이다.

| 설정 | leaves | min_child | trees | Val NDCG@10 | Val Recall@10 |
|---|---:|---:|---:|---:|---:|
| reference_untuned | 15 | 10 | 150 | 0.0226 | 3.25% |
| leaves15_minchild10 | 15 | 10 | 3 | 0.0250 | 3.48% |
| leaves15_minchild100 | 15 | 100 | 3 | 0.0250 | 3.48% |
| leaves31_minchild10 | 31 | 10 | 3 | 0.0260 | 3.76% |
| leaves31_minchild100 | 31 | 100 | 3 | 0.0260 | 3.76% |
| **leaves63_minchild10** | 63 | 10 | 54 | 0.0261 | 3.86% |
| leaves63_minchild100 | 63 | 100 | 19 | 0.0251 | 3.55% |

최종 모델: 선택한 설정, 트리 수 = validation best iteration, train + validation window prefix query로 재학습. Validation에서 C5 Recall@100 19.48%, R1 NDCG@10 0.0261.

## 4. 후보 생성 (test)

Recall@K = 사용자별 (후보 K개 안의 정답 수 / 전체 정답 수)의 평균.

| 후보 | Recall@20 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| **C5 C1+LightGCN RRF (Stage 1)** | 6.61% | 12.90% | 20.16% |
| C1 item-item | 5.83% | 11.35% | 17.81% |
| C4 LightGCN | 5.89% | 11.56% | 18.75% |
| 참고 · C0 전체 인기 | 2.67% | 5.56% | 9.04% |
| 참고 · C2 지역 인기 | 4.12% | 7.90% | 14.17% |
| 참고 · C3 quota RRF (이전 기준선) | 4.55% | 9.85% | 16.66% |

C5 − C3 Recall@100: +3.50%p [+2.30, +4.70] (paired bootstrap, C5가 나은 사용자 296명, 나쁜 사용자 148명). C0·C2·C3는 Stage 1에 결합하지 않고 비교용으로만 남긴다. C0 순위와 C2 지역 비율은 ranker feature로 쓴다.

## 5. LTR 재정렬 (test, Top-10)

R0는 C5 후보 순서를 그대로 자른 목록, R1은 LambdaRank로 재정렬한 목록이다. CI는 같은 사용자끼리 짝지은 paired bootstrap 2,000회.

| 지표 | R0 C5 순서 | R1 LambdaRank | R1 − R0 (95% CI) |
|---|---:|---:|---|
| Recall@5 | 2.38% | 1.91% | – |
| Recall@10 | 4.03% | 3.41% | -0.61%p [-1.26, +0.06] |
| Precision@5 | 1.81% | 1.54% | – |
| Precision@10 | 1.44% | 1.36% | -0.08%p [-0.25, +0.09] |
| NDCG@5 | 0.0221 | 0.0203 | – |
| NDCG@10 | 0.0276 | 0.0253 | -0.0023 [-0.0067, +0.0020] |
| MAP@5 | 0.0126 | 0.0130 | – |
| MAP@10 | 0.0135 | 0.0134 | -0.0002 [-0.0035, +0.0032] |
| MRR@5 | 0.0403 | 0.0390 | – |
| MRR@10 | 0.0463 | 0.0454 | – |
| Catalog coverage@10 | 46.72% | 62.88% | – |
| Novelty@10 | 10.63 | 11.27 | – |
| 지역 다양성@10 | 21.06% | 18.04% | – |

NDCG@10 기준 R1이 나은 사용자 115명, 나쁜 사용자 134명, 같은 사용자 1,324명.

## 6. 해석 (작성자 기입)

- 결론: Stage 1 교체 효과는 확인했다. C5 Recall@100 20.16%로 이전 기준선 C3보다 +3.50%p
  [+2.30, +4.70] 높다. 이 값은 LightGCN 비교 run의 C1+L 결과와 같아서 두 구현이 일치한다.
  그러나 최종 Top-10은 좋아지지 않았다. R1은 C5 순서를 그대로 자른 R0보다 NDCG@10이
  −0.0023 [−0.0067, +0.0020]이고, 이전 run의 R1(C3 후보, 0.0268)과도 차이가 없다
  (−0.0016 [−0.0051, +0.0020], 같은 test 사용자 1,573명 paired bootstrap, 이 run 작성 시 별도 계산).
  현재 R1은 C5 후보 순서에 더하는 것이 없다.
- 근거: Validation에서는 R1이 R0보다 높았다(NDCG@10 0.0261 vs 0.0237). 하지만 early
  stopping이 여러 설정에서 3 tree에서 멈췄고, validation NDCG 곡선이 첫 몇 tree 뒤로 계속
  내려갔다. 학습 query의 70%(튜닝)·74%(재학습)는 정답을 injection으로 넣은 것이라 학습 분포가
  평가 분포와 다르다. Feature 중요도 1위는 `lightgcn_score_z`인데, 학습 query는 최대 3개월 전
  그래프(사용자 16%는 그래프에 없음)로, 평가 query는 cutoff 직전 그래프로 계산한다.
  Coverage@10은 46.7%→62.9%로 넓어졌지만 지역 다양성은 21.1%→18.0%로 줄었다.
- 한계: seed 1회. R1 설정은 validation 한 번으로 골랐고 CI가 없다. 날짜의 19.7%는 연도를 추론한
  값이다.
- 다음 실험: ① checkpoint 간격 1개월(`--lightgcn-checkpoint-months 1`)로 학습·평가 feature 차이
  줄이기, ② injection 없는 학습 또는 injected 행 가중치 낮추기, ③ validation 선택 후보에 "재정렬
  안 함(R0)"을 넣어 R1이 R0보다 못하면 R0를 쓰는 규칙, ④ seed 3~5회 반복.

## 7. 재현

```bash
rating-recsys-experiment \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

같은 폴더의 `manifest.json`(조건·구간·환경), `metrics.json`(모든 수치), `queries_*.jsonl`·`recommendations_*.jsonl`(사용자별 정답과 Top-K), `model.txt`(최종 ranker)가 이 보고서의 원본이다.
