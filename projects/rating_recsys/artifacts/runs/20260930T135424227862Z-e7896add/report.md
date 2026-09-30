# 실험 보고서 · 20260930T135424227862Z-e7896add

- 실행: 2026-09-30 13:54 UTC · label `no-injection`
- 데이터: snapshot `e7896add5b4b5939` · 88,554 interactions · 사용자 14,008 · 식당 4,587 · 2016-04-19 ~ 2026-09-26
- 코드: commit `1dfeb828a7` (미커밋 변경 포함, `source.diff`)

## 1. 요약

- 후보 생성: C5 Recall@100 20.16% (C1 단독 17.81%, C4 LightGCN 단독 18.75%; 참고 C3 16.66%, C5 − C3 +3.50%p [+2.30, +4.70])
- LTR: NDCG@10 0.0276 → 0.0276 (-0.0000 [-0.0046, +0.0044]), Recall@10 4.03% → 3.71% (-0.32%p [-0.98, +0.37])
- Validation 선택: LambdaRank num_leaves 31 · min_child_samples 10 · trees 1. LightGCN은 고정 설정(3층 · 64차원 · L2 0.0001 · 20 epoch)

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

Ranker 학습 query는 3개월 단위 구간 시작일 이전 interaction으로 다시 학습한 LightGCN을 쓴다. 모델 40개 (학습 257s), LightGCN 후보가 있었던 학습 query 56,088/67,023 (83.68%). 나머지는 그 시점 그래프에 없던 사용자라 C1 순서만 쓴다.

Ranker 학습 group은 정답이 그 query의 C5 후보 100개 안에 있는 query만 쓴다(정답을 후보에 끼워 넣지 않음): 정답 있는 학습 query 65,875개 중 15,929개 (24.18%).

LambdaRank: 기준은 validation R1 NDCG@10 최대 (동률이면 Recall@10, 그다음 grid 순서). `reference_untuned`은 early stopping 없이 고정 설정으로 학습한 비교 기준이다.

| 설정 | leaves | min_child | trees | Val NDCG@10 | Val Recall@10 |
|---|---:|---:|---:|---:|---:|
| reference_untuned | 15 | 10 | 150 | 0.0215 | 3.10% |
| leaves15_minchild10 | 15 | 10 | 1 | 0.0246 | 3.56% |
| leaves15_minchild100 | 15 | 100 | 1 | 0.0246 | 3.56% |
| **leaves31_minchild10** | 31 | 10 | 1 | 0.0246 | 3.45% |
| leaves31_minchild100 | 31 | 100 | 1 | 0.0246 | 3.45% |
| leaves63_minchild10 | 63 | 10 | 1 | 0.0243 | 3.50% |
| leaves63_minchild100 | 63 | 100 | 1 | 0.0243 | 3.50% |

최종 모델: 선택한 설정, 트리 수 = validation best iteration, train + validation window prefix query로 재학습. Validation에서 C5 Recall@100 19.48%, R1 NDCG@10 0.0246.

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
| Recall@5 | 2.38% | 2.24% | – |
| Recall@10 | 4.03% | 3.71% | -0.32%p [-0.98, +0.37] |
| Precision@5 | 1.81% | 1.61% | – |
| Precision@10 | 1.44% | 1.45% | +0.01%p [-0.17, +0.18] |
| NDCG@5 | 0.0221 | 0.0219 | – |
| NDCG@10 | 0.0276 | 0.0276 | -0.0000 [-0.0046, +0.0044] |
| MAP@5 | 0.0126 | 0.0143 | – |
| MAP@10 | 0.0135 | 0.0152 | +0.0016 [-0.0018, +0.0051] |
| MRR@5 | 0.0403 | 0.0432 | – |
| MRR@10 | 0.0463 | 0.0492 | – |
| Catalog coverage@10 | 46.72% | 56.89% | – |
| Novelty@10 | 10.63 | 11.28 | – |
| 지역 다양성@10 | 21.06% | 15.74% | – |

NDCG@10 기준 R1이 나은 사용자 116명, 나쁜 사용자 139명, 같은 사용자 1,318명.

## 6. 해석 (작성자 기입)

- 결론: 학습 정답 끼워넣기(injection)를 없애자 R1이 R0보다 나빠지던 현상은 사라졌지만,
  R1은 여전히 C5 순서에 더하는 것이 없다. Test NDCG@10 R1 − R0 −0.0000 [−0.0046, +0.0044],
  Recall@10 −0.32%p [−0.98, +0.37]로 둘 다 차이가 없다. 이 run이 이전 run
  `20260930T043202619555Z-e7896add`(injection 사용)를 대체하는 C5 기준선이다. 두 run의 R1끼리는
  NDCG@10 +0.0023 [−0.0012, +0.0057]로 차이가 없다(같은 test 1,573명 paired bootstrap, 작성 시
  별도 계산). Stage 1 결과(C5 Recall@100 20.16%)는 이전 run과 같다.
- 근거: 이전 run의 최종 모델은 트리 54개 전부가 첫 분기를 `candidate_rank_inverse ≤ 0.00995`로
  했다. 끼워 넣은 정답 행만 후보 순위 101(=1/101)을 가져서, ranker가 feature 대신 이 위치를
  배웠다. 이번에는 정답이 자기 C5 후보 안에 있는 학습 query(24.18%, 15,929 group)만 쓴다.
  그러자 validation early stopping이 모든 grid에서 트리 1개에서 멈췄다. 트리 1개(첫 분기
  `region_affinity`)는 한 query 안의 후보를 몇 묶음으로만 나눈다. Test Top-10에서 서로 다른
  점수는 query당 중앙값 3개뿐이고 93%의 자리가 동점이다. 동점은 `rank_window`가 식당 id
  순으로 정렬하므로, 인접한 동점 쌍의 47%가 C5 순서와 반대로 놓였다. MAP·MRR@10이 조금 높고
  Recall@10이 조금 낮은 것은 이 묶음과 id 정렬의 결과로 보인다(CI는 모두 0을 포함).
- 한계: seed 1회. 학습 query는 정답 1개짜리 prefix query이고 평가 query는 약 4개월 window의 여러
  방문이다. 끼워넣기를 없애 학습 group이 54,566개에서 14,018개(튜닝)로 줄었다.
  날짜의 19.7%는 연도를 추론한 값이다.
- 다음 실험: ① ranker 점수가 같으면 C5 순위로 정렬(지금은 식당 id), ② validation 선택 후보에
  R0(재정렬 안 함)를 넣어 R1이 R0보다 못하면 R0를 쓰는 규칙, ③ C5 순서에 없는 정보를 주는
  feature(리뷰 텍스트 유사도 등)와 window형 학습 query(여러 정답), ④ seed 3~5회 반복.

## 7. 재현

```bash
rating-recsys-experiment \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

같은 폴더의 `manifest.json`(조건·구간·환경), `metrics.json`(모든 수치), `queries_*.jsonl`·`recommendations_*.jsonl`(사용자별 정답과 Top-K), `model.txt`(최종 ranker)가 이 보고서의 원본이다.
