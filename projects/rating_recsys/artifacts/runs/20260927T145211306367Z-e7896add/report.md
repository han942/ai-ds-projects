# 실험 보고서 · 20260927T145211306367Z-e7896add

- 실행: 2026-09-27 14:52 UTC · label `baseline`
- 데이터: snapshot `e7896add5b4b5939` · 88,554 interactions · 사용자 14,008 · 식당 4,587 · 2016-04-19 ~ 2026-09-26
- 코드: commit `ae3f4eae2a` (미커밋 변경 포함, `source.diff`)

## 1. 요약

- 후보 생성: C3 Recall@100 16.66% (C1 단독 17.81%)
- LTR: NDCG@10 0.0174 → 0.0268 (+0.0094 [+0.0043, +0.0144]), Recall@10 2.54% → 3.78% (+1.23%p [+0.50, +1.93])
- Validation 선택: C3 quota 0.75, LambdaRank num_leaves 15 · min_child_samples 10 · trees 83

## 2. 데이터와 조건

| 구간 | 기간 | interactions | 평가 사용자 | 사용자당 정답 | 정답이 catalog에 있음 |
|---|---|---:|---:|---:|---:|
| Train | ~ 2025-12-19 | 70,883 | – | – | – |
| Validation | 2025-12-19 이후 ~ 2026-05-04 | 8,934 | 1,639 | 3.79 | 96.24% |
| Test | 2026-05-04 이후 | 8,737 | 1,573 | 3.57 | 98.08% |

평가 사용자는 cutoff 이전 이력이 있고 window에 평점 3.0 이상 방문이 있는 사용자다. 이력이 없는 사용자(validation 1,139명, test 1,214명)는 평가하지 않는다. Ranker 학습 정답은 튜닝 시 2025-12-19, 최종 재학습 시 2026-05-04까지다.

주요 조건: `candidate_k` 100, `ranking_k` 10, `region_mode` with_region, `relevance_high_threshold` 4.0, `relevance_low_threshold` 3.0, `train_fraction` 0.8, `validation_fraction` 0.1, `random_seed` 42

기본값과 다른 조건: 없음

## 3. Validation에서 고른 설정

후보 quota: C3에서 C0+C1 순서로 먼저 채우는 후보 비율. 기준은 validation C3 Recall@100 최대 (동률이면 NDCG@100, 그다음 grid 순서).

| quota | Val C3 Recall@100 |
|---|---:|
| 0.0 | 14.51% |
| 0.25 | 14.51% |
| 0.5 | 14.59% |
| **0.75** | 15.40% |
| 1.0 | 15.07% |

LambdaRank: 기준은 validation R1 NDCG@10 최대 (동률이면 Recall@10, 그다음 grid 순서). `reference_untuned`은 early stopping 없이 고정 설정으로 학습한 비교 기준이다.

| 설정 | leaves | min_child | trees | Val NDCG@10 | Val Recall@10 |
|---|---:|---:|---:|---:|---:|
| reference_untuned | 15 | 10 | 150 | 0.0249 | 3.35% |
| **leaves15_minchild10** | 15 | 10 | 83 | 0.0270 | 3.71% |
| leaves15_minchild100 | 15 | 100 | 48 | 0.0252 | 3.30% |
| leaves31_minchild10 | 31 | 10 | 55 | 0.0261 | 3.52% |
| leaves31_minchild100 | 31 | 100 | 69 | 0.0257 | 3.31% |
| leaves63_minchild10 | 63 | 10 | 49 | 0.0255 | 3.22% |
| leaves63_minchild100 | 63 | 100 | 86 | 0.0260 | 3.48% |

최종 모델: 선택한 설정, 트리 수 = validation best iteration, train + validation window prefix query로 재학습. Validation에서 선택한 구성의 C3 Recall@100 15.40%, R1 NDCG@10 0.0270.

## 4. 후보 생성 (test)

Recall@K = 사용자별 (후보 K개 안의 정답 수 / 전체 정답 수)의 평균.

| 후보 | Recall@20 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| C0 전체 인기 | 2.67% | 5.56% | 9.04% |
| C1 item-item | 5.83% | 11.35% | 17.81% |
| C2 지역 인기 | 4.12% | 7.90% | 14.17% |
| C3 quota RRF | 4.55% | 9.85% | 16.66% |

## 5. LTR 재정렬 (test, Top-10)

R0는 C3 후보 순서를 그대로 자른 목록, R1은 LambdaRank로 재정렬한 목록이다. CI는 같은 사용자끼리 짝지은 paired bootstrap 2,000회.

| 지표 | R0 C3 순서 | R1 LambdaRank | R1 − R0 (95% CI) |
|---|---:|---:|---|
| Recall@5 | 1.26% | 2.33% | – |
| Recall@10 | 2.54% | 3.78% | +1.23%p [+0.50, +1.93] |
| Precision@5 | 1.02% | 1.65% | – |
| Precision@10 | 1.02% | 1.35% | +0.33%p [+0.14, +0.52] |
| NDCG@5 | 0.0127 | 0.0223 | – |
| NDCG@10 | 0.0174 | 0.0268 | +0.0094 [+0.0043, +0.0144] |
| MAP@5 | 0.0078 | 0.0145 | – |
| MAP@10 | 0.0087 | 0.0149 | +0.0061 [+0.0023, +0.0101] |
| MRR@5 | 0.0247 | 0.0368 | – |
| MRR@10 | 0.0303 | 0.0426 | – |
| Catalog coverage@10 | 33.85% | 63.28% | – |
| Novelty@10 | 9.75 | 11.34 | – |
| 지역 다양성@10 | 47.64% | 18.53% | – |

NDCG@10 기준 R1이 나은 사용자 154명, 나쁜 사용자 93명, 같은 사용자 1,326명.

## 6. 해석 (작성자 기입)

- 결론: 재정렬(R1)은 test에서 모든 정확도 지표를 올렸고 CI가 모두 0보다 크다.
  병목은 후보 생성이다. C3 Recall@100이 16.66%라 R1이 찾을 수 있는 정답이 여기서
  이미 제한된다.
- 근거: NDCG@10 +0.0094 [+0.0043, +0.0144], Recall@10 +1.23%p [+0.50, +1.93],
  MAP@10 +0.0061 [+0.0023, +0.0101]. Ranker는 test 이후는 물론 test window 데이터도
  보지 않았다(학습 정답 ≤ 2026-05-04). 후보 결합(C3)이 C1 단독보다 Recall@100이
  1.15%p 낮아, 지금의 quota 결합은 C1의 좋은 후보를 C0·C2로 밀어낸다.
- 한계: R1은 사용자의 주 지역 식당을 올려 목록 안의 지역 다양성이 47.6%에서
  18.5%로 줄었다(coverage는 33.9%→63.3%). Validation 선택 차이(NDCG 0.0270 vs
  고정 설정 0.0249)에는 CI가 없고 seed 1회다. 학습 positive의 70%(튜닝)·77%(재학습)가
  injection이다. 날짜의 19.7%는 연도를 추론한 값이다.
- 다음 실험: ① C1+C2(C0 제외) 결합과 quota grid 확장, ② `--candidate-k` 200/300,
  ③ injection 없는 학습과의 비교, ④ seed 3~5회 반복.

## 7. 재현

```bash
rating-recsys-experiment \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl
```

같은 폴더의 `manifest.json`(조건·구간·환경), `metrics.json`(모든 수치), `queries_*.jsonl`·`recommendations_*.jsonl`(사용자별 정답과 Top-K), `model.txt`(최종 ranker)가 이 보고서의 원본이다.
