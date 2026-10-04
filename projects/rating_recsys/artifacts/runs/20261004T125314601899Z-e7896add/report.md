# 실험 보고서 · 20261004T125314601899Z-e7896add

- 실행: 2026-10-04 12:53 UTC · label `window-personal-satisfaction-n10`
- 데이터: snapshot `e7896add5b4b5939` · 88,554 interactions · 사용자 14,008 · 식당 4,587 · 2016-04-19 ~ 2026-09-26
- 코드: commit `53765758aa` (미커밋 변경 포함, `source.diff`)

## 1. 요약

- 후보 생성: C5 Recall@100 20.16% (C1 단독 17.81%, C4 LightGCN 단독 18.75%; 참고 C3 16.66%, C5 − C3 +3.50%p [+2.30, +4.70])
- LTR: NDCG@10 0.0266 → 0.0255 (-0.0011 [-0.0051, +0.0027]), Recall@10 4.03% → 3.65% (-0.38%p [-1.05, +0.27])
- Validation 선택: LambdaRank num_leaves 63 · min_child_samples 100 · trees 76. LightGCN은 고정 설정(3층 · 64차원 · L2 0.0001 · 20 epoch)

## 2. 데이터와 조건

| 구간 | 기간 | interactions | 평가 사용자 | 사용자당 정답 | 정답이 catalog에 있음 |
|---|---|---:|---:|---:|---:|
| Train | ~ 2025-12-19 | 70,883 | – | – | – |
| Validation | 2025-12-19 이후 ~ 2026-05-04 | 8,934 | 1,639 | 3.79 | 96.24% |
| Test | 2026-05-04 이후 | 8,737 | 1,573 | 3.57 | 98.08% |

평가 사용자는 cutoff 이전 이력이 있고 window에 평점 3.0 이상 방문이 있는 사용자다. 이력이 없는 사용자(validation 1,139명, test 1,214명)는 평가하지 않는다. Ranker 학습 정답은 튜닝 시 2025-12-19, 최종 재학습 시 2026-05-04까지다.


**개인별 만족도 기준:** 과거 이력 10건 미만이면 기존 경계를 유지한다. 10건 이상이면 강한 만족 경계 = 기존 경계 + clip(0.5 × (과거 사용자 평균 − 기존 경계), −0.5, +0.5). 기본 설정에서는 3.5~4.5점이며, 약한 만족 경계 3점은 고정한다. 강한 만족 label 2/gain 3, 약한 만족 label 1/gain 1, 나머지 label 0/gain 0이다. 평균은 각 query cutoff 이전 이력만 사용한다. 기존 등급의 과거 NDCG와 직접 비교하지 않는다.
- Validation: 개인 기준 975명, 이력 부족으로 절대 기준 적용 670명; 미래 방문의 label 0/1/2 수 {'0': 50, '1': 1705, '2': 4500}.
- Test: 개인 기준 862명, 이력 부족으로 절대 기준 적용 718명; 미래 방문의 label 0/1/2 수 {'0': 43, '1': 1376, '2': 4246}.
주요 조건: `candidate_k` 100, `ranking_k` 10, `region_mode` with_region, `relevance_high_threshold` 4.0, `relevance_low_threshold` 3.0, `train_fraction` 0.8, `validation_fraction` 0.1, `random_seed` 42, `ranker_training_mode` window, `ranker_label_mode` relevance, `rating_shrinkage_strength` 0.0, `satisfaction_mode` history-aware, `satisfaction_min_history` 10, `satisfaction_mean_weight` 0.5, `satisfaction_max_shift` 0.5

기본값과 다른 조건: `ranker_training_mode` window (기본 prefix), `satisfaction_mode` history-aware (기본 absolute)

## 3. 설정 선택

LTR 학습 query: `window`. 학습 label: `relevance`. 평가의 relevance 기준은 학습 label과 독립적으로 유지한다.
최종 window 학습: group 8,043개 중 여러 positive가 검색된 group 2,270개, 서로 다른 관측 label이 있는 group 733개, 실제 평점이 다른 관측 쌍 2,651개. 평균 검색 positive 1.42개/group.
각 학습 window 시작 전까지 후보와 feature를 고정한다. 마지막 부분 window는 cutoff까지 자르고, 최종 refit에서는 T2까지 전체 window를 재구성하여 중복 학습하지 않는다.

C4 LightGCN: 3층 · 64차원 · L2 0.0001 · learning rate 0.005 · batch 2048 · 20 epoch (config 고정값, 이 run에서 고르지 않음). Validation window 모델은 train 70,883 edges, test window 모델은 train + validation 79,817 edges로 학습했다.

Ranker 학습 query는 3개월 단위 구간 시작일 이전 interaction으로 다시 학습한 LightGCN을 쓴다. 모델 40개 (학습 234s), LightGCN 후보가 있었던 학습 query 40,753/40,753 (100.00%). 나머지는 그 시점 그래프에 없던 사용자라 C1 순서만 쓴다.

Ranker 학습 group은 정답이 그 query의 C5 후보 100개 안에 있는 query만 쓴다(정답을 후보에 끼워 넣지 않음): 정답 있는 학습 query 40,342개 중 15,166개 (37.59%).

LambdaRank: 기준은 validation R1 NDCG@10 최대 (동률이면 Recall@10, 그다음 grid 순서). `reference_untuned`은 early stopping 없이 고정 설정으로 학습한 비교 기준이다.

| 설정 | leaves | min_child | trees | Val NDCG@10 | Val Recall@10 |
|---|---:|---:|---:|---:|---:|
| reference_untuned | 15 | 10 | 150 | 0.0244 | 3.26% |
| leaves15_minchild10 | 15 | 10 | 4 | 0.0211 | 2.99% |
| leaves15_minchild100 | 15 | 100 | 4 | 0.0211 | 2.99% |
| leaves31_minchild10 | 31 | 10 | 3 | 0.0249 | 3.84% |
| leaves31_minchild100 | 31 | 100 | 3 | 0.0217 | 3.17% |
| leaves63_minchild10 | 63 | 10 | 55 | 0.0245 | 3.71% |
| **leaves63_minchild100** | 63 | 100 | 76 | 0.0277 | 4.17% |

최종 모델: 선택한 설정, 트리 수 = validation best iteration, train + validation의 window query로 재학습. Validation에서 C5 Recall@100 19.48%, R1 NDCG@10 0.0277.

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
| Recall@5 | 2.38% | 2.13% | – |
| Recall@10 | 4.03% | 3.65% | -0.38%p [-1.05, +0.27] |
| Precision@5 | 1.81% | 1.49% | – |
| Precision@10 | 1.44% | 1.29% | -0.15%p [-0.32, +0.01] |
| NDCG@5 | 0.0206 | 0.0200 | – |
| NDCG@10 | 0.0266 | 0.0255 | -0.0011 [-0.0051, +0.0027] |
| MAP@5 | 0.0126 | 0.0137 | – |
| MAP@10 | 0.0135 | 0.0145 | +0.0010 [-0.0019, +0.0039] |
| MRR@5 | 0.0403 | 0.0418 | – |
| MRR@10 | 0.0463 | 0.0479 | – |
| Catalog coverage@10 | 46.72% | 51.78% | – |
| Novelty@10 | 10.63 | 10.74 | – |
| 지역 다양성@10 | 21.06% | 19.99% | – |

NDCG@10 기준 R1이 나은 사용자 121명, 나쁜 사용자 118명, 같은 사용자 1,334명.

### 절대 평점 보조 진단 (Top-10)

4점 이상 Recall@10: 3.51% (4점 이상 미래 방문이 있는 사용자 1,510명에 대한 사용자별 평균).
실제 3점 미만 미래 방문 43개 중 Top-10에 포함된 방문 3개 (6.98%). 미관측 식당은 이 분모에 넣지 않는다.

## 6. 해석

- **기준 선택:** T1까지 train에서 사용자 내 pooled 표준편차 0.6495점을 확인했다. 최소 이력 10건에서 평균의 근사 오차 폭은 ±0.403점, 개인 기준 적용 가능한 train Window query는 42.8%다. 15건의 적용률 29.1%와 정밀도를 함께 보고 10건을 고정했다. Validation/test 성능으로 선택하지 않았다.
- **적용:** 과거 이력 10건 이상에서 강한 만족 경계 = clip(4 + 0.5 × (과거 사용자 평균 − 4), 3.5, 4.5), 약한 경계는 3점이다. Test 이력 사용자 1,580명 중 개인 기준 862명·절대 기준 718명이며, 절대 등급 대비 미래 방문 676행의 등급이 바뀌었다. 평균에는 미래 평점을 넣지 않는다.
- **학습:** 튜닝용 Window group 7,123개, 최종 refit group 8,043개(804,232 rows)다. 앞 절의 학습 query/group 합계는 튜닝과 최종 refit을 합친 처리량이며 최종 모델 group 수가 아니다. Validation에서 leaves 63 / min_child_samples 100 / trees 76을 선택했다.
- **결과:** Test R1 Graded NDCG@10 0.025466567, Recall@10 3.6502%, 4점 이상 Recall@10 3.5105%. C5 후보 Recall@100 20.1626%. 실제 3점 미만 미래 방문 43개 중 Top-10에 든 것은 3개(6.98%)다.
- **재정렬 효과:** 같은 새 정답의 R0 NDCG@10은 0.026590111이다. R1 − R0 = −0.001123544, paired bootstrap 95% CI [−0.005053415, +0.002693691]. Validation에서의 개선은 test에서 이어지지 않았으며, 이번 실행에서 재정렬의 이득을 확인하지 못했다. Test로 설정을 다시 선택하지 않았다.
- **검증:** 전체 테스트 114개 통과. 저장된 snapshot·query·추천·후보 순위로 validation/test label·평균·NDCG·Recall·저평점 진단을 독립 재계산해 일치를 확인했다. 모든 이력 사용자의 미래 방문을 보존했고 과거 방문 식당은 추천에서 제외했다. 원본은 `verification.json`, 계산 코드는 `verification_source.py`다.
- **한계:** 한 snapshot·seed 42·이미 사용한 미래 기간의 결과다. 과거 Prefix/절대 등급 NDCG와 직접 비교하지 않는다. 학습 3개월과 test 약 145일의 차이, 이력 없는 신규 사용자 제외, 노출 기록 부재와 소수 저평점 표본의 한계가 있다. 10건은 정밀도와 적용 범위의 절충안이며 통계적 최적값이 아니다.
- **범위:** 개인별 만족도 기준 구현과 Window baseline 전체 실행까지 완료했다. 후속 후보·텍스트 모델은 이번 요청에서 실행하지 않았다.

기준의 분석 수치는 `history_analysis.json`, 실행 당시 분석 코드는 `history_analysis_source.py`에 보존했다. 상세 설명과 예시는 [개인별 만족도 기준](../../../SATISFACTION_BASELINE.md)에 있다. 파이프라인 시간 합계는 1,070.3초(약 17.8분), prepared 배열은 약 45 MB다.

## 7. 재현

```bash
rating-recsys-experiment \
  --snapshot artifacts/snapshots/e7896add5b4b5939.jsonl \
  --ranker-training-mode window \
  --satisfaction-mode history-aware
```

같은 폴더의 `manifest.json`(조건·구간·환경), `metrics.json`(모든 수치), `queries_*.jsonl`·`recommendations_*.jsonl`(사용자별 정답과 Top-K), `target_diagnostics_*.jsonl`(정답 방문별 후보·최종 순위), `model.txt`(최종 ranker)가 이 보고서의 원본이다.
