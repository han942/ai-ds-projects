# 보관: Primary leave-last-two-out 평가 (2026-09-21 ~ 2026-09-27)

2026-09-27까지 사용한 첫 번째 평가 프로토콜의 기록이다. 현재 프로젝트의 평가는
전역 날짜 cutoff 방식 하나로 통일했고([BASELINE_MODEL.md](../../../BASELINE_MODEL.md)),
이 프로토콜의 코드는 현재 브랜치에서 제거했다. 결과 문서와 재현 방법만 여기에 남긴다.

## 1. 프로토콜

| 항목 | 정의 |
|---|---|
| 대상 사용자 | 고유 식당을 3곳 이상 방문한 사용자. 나머지는 학습·후보 통계에서도 제외 |
| 분할 | 사용자별 시간순. 마지막 방문 = test, 직전 방문 = validation, 나머지 = train |
| Query | 사용자 1명 × 정답 1개. 이력은 정답 방문 이전의 그 사용자 방문 |
| 학습 query | train 구간의 이력 prefix(→ 다음 방문). Final ranker는 train + validation query로 재학습 |
| 정답 relevance | 평점 4 이상 = 2, 3 이상 4 미만 = 1, 그 외 = 0. relevance 0인 query는 평균에서 제외 |
| 후보·feature | 각 query의 정답 시점 이전 interaction으로만 계산 |
| 지표 | 후보 Recall@20/50/100, 최종 Recall/NDCG/MRR@5·10, coverage, novelty, 지역 다양성 |
| 설정 선택 | 고정 설정 (C3 quota 0.5, LightGBM 150 trees·15 leaves). Validation은 보고용 |

## 2. 보관한 이유

1. Query당 정답이 1개라 MAP은 MRR과 같고, NDCG도 평점 등급 없이 정답의 순위만
   반영한다. Precision은 의미가 없다.
2. 사용자마다 자르는 위치가 달라 전역 시간축이 섞인다. Final ranker 학습 query 중
   일부는 다른 사용자의 test 방문보다 늦다. Feature는 cutoff-safe였지만 모델
   가중치 수준에서는 미래 패턴을 학습한다.
3. Validation을 설정 선택에 쓰지 않았다.
4. 고유 식당 3곳 미만 사용자(현재 데이터 8,074명)가 후보 통계에서도 빠졌다.

같은 스냅샷에서 두 프로토콜을 모두 실행했을 때 R1 재정렬 효과의 방향은 같았다.
Primary R1−R0 Recall@10 +1.41%p [95% CI +0.84, +1.96], 현재 프로토콜
+1.23%p [+0.50, +1.93]. 이 결과가 R1 효과가 누수 때문만은 아니라는 근거다.

## 3. 시도별 결과

수치는 모두 test. 스냅샷이 다른 행끼리는 직접 비교하지 않는다.

| 날짜 | 문서 | 스냅샷 (interactions) | 비교 내용 | 핵심 결과 |
|---|---|---|---|---|
| 09-21 | [후보 K 점검](./2026-09-21_candidate_k.md) | `03a76325` (23,017) | 평가 대상 범위, 후보 K | C0+C1 Recall@100 46.72%, R1 Recall@10 11.57% |
| 09-23 | [지역 제거](./2026-09-23_region_ablation.md) | `03a76325` (23,017) | C3(지역 포함) vs C0+C1 | 지역 제거 시 후보 Recall@100 −4.81%p, R1 Recall@10 −1.91%p, 둘 다 CI가 0 미만 → 지역 유지 |
| 09-23 | [LightGCN 후보](./2026-09-23_lightgcn_candidate.md) | `03a76325` (23,017) | C0+C1+LightGCN RRF | C0+C1 대비 Recall@100 +1.53%p, 현재 C3 대비 −3.28%p → C3 유지. Ranker 미학습 |
| 09-27 | [새 데이터 기준선](./2026-09-27_baseline.md) | `e7896add` (88,554) | C3 → R1 재측정 | C3 Recall@100 21.11%, R0→R1 Recall@10 4.00%→5.41% |

## 4. 재현

코드는 commit `ae3f4ea`에 그대로 있다. 현재 작업 트리를 건드리지 않도록 별도
worktree에서 실행한다.

```bash
git worktree add /tmp/rating_recsys_primary ae3f4ea
cd /tmp/rating_recsys_primary/projects/rating_recsys
conda create --prefix ./.venv python=3.10 pip libgomp -y && conda activate ./.venv
pip install -e '.[experiment,dev]'

# 전체 데이터 기준선 (디스크 기반 학습 행, MLflow 생략)
python -m rating_recsys.experiments.large_cli --snapshot <dataset.jsonl>
# 작은 스냅샷 + MLflow 기록
rating-recsys-experiment
# 지역 제거 비교, LightGCN 후보 비교 (DB를 직접 읽음)
python -m rating_recsys.experiments.compare_region
python -m rating_recsys.experiments.compare_lightgcn
```

`--snapshot`에는 `artifacts/archive/primary/runs/<run_id>/dataset.jsonl`을 넣으면
같은 입력이 된다. 끝나면 `git worktree remove /tmp/rating_recsys_primary`로 정리한다.

## 5. 현재 코드로 옮길 때

현재 파이프라인과 공유하는 부분(후보 생성 `retrieval/baselines.py`, feature
`ranking/features.py`, LightGBM 설정)은 그대로이므로, 분할과 지표만 되살리면 된다.
`ae3f4ea`에서 가져올 위치:

| 역할 | 파일 · 심볼 |
|---|---|
| 사용자별 분할 | `datasets/split.py` · `build_seen_user_split` |
| 평가 query | `experiments/queries.py` · `build_holdout_queries` |
| 단일 정답 지표 | `evaluation/metrics.py` · `evaluate_rankings`, `source_contribution` |
| 전체 흐름 | `experiments/pipeline.py` · `run_baseline_experiment`, 대용량 버전 `experiments/scalable.py` |
| LightGCN | `retrieval/lightgcn.py`, `experiments/compare_lightgcn.py` |

```bash
git show ae3f4ea:projects/rating_recsys/src/rating_recsys/datasets/split.py
```

새 결과를 이 프로토콜로 남긴다면 위 표에 행을 추가하고, 문서 첫 줄에 스냅샷 ID,
run ID, commit을 적는다.

## 6. Artifact 위치

로컬 전용이다(git에 올라가지 않음). [artifacts/README.md](../../../artifacts/README.md) 참고.

- `artifacts/archive/primary/runs/`: primary run 폴더. 문서가 가리키는 run은
  `20260923T024727601730Z`, `20260923T025110890068Z`, `20260927T134353560158Z`
- `artifacts/archive/primary/comparisons/`: 지역 제거·LightGCN 비교 결과
- MLflow experiment `archive · primary leave-last-two-out`: 당시 MLflow 기록
