# v1 재실험 결과: MF · DeepCoNN · Hybrid

`diningcode_revision.ipynb` 실행 시 자동 생성된다. 수정 내용은 [`../REVISION_PLAN.md`](../REVISION_PLAN.md).

## 조건

- 데이터: 서울 · 경기 크롤링 19,297행 → 중복 제거 13,934행 (사용자 5,205명, 식당 353곳, sparsity 99.24%)
- Split: 사용자별 80/20 (seed 42). Train 11,841 = inner 9,449 + validation 2,392. Test 2,093행 / 1,557명 (리뷰 2건 이상 사용자, train에 있는 식당)
- 선택은 validation으로만, test는 train 전체로 재학습한 뒤 1회 평가. 모든 예측은 [1, 5] clip
- 모델 seed [42, 43, 44]: 평균 ± 표준편차. 차이는 MF 대비 RMSE, paired bootstrap 95% 구간 (음수 = MF보다 낮음)
- 텍스트 문서: 학습 이력 리뷰로만 구성, 학습 행은 자기 리뷰 제외 (leave-one-out). 맛 · 가격 · 응대 미사용

## 비교

| 모델 | Validation RMSE | Test RMSE | Test MAE | vs MF ΔRMSE [95% CI] |
|---|---:|---:|---:|---:|
| Global mean | 0.7246 | 0.7422 | 0.6230 | +0.0615 [+0.0469, +0.0765] |
| User+item bias | 0.6769 | 0.6772 | 0.5244 | -0.0035 [-0.0068, -0.0002] |
| MF | 0.6806 | 0.6807 ± 0.0003 | 0.5172 ± 0.0002 | – |
| DeepCoNN | 0.6987 | 0.6949 ± 0.0021 | 0.5509 ± 0.0033 | +0.0142 [+0.0053, +0.0230] |
| Hybrid | 0.6985 | 0.6946 ± 0.0016 | 0.5502 ± 0.0044 | +0.0139 [+0.0053, +0.0222] |

선택된 설정

- User+item bias: reg_user 3, reg_item 25
- MF: k 10, lr 0.01, reg 0.1, 28 epochs
- DeepCoNN: lr 0.0003, 4 epochs
- Hybrid: lr 0.0003, 4 epochs, ID k 10, ID reg 0.1

## 사용자 train 이력 수별 Test RMSE (seed 평균 예측)

| 모델 | 1-2 | 3-9 | 10+ |
|---|---:|---:|---:|
| Global mean | 0.7468 | 0.7361 | 0.7526 |
| User+item bias | 0.6986 | 0.6789 | 0.6493 |
| MF | 0.6967 | 0.6838 | 0.6528 |
| DeepCoNN | 0.7118 | 0.6954 | 0.6615 |
| Hybrid | 0.7120 | 0.6947 | 0.6610 |
| 행 수 | 483 | 1151 | 459 |

## 이전 기록과의 관계

기존 README 수치(MF 0.7098, DeepCoNN v1 0.8045, Improved 0.5749)는 split · 입력 문서 · 선택 규칙이 달라 위 표와 비교하지 않는다. DeepCoNN v1은 test 리뷰로 test 문서를 만들었고, Improved는 정답 리뷰가 입력 문서와 side feature에 들어 있었으며 test 기준으로 checkpoint를 골랐다. 원본 MF 설정 점검 결과는 `v1_mf_audit_metrics.json`에 있다 (고정 설정 test RMSE 0.7088, validation 튜닝 시 0.6848, user+item bias 0.6745).

## 파일

| 파일 | 내용 |
|---|---|
| `summary.md` | 이 문서 |
| `metrics.json` | 데이터 · split 통계, 비교표, 이력 구간별 RMSE, 선택 과정 (MF grid 곡선, DeepCoNN · Hybrid epoch별 기록) |
| `predictions_test.csv` | Test 행별 실제 평점과 모델 · seed별 예측 |
| `run_config.json` | 입력 파일 SHA-256, seed, 하이퍼파라미터, 실행 환경 |
| `v1_mf_audit_metrics.json` | 원본 MF 측정 점검 결과 (matfac 동작, 튜닝 부족, baseline) |
