# archive

2026-09-27에 MF · DeepCoNN · Hybrid 비교의 기준 notebook을 [`../diningcode_analysis.ipynb`](../diningcode_analysis.ipynb)로 정하면서 옮긴 파일이다. 기록 보존용이며 수정하지 않는다. Notebook 안의 상대 경로(`crawled_data/`, `../..`)는 원래 위치 기준이라 이 폴더에서 그대로 재실행되지 않는다.

| 파일 | 내용 |
|---|---|
| `diningcode_analysis_improved.ipynb` | DeepCoNN Improved (multi-scale CNN, user/item bias, 맛·가격·응대 입력, 정규화). 기존 README 대표 수치 RMSE 0.5749 |
| `diningcode_no_norm.ipynb` | Improved에서 정규화·clamp를 뺀 ablation (0.5780) |
| `demo_model.ipynb` | 초기 프로토타입 (공백 토큰화, Python 3.8) |
| `best_model.pt` | Improved의 checkpoint (test RMSE 기준 선택) |
| `predict_result.csv` | Improved의 test 예측 2,096행 |

알려진 문제: 정답 리뷰가 입력 문서와 side feature에 포함, test 기준 checkpoint 선택, attention 미사용, 응대·가격 인코딩 순서. 수정한 비교 실험은 [`../diningcode_revision.ipynb`](../diningcode_revision.ipynb), 결과는 [`../results/`](../results)에 있다.
