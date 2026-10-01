# 식당 평점 예측 v1 (legacy)

다이닝코드 평점 예측 실험을 보존한 디렉터리다. 현재 개발은
[v2 2-stage 추천 시스템](../../README.md)에서 진행한다. v1의 RMSE와 v2의 후보
Recall·추천 NDCG는 task와 평가 조건이 달라 직접 비교하지 않는다.

## 수정된 비교 실험

검토 후 재실행한 코드는 [diningcode_revision.ipynb](./diningcode_revision.ipynb),
결과는 [results/summary.md](./results/summary.md)에 있다. MF·DeepCoNN·텍스트+ID
Hybrid를 같은 데이터·분할·validation 선택 규칙으로 비교했다.

- 서울·경기 원본 19,297행 → 최초 사용자·식당 쌍 13,934행, 사용자 5,205명·식당 353곳.
- 사용자별 랜덤 80/20, seed 42. Train 11,841행 중 validation 2,392행을 분리한다.
  Test는 리뷰 2건 이상이며 train에 있는 식당을 평가한 사용자 1,557명·2,093행이다.
- 설정·epoch는 validation으로만 고르고 train 전체로 모델 seed 42·43·44를 재학습한다.
  예측은 [1, 5]로 clip한다.
- 문서에는 학습 이력 리뷰만 넣는다. 학습 행 자신의 리뷰는 사용자·식당 문서 양쪽에서
  빼고, 맛·가격·응대 평가를 입력에 넣지 않는다.

| 모델 | Test RMSE (seed 평균 ± 표준편차) | MF 대비 ΔRMSE [95% paired bootstrap CI] |
|---|---:|---:|
| Global mean | 0.7422 | +0.0615 [+0.0469, +0.0765] |
| User + item bias | 0.6772 | −0.0035 [−0.0068, −0.0002] |
| MF | 0.6807 ± 0.0003 | – |
| DeepCoNN | 0.6949 ± 0.0021 | +0.0142 [+0.0053, +0.0230] |
| Hybrid | 0.6946 ± 0.0016 | +0.0139 [+0.0053, +0.0222] |

이 조건에서는 DeepCoNN·Hybrid가 MF를 개선하지 못했다. 사용자별 랜덤 분할을 쓰는
v1 평점 비교이며, v2의 전역 날짜 기반 추천 평가와는 다르다. MAE·사용자 이력별
결과·선택 설정은 결과 요약에 있다.

## 이전 수치의 의미

원본 notebook의 기록은 MF RMSE 0.7098, DeepCoNN v1 0.8045, Improved 0.5749,
정규화 없는 Improved 0.5780이다. 기존 README의 “RMSE 19% 개선”은 이 기록에서
계산한 값이며 검증된 개선으로 사용하지 않는다. 원본 DeepCoNN은 test 리뷰로 test
문서를 만들었고, Improved는 정답 리뷰·세부 평가를 입력에 넣고 test RMSE로
checkpoint를 골랐다. 이전 설명과 달리 archive 구현에는 attention도 적용되지 않았다.
확인된 문제는 [archive/README.md](./archive/README.md)에 정리되어 있다.

기존 Precision@3·Recall@3은 사용자별 held-out 식당만 정렬한 결과라 후보 검색
성능을 판단하는 지표로 쓸 수 없다. 수정된 비교에서는 RMSE·MAE만 보고한다.

## 파일과 재현

| 경로 | 내용 |
|---|---|
| [diningcode_revision.ipynb](./diningcode_revision.ipynb) | 수정된 MF / DeepCoNN / Hybrid 비교 |
| [results/](./results/) | 요약, 지표, 예측, 입력 hash, 실행 설정 |
| [diningcode_analysis.ipynb](./diningcode_analysis.ipynb) | 원본 전처리·MF·DeepCoNN v1 |
| [rating_extraction.ipynb](./rating_extraction.ipynb) | 원본 Selenium / BeautifulSoup 크롤러 |
| [archive/](./archive/) | 이전 Improved·ablation notebook, checkpoint, 예측, MySQL 적재 코드 |
| [development.md](./development.md) | 당시 개발 기록 |
| [crawled_data/](./crawled_data/) | 지역별 CSV 입력 |
| [requirements.txt](./requirements.txt) | v1 의존성 |

이 디렉터리에서 Python 3.10 환경으로 수정된 notebook을 실행한다.

```bash
cd projects/rating_recsys/legacy/v1_rating_prediction
pip install -r requirements.txt
```

Okt에는 JDK가 필요하다. FastText 텍스트 vector를 `.cache/cc.ko.300.vec.gz`에 두거나
`FASTTEXT_PATH`를 `.vec` / `.vec.gz` 파일로 지정한다(`binary=False`로 로드).
원본 notebook의 `.bin` 입력과 달리 수정된 notebook은 텍스트 vector를 사용한다.
기록된 CPU 실행은 약
1.5시간이다. 의존성은 lock되어 있지 않으며 기록된 환경과 입력 hash는
[results/run_config.json](./results/run_config.json)에 있다. 원본 크롤러에는
Chrome/ChromeDriver가 추가로 필요하고, archive의 MySQL 적재 코드는 `.env`에서
접속 정보를 읽는다(`cp .env.example .env`). Archive notebook의 상대 경로는 원래
위치 기준이므로 그 폴더에서 재실행하려면 경로를 조정해야 한다.
[English documentation](./README.md).
