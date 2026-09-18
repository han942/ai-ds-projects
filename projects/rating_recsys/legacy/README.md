# Legacy archive

## `v1_rating_prediction`

리뷰 텍스트와 평점을 이용한 Matrix Factorization 및 DeepCoNN 실험 버전이다.
아래 항목을 기존 구조 그대로 보존한다.

- 원본 README와 개발 기록
- 크롤링 및 MySQL 적재 notebook
- MF·DeepCoNN 분석 notebook
- 지역별 수집 CSV
- 학습 checkpoint와 prediction 결과
- 당시 사용한 dependency 및 환경 변수 예시

이 디렉터리는 v2의 historical reference다. 신규 구현은 상위 프로젝트 루트와
향후 생성될 `src/`, `migrations/`, `tests/`에서 진행한다.
