# V1 수집 원본 CSV

2026-10-07 사용자 요청으로 과거 V1 평점 예측 notebook, 모델·예측 결과, 전용 환경과 FastText 캐시를 삭제했다. V1 실행 코드는 더 이상 이 폴더에 없다.

[crawled_data/](./crawled_data/)의 CSV 5개만 데이터 적재 원본으로 유지한다. 현재 `rating_recsys.config`와 프로젝트 `.env.example`의 기본 적재 경로가 이 폴더를 참조한다.

현재 추천·E5 임베딩·LTR 실행은 [프로젝트 README](../../README.md)를 따른다. 로컬 산출물의 보존 범위는 [파일 안내](../../artifacts/README.md)에 있다.
