# rating_recsys 프로젝트 맥락

## 주제

- 식당 평점만으로는 개인의 식당 경험을 충분히 반영하기 어렵다.
- 이를 개선하기 위해 텍스트 리뷰 데이터를 결합한 추천 모델을 개발하고 실험한다.

## 과정

- 현재 baseline은 텍스트 없는 Two-stage 추천: item-item·LightGCN 후보 검색 + LambdaRank 재정렬이다. Two-Tower는 아직 적용하지 않았다.
- 전체 범위는 현재 baseline → 후보 모델 비교 → 리뷰 텍스트 결합 → Generative Retrieval → 리스트 생성형 추천·OneRec 통합 실험이다.
- 후보 비교에는 MF, Two-Tower, 그래프 추천, 순차 추천, 텍스트·하이브리드 검색을 포함한다. 상세 단계와 구현 상태는 `PLAN.md`를 따른다.
- 텍스트는 기존 후보 검색과 ranker feature에 각각 추가하고, Two-Tower 입력으로 쓰는 방식도 비교한다. OpenRouter 임베딩 API와 Supabase pgvector·Milvus·Astra DB를 검토 범위에 포함한다.
- 시간 분할은 평가 규칙, 순차 추천은 이력 순서·전이를 학습하는 방식, Two-stage는 검색·재정렬 구조, Two-Tower는 사용자·식당 encoder 구조다. 서로 다른 실험 축으로 구분한다.
- 같은 데이터·평가 조건에서 성능과 비용을 비교한다. 뒤 단계가 더 좋은 결과를 낸다고 미리 가정하지 않는다.

## 데이터

- Playwright를 통해 수집하고 Supabase에 적재한 데이터를 활용한다.
- 각 추천 시점 이후의 방문·평점·리뷰는 입력에서 제외한다. 임베딩 모델·차원·본문 hash·전처리 버전을 기록하고, cutoff별 프로필을 구성한다.

## 문서

- `docs_style.md`를 따른다. 현재 구현, 설계, 미실행 실험을 구분하고 핵심 설명·비교 표를 먼저 제시한다.
