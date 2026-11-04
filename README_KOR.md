# Anthony Han — AI·데이터 사이언스 포트폴리오

[English](./README.md) · [프로젝트 목록](./projects/README_KOR.md) · [해커톤](./hackathon/README_KOR.md)


## 주요 프로젝트

| 프로젝트 | 내용 | 기술 스텍 | 성과 |
| --- | --- | --- | --- |
| **[식당 추천 시스템](./projects/rating_recsys/README_KOR.md)** | 직접 수집한 다이닝코드 방문 기록·별점·리뷰를 활용한 2단계 추천 시스템. item-item과 LightGCN 후보를 RRF로 결합하고 LightGBM LambdaRank로 재정렬. BM25, E5, Two-Tower, RLMRec, RQ-VAE 등 리뷰 활용 방법을 행동 기반 기준 모델과 비교 | Python, PyTorch, LightGBM, PostgreSQL, Supabase, Playwright, MLflow, Streamlit | 사용자–식당 방문 기록 **88,554건**을 시간 순서로 나눠 평가. 행동 기반 기준 모델은 이후 방문한 정답 식당의 평균 **20.2%**를 후보 100개 안에 포함(test Recall@100). 리뷰를 활용한 방법의 일관된 추가 개선은 아직 확인하지 못함 |
| **[청소년을 위한 뉴스 쉬운 글 변환](./projects/NLP_Newspaper_CUAI/README_KOR.md)** | ChatGPT API로 원문·쉬운 문장 병렬 데이터를 생성하고, **[Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it)**를 QLoRA로 파인튜닝한 한국어 문체 변환 파이프라인 | Python, PyTorch, Hugging Face, TRL, QLoRA | 저장된 출력 **2,340건** 재평가: 한국어 형태소 SARI **58.80**(원문 복사 **46.46**), 드문 명사 비율 **7.3% → 4.1%**, 문장당 어절 수 **15.91 → 13.89**. 자동 지표에서 확인한 성과이며 사실 보존 검수는 별도로 필요 |
| **[AskIntern](./projects/Multi_Agent_ibm/README_KOR.md)** | supervisor가 점심 식당, IBM 제품 문서, 회의 노트 관련 질문을 전문 에이전트로 연결하는 업무 지원 어시스턴트 | IBM watsonx Orchestrate, Astra DB, Granite embeddings, MCP, Langfuse | 프로토타입에서 전문 에이전트 라우팅, 하이브리드 검색·RAG, 역할별 접근 제어, 근거가 없는 질문에 대한 답변 보류, 에이전트 실행 추적을 시연 |
| **[글로벌 슈퍼마켓 분석](./projects/Global_Supermarket_Analysis/README_KOR.md)** | 소매 데이터 탐색부터 손실 분석, MySQL 테이블 설계, 비즈니스 보고서와 독립 실행형 인터랙티브 대시보드 제작까지 구현 | Python, Pandas, NumPy, Seaborn, SciPy, MySQL, JavaScript | 주문 데이터 **51,290행** 분석. 할인율 40% 초과 거래가 전체의 **13.6%**이지만 총손실의 **68.2%**를 차지한다는 점을 확인 |

## 해커톤

### [Campus Mate](./hackathon/README_KOR.md)

**Codex Community Hackathon — Seoul for Students** · 2026.08.16 · Team 10

수업 시간표를 매칭 데이터로 활용해 함께 점심을 먹을 캠퍼스 친구를 찾는 서비스 MVP를 개발하였습니다.  

- 점심시간의 공강이 겹치는 학생을 찾고, 공통 관심사와 자연어로 입력한 선호를 반영해 후보 순서를 정합니다.
- 만남 가능 시간과 일정 충돌은 정해진 규칙으로 검사하며, AI는 조건을 통과한 후보의 순서만 조정합니다.
- **기술 스택:** React, TypeScript, Node.js, Express, PostgreSQL, Supabase, Docker Compose, OpenAI API
- **링크:** [서비스](https://campusmate.site) · [데모 영상](https://github.com/han942/codex-hackerthon/blob/main/campusmate_demo.mov) · [소스 코드](https://github.com/han942/codex-hackerthon) · [행사 페이지](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026)

## 스터디 및 경진대회

### 추천 시스템 스터디 (`Study/RecSys/`)

- **목표:** 핵심 추천 모델을 직접 구현하며 동작 원리를 이해합니다.
- **기술 스택:** Python, PyTorch, NumPy
- **데이터셋:** MovieLens 100K (`datafile/Recsys/ml-100k`)
- **구현 내용:**
  - `matrixfactorization/` — 편향항을 포함한 행렬 분해
  - `SVD/` — SVD 기반 협업 필터링
  - `multvae/` — 암묵적 피드백을 위한 Mult-VAE
  - `deepCONN/` — 리뷰 텍스트를 CNN으로 인코딩하는 DeepCoNN 평점 예측 모델

### 데이콘 경진대회 (`Dacon/`)

- **목표:** 피처 설계, 결측치 처리, 모델 선택을 통해 리더보드 평가 지표에 맞춘 정형 데이터 모델링을 연습합니다.
- **기술 스택:** Python, Pandas, Scikit-learn, XGBoost, LightGBM
- **노트북:**
  - `Toss_CTR_prediction/` — 클릭률 예측
  - `부동산 허위매물 분류 해커톤/` — 부동산 허위매물 분류
  - `스트레스 지수 예측 해커톤/` — 스트레스 지수 회귀 예측 및 결측치 처리 비교
  - `전기차 가격 예측 해커톤/` — 전기차 가격 예측([회고록](./Dacon/전기차%20가격%20예측%20해커톤/회고록.md))
  - `스마트 창고 출고 지연 해커톤/` — 창고 출고 지연 예측
  - `흡연 여부 예측 해커톤/` — 흡연 여부 분류

## 그 외 프로젝트

- **[Target 이커머스 분석](./projects/Target_Ecommerce)** — 브라질 이커머스 주문 데이터 탐색 분석

## 저장소 구성

| 폴더 | 내용 |
| --- | --- |
| [`projects/`](./projects) | 데이터 탐색·보고서, LLM 파인튜닝, 추천 시스템, 멀티 에이전트 등 주요 프로젝트 |
| [`hackathon/`](./hackathon) | 하루 동안 진행한 Codex Community Hackathon 팀 프로젝트 기록 |
| [`Study/`](./Study) | 추천 시스템 논문과 모델을 직접 구현한 스터디 |
| [`Dacon/`](./Dacon) | 데이콘 경진대회·해커톤 노트북 및 제출물 |
| [`SQL/`](./SQL) | 프로젝트에서 사용하는 데이터베이스 스키마와 ERD |
| [`datafile/`](./datafile) | 위 노트북에서 참조하는 공통 원본 데이터 |

---

## 연락처

Anthony Han — [@han942](https://github.com/han942)

저장소: <https://github.com/han942/ai-ds-projects>
