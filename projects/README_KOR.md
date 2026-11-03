# AI/데이터 사이언스 프로젝트

주요 데이터 사이언스 및 AI 프로젝트를 모아 둠. 프로젝트별 자세한 내용과 실행 방법은 각 폴더의 README에 있음.

[English](./README.md)

## 1. 방문·평점 이력 기반 Multi-Stage 맛집 추천 시스템

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- **목표:** 직접 수집한 방문 기록과 별점, 리뷰로 사용자가 만족할 식당을 더 잘 추천하는 것
- **기술 스택:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=microsoft&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) ![Playwright](https://img.shields.io/badge/Playwright-2EAD33?style=flat&logo=playwright&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
- **주요 코드:** `src/rating_recsys/` (`rating-recsys-experiment`로 실행)
- **주요 내용:**
  - Playwright로 다이닝코드 리뷰 96,922건을 수집하고, 중복을 걸러 Supabase PostgreSQL에 저장함
  - item-item CF와 LightGCN으로 후보를 찾고 LightGBM LambdaRank로 순서를 정하는 2단계 추천 모델을 구축함. 시간 순서로 나눈 test에서 Recall@100 20.2%를 기록함
  - 리뷰 텍스트를 후보 검색, LightGCN 학습, 재정렬에 적용해 비교함(BM25, E5, Two-Tower, RLMRec, RQ-VAE)

---

## 2. LLM Fine-tuning을 활용한 청소년 맞춤 뉴스 번역

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- **목표:** 어려운 뉴스 기사를 초등학교 고학년부터 중학생까지 쉽게 읽을 수 있는 글로 바꾸는 것
- **기술 스택:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![LLM 파인튜닝](https://img.shields.io/badge/LLM%20Fine--Tuning-412991?style=flat&logo=openai&logoColor=white)
- **주요 노트북:** `Final_NLP_Newspaper.ipynb`
- **주요 내용:**
  - 국립국어원 말뭉치를 바탕으로 ChatGPT API를 이용해 원문과 쉬운 문장을 짝지은 병렬 말뭉치 15,894쌍을 구축함
  - [Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it)를 QLoRA로 파인튜닝함
  - test 문장 2,340개에서 SARI 57.02, FKGL 1.28을 기록함(사람이 쓴 정답 문장의 FKGL은 1.34)

---

## 3. Multi-Agent 시스템 (AskIntern)

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/Multi_Agent_ibm
- **목표:** IBM 인턴들이 자주 묻는 점심 식당, IBM 제품, 회의 노트 관련 질문을 하나의 웹 채팅에서 해결하는 것
- **기술 스택:** ![IBM watsonx Orchestrate](https://img.shields.io/badge/IBM%20watsonx%20Orchestrate-052FAD?style=flat&logo=ibm&logoColor=white) ![Astra DB](https://img.shields.io/badge/Astra%20DB-5C2D91?style=flat&logo=datastax&logoColor=white) ![MCP](https://img.shields.io/badge/MCP-000000?style=flat&logoColor=white) ![Langfuse](https://img.shields.io/badge/Langfuse-0A0A0A?style=flat&logoColor=white)
- **주요 파일:** `README_KOR.md`, `AskIntern_presentation.pdf`, `demo_askintern_zoom.mp4`
- **주요 내용:**
  - supervisor 에이전트가 질문을 점심, IBM 제품 문서, 회의 노트 담당 에이전트 중 하나로 연결함
  - IBM 제품 에이전트 평가에서 7개 사례 중 6개를 통과하고, 인용 정확도 0.86을 기록함
  - 고정 ID, 쓰기 순서 지정, 재시도를 적용해 동시 쓰기 중 문서 유실을 51%에서 0%로 줄임

---

## 4. 글로벌 소매 데이터 기반 비즈니스 성과 분석 및 개선 전략을 담은 대시보드 제작

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- **목표:** 글로벌 슈퍼마켓의 손실 원인을 찾고, 손실을 줄일 할인 정책과 상품 운영 전략을 제안하는 것
- **기술 스택:** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
- **주요 파일:** `global_supermarket.ipynb`, `Global_Supermarket_Analysis_korean.pdf`, `dashboard.html`
- **주요 내용:**
  - 2019–2022년 7개 시장의 주문 데이터 51,290행을 고객, 상품, 시장, 주문, 배송 관점에서 분석함
  - 할인율 40%를 넘는 거래가 전체의 13.6%에 불과하지만 전체 손실의 68.2%를 차지한다는 점을 발견함
  - 개선 전략을 담은 보고서(영문, 국문 PDF)와 브라우저에서 바로 열리는 인터랙티브 대시보드를 제작함
