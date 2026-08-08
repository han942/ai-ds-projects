# AI/데이터 사이언스 프로젝트
---

## 1. 한국 식당을 위한 텍스트 임베딩 기반 하이브리드 추천 모델 구축

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- **목표:** 식당 추천 웹사이트에서 크롤링한 사용자 리뷰 데이터를 활용하여 추천 모델 구축
- **기술 스택:** Python, SQL, pymysql, PyTorch, Pandas, Selenium, Scikit-learn
- **주요 노트북/스크립트:** `diningcode_analysis.ipynb`
- **주요 내용:**
  - Selenium을 이용한 웹 크롤링 기법으로 실시간 데이터셋 구축
  - SQL 쿼리를 사용하여 크롤링된 데이터를 로컬 MySQL 서버에 저장 및 조회
  - 사용자 리뷰를 텍스트 임베딩하여 기존 추천 모델에 통합

---

## 2. LLM(대형 언어 모델) 파인튜닝을 통한 청소년 대상 뉴스 단순화

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- **목표:** 어린 학생들과 청소년들을 위해 어려운 뉴스 기사를 쉽게 단순화
- **기술 스택:** Python, LLM 파인튜닝, PyTorch
- **주요 노트북/스크립트:** `Final_NLP_Newspaper.ipynb`
- **주요 내용:**
  - LLM(GPT-4o) 기반 데이터 증강을 활용한 병렬 말뭉치(Parallel corpus) 구축
  - 주어진 TST(텍스트 스타일 변환, Text-Style-Transfer) 작업에 맞춰 [`Gemma 3-1B 모델`](https://huggingface.co/google/gemma-3-1b-it) 적용 및 파인튜닝
  - 구축된 모델이 원래의 정확도를 유지하면서 가독성을 향상시켰는지 확인하기 위한 모델 평가

---

## 3. Seaborn을 활용한 글로벌 슈퍼마켓 소매 데이터 시각화 및 비즈니스 보고서 작성

- **링크:** https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- **목표:** 글로벌 슈퍼마켓 판매 데이터에 대한 탐색적 데이터 분석(EDA) 및 시각화
- **기술 스택:** Python, Pandas, Matplotlib/Seaborn
- **주요 노트북/스크립트:** `supermarket_analysis.ipynb`, `Global_supermarket_Analysis.pdf`
- **주요 내용:**
  - 글로벌 판매 및 물류 데이터에 대한 종합적인 탐색적 데이터 분석(EDA)
  - 경제 산업에서 매우 중요한 특정 변수에 대한 손실(Loss) 분석
  - 데이터 기반의 비즈니스 전략 보고서 작성
