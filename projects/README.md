# AI/Data Science Projects

This folder contains my main data science/AI projects, organized by topic.
- [Korean](./README_KOR.md)

## 1. Constructing text-embedded hybrid recommendation models for korean dining resturants   

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- Goal: Constructing recommnedation model that represents user-reviews, leading to high quality recommendation in restuarant domains.
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) ![Playwright](https://img.shields.io/badge/Playwright-2EAD33?style=flat&logo=playwright&logoColor=white) ![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat&logo=selenium&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white) 
- Main code: `src/rating_recsys/` (run `rating-recsys-experiment`); v1 notebooks and archived results are documented in the [Legacy V1 README](./rating_recsys/legacy/v1_rating_prediction/README.md)
- Highlights:
  - Crawling DiningCode reviews with Selenium/Playwright and loading them into Supabase PostgreSQL with deduplication
  - Two-stage recommender: item-item CF + LightGCN candidates fused with RRF, then LightGBM LambdaRank reranking, evaluated on a leakage-free global date split
  - Review-text DeepCoNN (CNN towers over past reviews) compared as a candidate source under the same protocol

---

## 2. News Simplification for Youth via Fine-Tuning LLMs (Large-Language Models) 

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- Goal: Translating hard news articles for young students and teenagers.
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![LLM Fine-Tuning](https://img.shields.io/badge/LLM%20Fine--Tuning-412991?style=flat&logo=openai&logoColor=white)
- Main notebook/script: `Final_NLP_Newspaper.ipynb`
- Highlights:
  - Constructing parallel corpus using LLM(GPT-4o) based data augmentation
  - Adapting and fine-tuning [Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it) for the given TST(Text-Style-Transfer) task 
  - Validating model's ability to maintain 100% of its factual accuracy in **75%+** of the original data, also improving readability for younger audiences.

---

## 3. Data visualization and composing business report for global supermarket retail data with Seaborn

- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- Goal: Conducting exploratory data analysis (EDA) and loss analysis to develop strategies to improve loss status.
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
- Main notebook/script: `supermarket_analysis.ipynb`, `Global_supermarket_Analysis.pdf`
- Highlights:
  - Comprehensive EDA on global sales and logistics data
  - Loss analysis on specific features that is crucial in economic industry
  - Composing data-driven business strategy report

---
