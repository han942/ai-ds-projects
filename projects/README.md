# AI/Data Science Projects

This folder contains my main data science and AI projects. Each project folder has its own README with details, results, and usage.

[Korean](./README_KOR.md)

## 1. Multi-Stage Restaurant Recommendation System

- Goal: Test whether user-review text helps recommend restaurants that users will enjoy, compared with a model that uses only visits and ratings.
- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/rating_recsys
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=microsoft&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white) ![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white) ![Playwright](https://img.shields.io/badge/Playwright-2EAD33?style=flat&logo=playwright&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
- Main code: `src/rating_recsys/` (run `rating-recsys-experiment`)
- Highlights:
  - Crawled DiningCode reviews with Playwright and loaded them into Supabase PostgreSQL with deduplication
  - Two-stage recommender: item-item CF and LightGCN candidates fused with RRF, then LightGBM LambdaRank reranking, evaluated with a time-based split (test Recall@100 20.2%)
  - Added review text to retrieval, LightGCN training, and reranking (BM25, E5, Two-Tower, RLMRec, RQ-VAE); none has yet beaten the behavioral baseline consistently

---

## 2. News Simplification for Youth via LLM Fine-Tuning

- Goal: Rewrite hard news articles into language that upper-elementary and middle-school students can read.
- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/NLP_Newspaper_CUAI
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black) ![LLM Fine-Tuning](https://img.shields.io/badge/LLM%20Fine--Tuning-412991?style=flat&logo=openai&logoColor=white)
- Main notebook: `Final_NLP_Newspaper.ipynb`
- Highlights:
  - Built a 15,894-pair parallel corpus with the ChatGPT API
  - Fine-tuned [Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it) with QLoRA for the text-style-transfer (TST) task
  - Re-evaluated 2,340 saved outputs: Korean-morpheme SARI 58.80 versus source-copy 46.46; rare-noun ratio 7.3% → 4.1% and words per sentence 15.91 → 13.89
  - Mean numeric-value recall 87.5% and proper-name recall 91.6% under automatic detection; factual review remains necessary ([report](./NLP_Newspaper_CUAI/evaluation/REPORT.md))

---

## 3. Multi-AI Agent System for Enterprise Applications (AskIntern)

- Goal: Answer interns' recurring lunch, IBM product, and meeting-note questions through one web chat.
- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/Multi_Agent_ibm
- Tech stack: ![IBM watsonx Orchestrate](https://img.shields.io/badge/IBM%20watsonx%20Orchestrate-052FAD?style=flat&logo=ibm&logoColor=white) ![Astra DB](https://img.shields.io/badge/Astra%20DB-5C2D91?style=flat&logo=datastax&logoColor=white) ![MCP](https://img.shields.io/badge/MCP-000000?style=flat&logoColor=white) ![Langfuse](https://img.shields.io/badge/Langfuse-0A0A0A?style=flat&logoColor=white)
- Main files: `README.md`, `AskIntern_presentation.pdf`, `demo_askintern_zoom.mp4`
- Highlights:
  - A supervisor agent routes each request to one of three specialist agents: lunch, IBM product documentation, and meeting notes
  - Evaluation of the IBM product agent: 6 of 7 cases passed, with citation accuracy 0.86
  - Reduced document loss during concurrent writes from 51% to 0% with deterministic IDs, ordered writes, and retries

---

## 4. Global Supermarket Retail Loss & Business Dashboard

- Goal: Find where a global supermarket loses profit and recommend pricing changes to reduce losses.
- Link: https://github.com/han942/ai-ds-projects/tree/main/projects/Global_Supermarket_Analysis
- Tech stack: ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat&logo=python&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat&logo=mysql&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
- Main files: `global_supermarket.ipynb`, `Global_Supermarket_Analysis.pdf`, `dashboard.html`
- Highlights:
  - EDA of 51,290 order lines across seven markets from 2019 to 2022
  - Found that discounts above 40% make up 13.6% of transactions but 68.2% of total losses
  - Wrote a business report with pricing recommendations (English and Korean PDF) and built a self-contained interactive dashboard
