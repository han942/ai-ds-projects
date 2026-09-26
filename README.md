# Data Science & AI Portfolio

[Projects Overview](./projects/README.md) · [Korean](./projects/README_KOR.md) · [Hackathon](./hackerthon)

## Abstract

A working repository for my data science / AI works, which span through end-to-end projects, team hackathons, and others.

Each flagship project ships with its own write-up covering the goal, the data,  modelling
decisions, and final results/accomplishments.

## 1. Projects

Flagship projects done through my academic - professional pathway. See [`projects/README.md`](./projects/README.md) for full write-ups.

- **[Restaurant Rating RecSys](./projects/rating_recsys)** — Various approaches trying to enhance the recommnedation quality by integrating user review data, based on Selenium-crawled DiningCode data.
- **[News Simplification for Youth](./projects/NLP_Newspaper_CUAI)** — Text-style-transfer
  via fine-tuned [Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it), with a
  GPT-4o-augmented parallel corpus. Retained **100%** factual accuracy on **75%+** of samples.
- Others: **[Global Supermarket Analysis](./projects/Global_Supermarket_Analysis)** | **[Target E-commerce Analysis](./projects/Target_Ecommerce)**

## 2. Hackathon

- **[Codex Community Hackathon — Seoul for Students](./hackerthon)** ([Korean](./hackerthon/README_KOR.md)) —
  A [one-day event](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026)
  where **teams are formed on site** and
  projects are built from scratch that day.
  -  **[Campus Mate](https://campusmate.site)**: a campus lunch-mate matcher that turns class
    timetables into a matching signal.
  - Tech stack: React, Node.js | Express, PostgreSQL, Supabase | Docker Compose

## 3. RecSys Study (`Study/RecSys/`)

- Goal: Reimplement core recommendation models from scratch to understand their mechanics.
- Tech stack: Python, PyTorch, NumPy
- Dataset: MovieLens 100K (`datafile/Recsys/ml-100k`)
- Implementations (each with `preprocessing.py` / model / `main.py`):
  - `matrixfactorization/` — biased matrix factorization
  - `SVD/` — SVD-based collaborative filtering
  - `multvae/` — Mult-VAE for implicit feedback
  - `deepCONN/` — DeepCoNN, review-text CNN towers for rating prediction

## 4. Competitions (`Dacon/`)

- Goal: Tabular modelling practice under a leaderboard metric — feature engineering,
  imputation strategy, and model selection.
- Tech stack: Python, Pandas, Scikit-learn, XGBoost, LightGBM
- Notebooks:
  - `Toss_CTR_prediction/` — click-through-rate (CTR) prediction
  - `부동산 허위매물 분류 해커톤/` — fake real-estate listing classification (LightGBM)
  - `스트레스 지수 예측 해커톤/` — stress-index regression (XGBoost, imputation ablations)
  - `전기차 가격 예측 해커톤/` — EV price prediction (+ [retrospective](./Dacon/전기차%20가격%20예측%20해커톤/회고록.md))
  - `스마트 창고 출고 지연 해커톤/` — warehouse shipping-delay prediction
  - `흡연 여부 예측 해커톤/` — smoking-status classification

## Repository Map

| Folder | What's inside |
| --- | --- |
| [`projects/`](./projects) | Main end-to-end projects (EDA & reporting, LLM fine-tuning, recommender systems) |
| [`hackerthon/`](./hackerthon) | Codex Community Hackathon write-up (team build, one day) |
| [`Study/`](./Study) | From-scratch implementations of recommender-system papers |
| [`Dacon/`](./Dacon) | Dacon competition / hackathon notebooks and submissions |
| [`SQL/`](./SQL) | Database schemas and ERDs backing the projects |
| [`datafile/`](./datafile) | Shared raw datasets referenced by the notebooks above |
---
## Contact

Anthony Han — [@han942](https://github.com/han942)

Repository: https://github.com/han942/ai-ds-projects
