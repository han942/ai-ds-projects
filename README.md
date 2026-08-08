# Data Science & AI Portfolio

A working repository for my data science / AI work: end-to-end projects, ML competition
notebooks, recommender-system implementations, and SQL schema design.

- Projects overview: [English](./projects/README.md) · [Korean](./projects/README_KOR.md)

---

## Repository Map

| Folder | What's inside |
| --- | --- |
| [`projects/`](./projects) | Main end-to-end projects (EDA & reporting, LLM fine-tuning, recommender systems) |
| [`Dacon/`](./Dacon) | Dacon competition / hackathon notebooks and submissions |
| [`Study/`](./Study) | From-scratch implementations of recommender-system papers |
| [`SQL/`](./SQL) | Database schemas and ERDs backing the projects |
| [`datafile/`](./datafile) | Shared raw datasets referenced by the notebooks above |

---

## 1. Projects

Flagship work, each with its own README. See [`projects/README.md`](./projects/README.md) for full write-ups.

- **[Restaurant Rating RecSys](./projects/rating_recsys)** — Hybrid recommender combining
  review-text embeddings with collaborative filtering, on Selenium-crawled DiningCode data.
- **[News Simplification for Youth](./projects/NLP_Newspaper_CUAI)** — Text-style-transfer
  via fine-tuned [`Gemma 3-1B`](https://huggingface.co/google/gemma-3-1b-it), with a
  GPT-4o-augmented parallel corpus. Retained 100% factual accuracy on 75%+ of samples.
- **[Global Supermarket Analysis](./projects/Global_Supermarket_Analysis)** — EDA, loss
  analysis and a data-driven business report on global retail data
  (Pandas, Seaborn, MySQL).
- **[Target E-commerce Analysis](./projects/Target_Ecommerce)** — Multi-table order /
  customer / seller analysis on the Brazilian e-commerce dataset.

---

## 2. Competitions (`Dacon/`)

- Goal: Tabular modelling practice under a leaderboard metric — feature engineering,
  imputation strategy, and model selection.
- Tech stack: Python, Pandas, Scikit-learn, XGBoost, LightGBM
- Notebooks:
  - `Toss_CTR_prediction/` — click-through-rate prediction
  - `부동산 허위매물 분류 해커톤/` — fake real-estate listing classification (LightGBM)
  - `스트레스 지수 예측 해커톤/` — stress-index regression (XGBoost, imputation ablations)
  - `전기차 가격 예측 해커톤/` — EV price prediction (+ [retrospective](./Dacon/전기차%20가격%20예측%20해커톤/회고록.md))
  - `스마트 창고 출고 지연 해커톤/` — warehouse shipping-delay prediction
  - `흡연 여부 예측 해커톤/` — smoking-status classification

---

## 3. RecSys Study (`Study/RecSys/`)

- Goal: Reimplement core recommendation models from scratch to understand their mechanics.
- Tech stack: Python, PyTorch, NumPy
- Dataset: MovieLens 100K (`datafile/Recsys/ml-100k`)
- Implementations (each with `preprocessing.py` / model / `main.py`):
  - `matrixfactorization/` — biased matrix factorization
  - `SVD/` — SVD-based collaborative filtering
  - `multvae/` — Mult-VAE for implicit feedback
  - `deepCONN/` — DeepCoNN, review-text CNN towers for rating prediction

---

## 4. SQL (`SQL/`)

- Goal: Model the project datasets relationally instead of keeping them as flat CSVs.
- Tech stack: MySQL, MySQL Workbench, pymysql
- Contents:
  - `dining/` — schema, ERD and load scripts for the crawled restaurant data
  - `global_supermarket/` — schema for the supermarket retail data

---

## Setup

Notebooks that connect to MySQL or Hugging Face read their credentials from a `.env`
file at the repository root. Copy the template and fill in your own values:

```bash
cp .env.example .env
pip install python-dotenv
```

Notebooks read data via relative paths into `datafile/`, so run them from their own
directory.
