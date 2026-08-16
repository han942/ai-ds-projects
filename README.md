# Data Science & AI Portfolio

[Projects Overview](./projects/README.md) · [Korean](./projects/README_KOR.md) · [Hackathon](./hackerthon)

## About The Project

A working repository for my data science / AI work, spanning end-to-end projects, a team
hackathon build, from-scratch paper reimplementations, and competition notebooks.

Each flagship project ships with its own write-up covering the goal, the data, the modelling
decisions, and the results — including the ones that did not improve the metric.

### Built With

- **Languages** — Python, SQL, TypeScript
- **ML / DL** — PyTorch, Scikit-learn, XGBoost, LightGBM, Transformers
- **Data** — Pandas, NumPy, SciPy, Selenium, BeautifulSoup
- **Visualization** — Matplotlib, Seaborn
- **Storage** — MySQL, PostgreSQL
- **Ops** — MLflow, Docker Compose

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

## 2. Hackathon

- **[Codex Community Hackathon — Seoul for Students](./hackerthon)** ([Korean](./hackerthon/README_KOR.md)) —
  A [one-day event](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026)
  (2026.08.16, 100 university students · 25 teams) where **teams are formed on site** and
  projects are built from scratch that day. Team 10 shipped
  **[Campus Mate](https://campusmate.site)**: a campus lunch-mate matcher that turns class
  timetables into a matching signal.
- Tech stack: React 19, Node.js 22 / Express 5, PostgreSQL 17, Supabase, OpenAI API, Docker Compose
- My role: Backend B (Social Flow) — match / venue / proposal routes and the chat-based AI matching API
- The write-up covers the goal, stack, local run, and hour-by-hour timeline — including how a
  spec-first contract and a `CoreQueryPort` interface seam let four people who had just met
  write code in parallel without collisions.

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

## 4. Competitions (`Dacon/`)

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

## Getting Started

### Prerequisites

- Python 3.10+ with Jupyter
- MySQL, for the notebooks that load into a relational schema

### Setup

Notebooks that connect to MySQL or Hugging Face read their credentials from a `.env`
file at the repository root. Copy the template and fill in your own values:

```bash
cp .env.example .env
pip install python-dotenv
```

Notebooks read data via relative paths into `datafile/`, so run them from their own
directory.

## Contact

Seung-Won Han — [@han942](https://github.com/han942)

Repository: https://github.com/han942/ai-ds-projects
