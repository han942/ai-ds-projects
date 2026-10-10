# Anthony Han — AI & Data Science Portfolio

[Projects Overview](./projects/README.md) · [Korean Project Index](./projects/README_KOR.md) · [Hackathon](./hackathon)

## Featured Projects

| Project | What I built | Stack | Result |
| --- | --- | --- | --- |
| **[Restaurant RecSys](./projects/rating_recsys)** | A two-stage restaurant recommender using crawled resturant reviews: candidate retrieval with LightGCN + cosine similarity;  followed by LightGBM LambdaRank reranking | Python, PyTorch, LightGBM, PostgreSQL, Supabase, MLflow, Streamlit | Leakage-free chronological evaluation over an immutable snapshot of **88,554 eligible interactions**; switching retrieval to item-item + LightGCN raised test Recall@100 from **16.7% to 20.2%**. The earlier review-text rating model is preserved under [Legacy V1 README](./projects/rating_recsys/legacy/v1_rating_prediction/README.md) |
| **[News Simplification for Youth](./projects/NLP_Newspaper_CUAI)** | A Korean text-style-transfer pipeline with a GPT-4o-augmented parallel corpus and QLoRA fine-tuning of **[Gemma 3-1B](https://huggingface.co/google/gemma-3-1b-it)** | Python, PyTorch, Hugging Face, TRL, QLoRA | Evaluated on **2,340** test sentences. Preserving numerical data perfectly,  translating hard chinese character based korean words and  * |
| **[AskIntern](./projects/Multi_Agent_ibm)** | A multi-agent workplace assistant whose supervisor routes lunch, IBM product-documentation, and meeting-note questions to specialist agents | IBM watsonx Orchestrate, Astra DB, Granite embeddings, MCP, Langfuse | Prototype demonstrates specialist routing, hybrid/RAG retrieval, role-based controls, safe abstention, and observable agent workflows |
| **[Global Supermarket Analysis](./projects/Global_Supermarket_Analysis)** | End-to-end retail EDA, loss analysis, SQL normalization, business reporting, and a self-contained interactive dashboard | Python, Pandas, NumPy, Seaborn, SciPy, MySQL, JavaScript | Analyzed **51,290** transactions and found that discounts above 40% occurred in **13.6%** of transactions but accounted for **68.2%** of total losses |

## Hackathon

### [Campus Mate](./hackathon)

**Codex Community Hackathon — Seoul for Students** · 2026.08.16 · Team 10 · [Korean](./hackathon/README_KOR.md)

A campus lunch-mate matcher that turns class timetables into a matching signal. Four teammates who met on the morning of the event took the project from feature definition to deployment and presentation in one day.

- Finds overlapping lunch-hour gaps and ranks eligible students using shared interests and natural-language preferences.
- Uses deterministic availability and conflict rules; AI only adjusts the order of already eligible matches.
- **Stack:** React, TypeScript, Node.js, Express, PostgreSQL, Supabase, Docker Compose, OpenAI API
- **Links:** [Live site](https://campusmate.site) · [Demo video](https://github.com/han942/codex-hackerthon/blob/main/campusmate_demo.mov) · [Source code](https://github.com/han942/codex-hackerthon) · [Event page](https://codex-community-korea.skysplit.chatgpt.site/en/hackathon/seoul-2026)

## Studies & Competitions

### RecSys Study (`Study/RecSys/`)

- **Goal:** Reimplement core recommendation models from scratch to understand their mechanics.
- **Stack:** Python, PyTorch, NumPy
- **Dataset:** MovieLens 100K (`datafile/Recsys/ml-100k`)
- **Implementations:**
  - `matrixfactorization/` — biased matrix factorization
  - `SVD/` — SVD-based collaborative filtering
  - `multvae/` — Mult-VAE for implicit feedback
  - `deepCONN/` — DeepCoNN review-text CNN towers for rating prediction

### Dacon Competitions (`Dacon/`)

- **Goal:** Practice tabular modelling under leaderboard metrics through feature engineering, imputation strategy, and model selection.
- **Stack:** Python, Pandas, Scikit-learn, XGBoost, LightGBM
- **Notebooks:**
  - `Toss_CTR_prediction/` — click-through-rate prediction
  - `부동산 허위매물 분류 해커톤/` — fake real-estate listing classification
  - `스트레스 지수 예측 해커톤/` — stress-index regression and imputation ablations
  - `전기차 가격 예측 해커톤/` — EV price prediction ([retrospective](./Dacon/전기차%20가격%20예측%20해커톤/회고록.md))
  - `스마트 창고 출고 지연 해커톤/` — warehouse shipping-delay prediction
  - `흡연 여부 예측 해커톤/` — smoking-status classification

## More Projects

- **[Target E-commerce Analysis](./projects/Target_Ecommerce)** — exploratory analysis of Brazilian e-commerce orders.

## Repository Map

| Folder | What's inside |
| --- | --- |
| [`projects/`](./projects) | Main end-to-end projects: EDA and reporting, LLM fine-tuning, recommender systems, and a multi-agent assistant |
| [`hackathon/`](./hackathon) | Codex Community Hackathon write-up for a one-day team build |
| [`Study/`](./Study) | From-scratch implementations of recommender-system papers |
| [`Dacon/`](./Dacon) | Dacon competition and hackathon notebooks and submissions |
| [`SQL/`](./SQL) | Database schemas and ERDs backing the projects |
| [`datafile/`](./datafile) | Shared raw datasets referenced by the notebooks above |

---

## Contact

Anthony Han — [@han942](https://github.com/han942)

Repository: <https://github.com/han942/ai-ds-projects>
