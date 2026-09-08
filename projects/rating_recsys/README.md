# Text-Embedded Hybrid Recommender for Korean Restaurants

> Review text → rating prediction, with **19% lower RMSE** than a rating-only
> baseline

Personal project · 2025.12.01–ongoing · [Korean](./README_KOR.md)

This project combines DiningCode review text with collaborative and structured
signals to predict restaurant ratings. The model is based on
[DeepCoNN (WSDM '17)](https://arxiv.org/pdf/1701.04783) and extends its dual-CNN
design with side features, user/item biases, and a Factorization Machine head.

## 1. Goal

Rating-only recommenders know **what** score a user gave, but not **why**. That
limitation is especially costly in this dataset, where the user–item matrix is
**99.24% sparse** and collaborative signals alone are weak.

The goal is to improve rating prediction by learning from three signal types:

- the language used across all reviews written by a user;
- the language used across all reviews received by a restaurant; and
- structured context such as taste, price, service, and rating tendencies.

Aggregating reviews into user and item documents gives even low-activity users
a dense text representation. Matrix Factorization serves as the rating-only
baseline, while the text-embedded model tests whether review content adds useful
predictive information.

## 2. Architecture

```mermaid
flowchart LR
    A["DiningCode<br/>regional foodrank pages"] --> B["Selenium + BeautifulSoup<br/>crawl and recover"]
    B --> C["CSV + MySQL<br/>regional storage"]
    C --> D["Clean and deduplicate<br/>19,297 → 13,944 rows"]
    D --> E["Okt + fastText<br/>user/item documents"]

    E --> U["User CNN tower<br/>kernels 2, 3, 4 + attention"]
    E --> I["Item CNN tower<br/>kernels 2, 3, 4 + attention"]
    U --> F["Feature fusion<br/>text + side features + biases"]
    I --> F
    F --> G["Factorization Machine head"]
    G --> H["Rating prediction"]
    H --> M["MLflow<br/>metrics · params · checkpoints"]

    classDef data fill:#e8f0fe,stroke:#4a6da7,color:#1f2328
    classDef model fill:#fdf0e3,stroke:#c98b3a,color:#1f2328
    classDef result fill:#e9f5ec,stroke:#4a8a5f,color:#1f2328
    class A,B,C,D,E data
    class U,I,F,G model
    class H,M result
```

### Data pipeline

- **Source:** DiningCode `foodrank` pages for Seoul, Gyeonggi, Busan, Daegu,
  and an earlier nationwide crawl.
- **Collection:** Selenium and BeautifulSoup collect ratings, review text,
  menus, user metadata, and taste/price/service labels. Popup interception,
  tab handling, retries, and “더보기” pagination make the crawler resilient.
- **Cleaning:** duplicate review blocks caused by Selenium timing are removed
  (**27.7% of collected rows**). Ratings and Korean category labels are parsed,
  badges are separated from user IDs, and truncated-review markers are removed.
- **Modeling set:** Seoul and Gyeonggi provide **13,944 deduplicated rows**, 353
  restaurants, and 5,205 users. Interactions are split 80/20 per user with seed
  42; the other crawled regions are stored but not used for training.

### Text and feature pipeline

Reviews are aggregated into one document per user and one per restaurant, then
processed as follows:

```text
Okt morphology (stem=True)
→ vocabulary of 13,548 observed tokens
→ pad/truncate to 550 tokens
→ fine-tuned fastText cc.ko.300 embeddings
```

The pretrained embedding coverage is **62.4%**; out-of-vocabulary vectors are
initialized from `N(0, 0.6)`, and the test `<UNK>` rate is 1.24%.

### Model

The user and item documents pass through separate CNN towers. Their embeddings
are concatenated with taste, price, and service features, then scored by a
Factorization Machine with user/item bias terms:

```text
rating = global_bias + W·z + ½ Σ[(z·V)² − (z²·V²)] + b_user + b_item
```

The improved model adds five changes to DeepCoNN v1:

| # | Improvement | Purpose |
|---|---|---|
| 1 | Multi-scale CNN kernels `[2,3,4]` | Capture several n-gram widths in parallel |
| 2 | Attention pooling | Preserve context beyond the strongest activation |
| 3 | Taste/price/service fusion | Combine structured and textual evidence |
| 4 | User/item bias embeddings | Separate rating habits from review sentiment |
| 5 | Rating normalization and clipping | Keep regression within the valid rating range |

Training uses MSE loss, RMSprop (`alpha=0.9`, learning rate `1e-3`, weight decay
`1e-4`), dropout 0.2, batch size 64, gradient clipping at 1.0, and up to 15
epochs with early stopping (`patience=3`). The best validation checkpoint is
saved to `best_model.pt`; MLflow records parameters, metrics, improvement tags,
and artifacts under the `deepconn_improved` experiment.

## 3. Results

### Model comparison

| Model | RMSE ↓ | Precision@3 | Recall@3 | NDCG@3 |
|---|---:|---:|---:|---:|
| Matrix Factorization baseline | 0.7098 | 0.4301 | 0.9901 | 0.9870 |
| DeepCoNN v1 | 0.8045 | 0.4301 | 0.9901 | 0.9873 |
| **DeepCoNN Improved** | **0.5749** | 0.4301 | 0.9901 | 0.9918 |
| Improved without normalization | 0.5780 | 0.4301 | 0.9901 | 0.9929 |

The improved model reduced RMSE from **0.8045 to 0.5749** relative to the first
DeepCoNN implementation and from **0.7098 to 0.5749** relative to Matrix
Factorization—a **19% reduction** over the rating-only baseline.

Only **RMSE is discriminative in the current evaluation**. Precision@3 and
Recall@3 are identical across all four models because each user's test set is
too small for a top-3 slice to be informative. Ranking performance therefore
requires a future leave-one-out evaluation with sampled negatives.

### What improved the model

- **The prediction head was the initial bottleneck.** DeepCoNN v1 performed
  worse than Matrix Factorization despite having more capacity. User/item
  biases, output-range control, and gradient clipping stabilized prediction.
- **Bounded targets need bounded behavior.** Random FM initialization and an
  unconstrained quadratic term produced negative ratings early in training.
  Normalization and clipping reduced first-epoch loss from about 1,294 to 0.08.
- **Multi-scale kernels and attention pooling produced the largest quality
  gain**, moving RMSE from roughly 0.80 to 0.57. Max-pooling one activation per
  filter discarded too much information from 550-token review documents.
- **Crawler robustness dominated the data work.** Popup handling, tab recovery,
  and retries required more implementation effort than parsing itself.

### Limitations and next steps

- **37.6% of the vocabulary lacks a pretrained fastText vector**, including
  sentiment-heavy review slang. A subword, KoBERT, or Gemma encoder is the next
  comparison target.
- Training currently uses only Seoul and Gyeonggi; the Busan, Daegu, and
  nationwide crawls remain unused.
- The vocabulary and fitted `LabelEncoder` objects are not stored with the
  checkpoint, so preprocessing must currently be rerun for inference.
- The ranking protocol must be redesigned before making ranking-quality claims.

## Reproducing the project

```bash
pip install -r requirements.txt
```

Python 3.10 is recommended. The notebooks were originally run across Python
3.8.20, 3.9.21, and 3.10.18, and dependency versions are not pinned.

Additional requirements:

- **KoNLPy/Okt:** a JDK;
- **Selenium:** compatible Chrome and ChromeDriver versions;
- **MySQL:** used by the regional bulk-load notebook, with credentials in
  `.env`;
- **MLflow:** a tracking server at `http://localhost:5000` (`mlflow ui`); and
- **fastText:** `cc.ko.300.bin`, downloaded separately from
  [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html).

### Storage

Regional CSVs are bulk-loaded into local MySQL with `LOAD DATA LOCAL INFILE`
instead of row-wise inserts. The SQL `SET` clause parses rating suffixes, maps
Korean category labels to ordinals, and converts Korean date strings. UTF-8
encoding is fixed to `utf8mb4`, and newlines inside reviews are collapsed before
loading so a single review is not split across rows.

MySQL is the durable landing zone; the modeling notebooks currently read the
CSV files directly.

## Repository contents

| File | Description |
|---|---|
| [rating_extraction.ipynb](./rating_extraction.ipynb) | DiningCode Selenium/BeautifulSoup crawler |
| [sql_sending.ipynb](./sql_sending.ipynb) | Cleaning, MySQL schema, and regional bulk loading |
| [diningcode_analysis.ipynb](./diningcode_analysis.ipynb) | Preprocessing, MF baseline, and DeepCoNN v1 |
| [diningcode_analysis_improved.ipynb](./diningcode_analysis_improved.ipynb) | Improved model, MLflow training, and evaluation |
| [diningcode_no_norm.ipynb](./diningcode_no_norm.ipynb) | Ablation without normalization and clipping |
| [development.md](./development.md) | Experiment log |
| [crawled_data/](./crawled_data/) | Raw regional crawl outputs |
| `best_model.pt` | Best checkpoint by validation RMSE |
| [predict_result.csv](./predict_result.csv) | Test predictions joined with ratings and review text |
| [requirements.txt](./requirements.txt) | Project dependencies |
