# Restaurant rating prediction v1 (legacy)

This directory preserves the earlier DiningCode rating experiments. Active development
uses the [v2 two-stage recommender](../../README.md). v1 predicts ratings; its RMSE
cannot be compared directly with v2 retrieval Recall or ranking NDCG.

## Revised comparison

The reviewed experiment is [diningcode_revision.ipynb](./diningcode_revision.ipynb),
with results in [results/summary.md](./results/summary.md). It compares MF, DeepCoNN,
and a text + ID Hybrid under the same inputs, split, and validation selection rules.

- Seoul and Gyeonggi: 19,297 raw rows → 13,934 first user/restaurant pairs;
  5,205 users and 353 restaurants.
- Per-user random 80/20 split, seed 42; train 11,841 rows, including 2,392 validation
  rows. Test: 2,093 rows from 1,557 users with at least two reviews and known items.
- Hyperparameters and epochs selected on validation; refit on all train rows for
  model seeds 42, 43, and 44. Predictions clipped to [1, 5].
- Text documents contain training reviews only. Each training row's own review is
  excluded from both documents; taste, price, and service labels are excluded.

| Model | Test RMSE (mean ± SD across seeds) | ΔRMSE vs MF [95% paired bootstrap CI] |
|---|---:|---:|
| Global mean | 0.7422 | +0.0615 [+0.0469, +0.0765] |
| User + item bias | 0.6772 | −0.0035 [−0.0068, −0.0002] |
| MF | 0.6807 ± 0.0003 | – |
| DeepCoNN | 0.6949 ± 0.0021 | +0.0142 [+0.0053, +0.0230] |
| Hybrid | 0.6946 ± 0.0016 | +0.0139 [+0.0053, +0.0222] |

Under this protocol, DeepCoNN and Hybrid did not improve on MF. The split is random
within users, so this remains a v1 rating comparison, not v2's temporal recommendation
benchmark. See the summary for MAE, history segments, and selected settings.

## Historical results

The original notebooks recorded MF RMSE 0.7098, DeepCoNN v1 0.8045, Improved 0.5749,
and Improved without normalization 0.5780. The previously advertised “19% lower
RMSE” comes from these historical results and is not a validated improvement.
The original DeepCoNN used test reviews to build test documents; Improved included
target reviews and aspect scores as inputs and selected checkpoints on test RMSE.
The archived implementation also did not use attention as previously described.
Details are recorded in [archive/README.md](./archive/README.md).

The former Precision@3 and Recall@3 values were not useful measures of retrieval:
only each user's held-out items were ranked. They are omitted from the revised
comparison, which reports RMSE and MAE.

## Files and reproduction

| Path | Purpose |
|---|---|
| [diningcode_revision.ipynb](./diningcode_revision.ipynb) | Revised MF / DeepCoNN / Hybrid comparison |
| [results/](./results/) | Summary, metrics, predictions, input hashes, and run configuration |
| [diningcode_analysis.ipynb](./diningcode_analysis.ipynb) | Original preprocessing, MF, and DeepCoNN v1 |
| [rating_extraction.ipynb](./rating_extraction.ipynb) | Original Selenium / BeautifulSoup crawler |
| [archive/](./archive/) | Historical Improved / ablation notebooks, checkpoint, predictions, and MySQL loader |
| [development.md](./development.md) | Historical development log |
| [crawled_data/](./crawled_data/) | Regional CSV inputs |
| [requirements.txt](./requirements.txt) | v1 dependencies |

Run the revised notebook from this directory with a Python 3.10 environment:

```bash
cd projects/rating_recsys/legacy/v1_rating_prediction
pip install -r requirements.txt
```

Okt requires a JDK. Place the fastText text vectors at `.cache/cc.ko.300.vec.gz`,
or set `FASTTEXT_PATH` to a `.vec` / `.vec.gz` file (loaded with `binary=False`).
The original notebooks used `.bin`; the revised notebook uses text vectors.
The notebook takes about
1.5 hours on the recorded CPU setup. Dependency versions are not locked; the
recorded environment and input hashes are in [results/run_config.json](./results/run_config.json).
The older crawler additionally needs Chrome/ChromeDriver, and the archived MySQL
loader reads credentials from `.env` (`cp .env.example .env`). Archive notebooks
retain paths from their original locations and cannot be rerun there without path
adjustments. See [Korean documentation](./README_KOR.md).
