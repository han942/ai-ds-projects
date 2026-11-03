# News simplification re-evaluation

This evaluates the **saved Gemma 3-1B outputs** in `../result.csv`, not the
LangGraph agent and not a newly generated model run. See [REPORT.md](REPORT.md)
for the completed local evaluation and its interpretation.

Run from `projects/NLP_Newspaper_CUAI`:

```bash
python3.12 -m venv .venv-eval
source .venv-eval/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r evaluation/requirements-eval.txt
python -m evaluation --work-dir evaluation/runs/full --threads 8
```

Once models and the BLEU metric are cached, add `--offline` to avoid contacting
Hugging Face. The three models are listed in `neural.py`; their cached revisions
for the recorded run are saved in `runs/full/model_revisions.json`.

Every completed metric stage is saved atomically. Running the same command
reuses those results and rebuilds the report. A manifest records the input hash,
row count, shuffle seed, metric source hashes, and package versions. Changed
inputs, metric code, or packages require a new `--work-dir`. Keep this directory
to resume after an interruption; it is ignored by Git. A failed stage runs again.

The work directory contains `per_sample.csv`, bootstrap `summary.csv`,
`corpus.json`, the manifest, and stage checkpoints. Treat pickle checkpoints
as trusted local files: do not reuse checkpoint directories from other people.

The optional judge sends source articles and both outputs to an external API
and incurs provider charges. It is disabled by default. After authorizing that
transfer and cost, run:

```bash
python -m evaluation --offline --work-dir evaluation/runs/full --judge-n 150
```

It uses `OPENROUTER_API_KEY` (OpenRouter, `openai/gpt-4o-mini`) when available,
otherwise `OPENAI_API_KEY` (OpenAI, `gpt-4o-mini`). Override `--judge-model` when
needed. Each successful rating is saved separately and reused only for the
same source, output, rubric, and model. Failed calls can be retried by rerunning.
The report discloses missing or failed ratings.

For the pending 150-source judge run, inspect `runs/full/judge_preview.csv` and
`runs/full/judge_plan.json` before approving the external transfer.

```bash
python -m unittest evaluation.test_evaluation -v
```

These tests check parity with the cached original SARI implementation, number
normalization examples, combined BERTScore pairing, paired differences, strict
judge parsing, per-output judge cache invalidation, and RoBERTa's effective
input limit (510 tokens for this NLI checkpoint). The SARI parity test is
skipped when the original implementation is not cached.
