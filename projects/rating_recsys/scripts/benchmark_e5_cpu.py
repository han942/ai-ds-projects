"""Benchmark E5 on the same raw samples as the earlier Gemma CPU test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.snapshot import load_review_texts, load_snapshot, review_texts_path
from rating_recsys.retrieval.review_embeddings import ReviewEmbeddingConfig
from rating_recsys.retrieval.review_profiles import build_profiles


MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def quantile_sample(values, count):
    ordered = sorted(values, key=len)
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    return [ordered[int(index)] for index in indices]


def e5_input(value):
    for source, target in (("task: search result | query: ", "query: "),
                           ("title: none | text: ", "passage: ")):
        if value.startswith(source):
            return target + value[len(source):]
    raise ValueError("Unknown sample role")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--review-samples", type=int, default=256)
    parser.add_argument("--profile-samples", type=int, default=64)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if min(args.review_samples, args.profile_samples, args.max_tokens,
           args.repeats, *args.threads, *args.batch_sizes) < 1:
        parser.error("sample sizes, threads, batches, tokens and repeats must be positive")
    if args.max_tokens > 512:
        parser.error("E5 supports at most 512 tokens")

    rows = load_snapshot(args.snapshot)
    texts, _ = load_review_texts(review_texts_path(args.snapshot), rows)
    split = build_global_temporal_split(rows, train_fraction=0.7, validation_fraction=0.15)
    past = split.train
    reviews = [" ".join((texts[row.review_id] or "").split()) for row in past]
    reviews = [value for value in reviews if value]
    users, items = build_profiles(past, texts, sorted({row.user_id for row in past}),
                                 ReviewEmbeddingConfig())
    # Select with the original prefixes before translating roles, preserving samples.
    profiles = [f"task: search result | query: {value}" for value in users.values()]
    profiles += [f"title: none | text: {value}" for value in items.values()]
    reference_samples = {
        "individual_reviews": [f"title: none | text: {value}"
                               for value in quantile_sample(reviews, args.review_samples)],
        "positive_concat_profiles": quantile_sample(profiles, args.profile_samples),
    }
    workloads = {name: [e5_input(value) for value in values]
                 for name, values in reference_samples.items()}
    result = {
        "model": MODEL, "revision": REVISION, "device": "cpu", "dtype": "float32",
        "sampling": "same raw samples and role selection as the historical Gemma CPU benchmark",
        "reference_samples_sha256": {
            name: hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
            for name, values in reference_samples.items()
        },
        "cutoff": split.train_cutoff.isoformat(), "max_tokens": args.max_tokens,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "sentence-transformers", "tokenizers")},
        "dataset": {"interactions": len(rows), "train_user_profiles": len(users),
                    "train_item_profiles": len(items),
                    "nonempty_reviews": sum(bool(value and value.strip()) for value in texts.values())},
        "runs": [],
    }
    logging.disable_progress_bar()
    logging.set_verbosity_error()
    torch.set_num_threads(args.threads[0])
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu",
                                cache_folder=str(args.cache_dir), local_files_only=True,
                                trust_remote_code=False,
                                model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = args.max_tokens
    result["load_seconds_excluding_download"] = time.perf_counter() - started
    result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    result["loaded_rss_mib"] = psutil.Process().memory_info().rss / 1024 ** 2
    result["loaded_memory_available_mib"] = psutil.virtual_memory().available / 1024 ** 2
    print(json.dumps({key: result[key] for key in
                      ("parameters", "loaded_rss_mib", "load_seconds_excluding_download")}), flush=True)

    for threads in args.threads:
        torch.set_num_threads(threads)
        for batch_size in args.batch_sizes:
            for name, values in workloads.items():
                token_counts = [len(model.tokenizer(value)["input_ids"]) for value in values]
                options = dict(batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True, prompt="")
                model.encode(values[:batch_size], **options)
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    embeddings = model.encode(values, **options)
                    elapsed = time.perf_counter() - started
                    if embeddings.shape != (len(values), 384) or not np.isfinite(embeddings).all():
                        raise RuntimeError("Invalid embedding dimensions or nonfinite output")
                    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
                        raise RuntimeError("Embeddings are not unit-normalized")
                    run = {
                        "workload": name, "threads": threads, "batch_size": batch_size,
                        "repeat": repeat, "samples": len(values), "seconds": elapsed,
                        "inputs_per_second": len(values) / elapsed,
                        "original_tokens_mean": float(np.mean(token_counts)),
                        "original_tokens_p90": float(np.percentile(token_counts, 90)),
                        "truncated_samples": sum(count > args.max_tokens for count in token_counts),
                        "rss_mib": psutil.Process().memory_info().rss / 1024 ** 2,
                        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        "hours_for_96922_inputs": 96922 * elapsed / len(values) / 3600,
                        "hours_for_59449_inputs": 59449 * elapsed / len(values) / 3600,
                    }
                    result["runs"].append(run)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
