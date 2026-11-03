"""Evaluate result.csv and write evaluation/REPORT.md.

    python -m evaluation                    # all local stages
    python -m evaluation --limit 50         # quick smoke test
    python -m evaluation --judge-n 150      # also run the LLM judge (external API)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import judge as judge_mod
from . import lexical, neural, report, rules

PROJECT = Path(__file__).resolve().parent.parent
COLUMNS = ("original", "simplified_human", "simplified_by_model")


def _shuffled(outputs: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(outputs))
    while len(outputs) > 1 and (perm == np.arange(len(outputs))).any():
        perm = rng.permutation(len(outputs))
    return [outputs[i] for i in perm]


def _long(values: dict[str, dict[str, np.ndarray | list]], n: int) -> pd.DataFrame:
    frames = []
    for system, cols in values.items():
        f = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in cols.items()})
        f.insert(0, "idx", np.arange(n)), f.insert(1, "system", system)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=PROJECT / "result.csv")
    parser.add_argument("--out", type=Path, default=PROJECT / "evaluation" / "REPORT.md")
    parser.add_argument("--limit", type=int, help="evaluate only the first N rows")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--work-dir", type=Path, default=PROJECT / "evaluation" / "runs" / "full", help="persistent per-stage results for resuming")
    parser.add_argument("--offline", action="store_true", help="use cached models and metrics without contacting Hugging Face")
    parser.add_argument("--judge-n", type=int, default=0, help="number of sources for the LLM judge (0 = skip)")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 3:
        parser.error("--limit must be at least 3")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    import torch

    torch.set_num_threads(args.threads)
    data = pd.read_csv(args.csv)
    missing = [c for c in COLUMNS if c not in data.columns]
    if missing:
        sys.exit(f"missing columns in {args.csv}: {missing}")
    data = data[list(COLUMNS)]
    if data.isna().any().any() or data.astype(str).apply(lambda c: c.str.strip().eq("")).any().any():
        sys.exit("empty input text: fix the CSV before evaluation")
    data = data.astype(str)
    if args.limit:
        data = data.head(args.limit)
    data = data.reset_index(drop=True)
    n = len(data)
    if n < 3:
        parser.error("at least 3 rows are required for source-length groups")
    if not 0 <= args.judge_n <= n:
        parser.error("--judge-n must be between 0 and the evaluated row count")
    src, ref, out = data.original.tolist(), data.simplified_human.tolist(), data.simplified_by_model.tolist()
    shuffled = _shuffled(out, args.seed)
    systems = {"model": out, "reference": ref, "copy": src, "shuffled": shuffled}

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_METRICS_CACHE", str(work / "metric-cache"))
    versions = {p: importlib.metadata.version(p) for p in (
        "torch", "numpy", "pandas", "sacrebleu", "rouge", "textstat", "kiwipiepy",
        "wordfreq", "transformers", "sentence-transformers", "bert-score", "evaluate")}
    identity = {
        "input_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "rows": n, "seed": args.seed,
        "metric_code": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.glob("*.py") if p.name in (
                            "__init__.py", "lexical.py", "rules.py", "sari.py", "korean.py", "neural.py")},
        "packages": versions,
    }
    manifest = work / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            sys.exit(f"cached inputs, metric code, or packages changed; choose a new --work-dir: {work}")
    elif list(work.glob("*.pkl")):
        sys.exit(f"unverified legacy checkpoints; choose a new --work-dir: {work}")
    else:
        manifest.write_text(json.dumps({"identity": identity, "csv": str(args.csv.resolve()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duplicate_sources": int(data.original.duplicated().sum()),
            "duplicate_rows": int(data.duplicated().sum())}, indent=2))

    def stage(name, fn):
        path = work / f"{name}.pkl"
        if path.exists():
            print(f"[{name}] reuse {path}", flush=True)
            return pd.read_pickle(path)
        t0 = time.time()
        print(f"[{name}] running…", flush=True)
        result = fn()
        tmp = path.with_suffix(".tmp")
        pd.to_pickle(result, tmp)
        tmp.replace(path)
        print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
        return result

    def run_lexical():
        frames = []
        for s in ("model", "copy", "shuffled"):
            f = lexical.per_sample(src, systems[s], ref)
            f.insert(0, "idx", np.arange(n)), f.insert(1, "system", s)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def run_rules():
        frames = []
        for s in ("model", "reference", "copy"):
            f = rules.per_sample(src, systems[s])
            f.insert(0, "idx", np.arange(n)), f.insert(1, "system", s)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def run_sbert():
        sims = neural.sbert_similarity({
            "model|src": (src, out), "reference|src": (src, ref), "shuffled|src": (src, shuffled),
            "model|ref": (ref, out), "copy|ref": (ref, src), "shuffled|ref": (ref, shuffled)})
        return _long({
            "model": {"sbert_src": sims["model|src"], "sbert_ref": sims["model|ref"]},
            "reference": {"sbert_src": sims["reference|src"]},
            "copy": {"sbert_src": np.ones(n), "sbert_ref": sims["copy|ref"]},
            "shuffled": {"sbert_src": sims["shuffled|src"], "sbert_ref": sims["shuffled|ref"]},
        }, n)

    def run_bertscore():
        bs = neural.bertscore({
            "model|src": (out, src), "reference|src": (ref, src), "shuffled|src": (shuffled, src),
            "model|ref": (out, ref), "copy|ref": (src, ref), "shuffled|ref": (shuffled, ref)})
        to_src = lambda k: {"bertscore_src_p": bs[k]["p"], "bertscore_src_r": bs[k]["r"], "bertscore_src_f": bs[k]["f"]}
        return _long({
            "model": {**to_src("model|src"), "bertscore_ref_f": bs["model|ref"]["f"]},
            "reference": to_src("reference|src"),
            "copy": {"bertscore_src_p": np.ones(n), "bertscore_src_r": np.ones(n), "bertscore_src_f": np.ones(n),
                     "bertscore_ref_f": bs["copy|ref"]["f"]},
            "shuffled": {**to_src("shuffled|src"), "bertscore_ref_f": bs["shuffled|ref"]["f"]},
        }, n)

    def run_nli():
        return _long(neural.nli_faithfulness({s: (src, systems[s]) for s in ("model", "reference", "copy")},
            checkpoint_dir=work / "nli"), n)

    def run_corpus():
        values = lexical.notebook_reproduction(src, out, ref)
        for s in ("model", "copy", "shuffled"):
            values.update({f"{s}_{k}": v for k, v in lexical.corpus(src, systems[s], ref).items()})
        return values

    per = pd.DataFrame({"idx": np.repeat(np.arange(n), len(systems)), "system": list(systems) * n})
    for name, fn in [("lexical", run_lexical), ("rules", run_rules), ("sbert", run_sbert),
                     ("bertscore", run_bertscore), ("nli", run_nli)]:
        per = per.merge(stage(name, fn), on=["idx", "system"], how="left")
    corpus = stage("corpus", run_corpus)
    per.to_csv(work / "per_sample.csv", index=False)
    report.summarize(per).to_csv(work / "summary.csv", index=False)
    (work / "corpus.json").write_text(json.dumps(corpus, indent=2))

    judged = None
    if args.judge_n:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env", override=False)
        if args.judge_model is None:
            args.judge_model = "openai/gpt-4o-mini" if os.getenv("OPENROUTER_API_KEY") else "gpt-4o-mini"
        indices = judge_mod.stratified_sample(data.original.str.len(), args.judge_n, args.seed)
        judged = judge_mod.judge(src, {"model": out, "reference": ref}, indices, args.judge_model,
            checkpoint=work / f"judge_{args.judge_model.replace('/', '_')}_{args.judge_n}.jsonl")
        judged.to_csv(work / "judge.csv", index=False)

    command = "python -m evaluation --work-dir " + shlex.quote(str(work.relative_to(PROJECT) if work.is_relative_to(PROJECT) else work))
    if args.offline:
        command += " --offline"
    if args.limit:
        command += f" --limit {args.limit}"
    if args.seed:
        command += f" --seed {args.seed}"
    command += f" --threads {args.threads}"
    if args.csv.resolve() != (PROJECT / "result.csv").resolve():
        command += " --csv " + shlex.quote(str(args.csv))
    if args.out.resolve() != (PROJECT / "evaluation" / "REPORT.md").resolve():
        command += " --out " + shlex.quote(str(args.out))
    if args.judge_n:
        command += f" --judge-n {args.judge_n} --judge-model {shlex.quote(args.judge_model)}"
    from .figures import sari_baselines

    sari_baselines(report.summarize(per), args.out.parent / "figures" / "sari_baselines.png")
    limits_path = work / "token_limits.json"
    limits = json.loads(limits_path.read_text()) if limits_path.exists() else None
    text = report.build_report(data, per, corpus, judged, args.out, command, seed=args.seed, token_limits=limits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    (work / "completed.json").write_text(json.dumps({"completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": n, "report": str(args.out.resolve()), "judge_n": args.judge_n}, indent=2))


if __name__ == "__main__":
    main()
