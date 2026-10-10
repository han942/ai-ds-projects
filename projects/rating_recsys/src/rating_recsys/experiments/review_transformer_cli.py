"""Explicit local preparation and validation-only review Transformer experiment.

python -m rating_recsys.experiments.review_transformer_cli --epochs 3
Use --plan to inspect input counts without E5 inference or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import evaluate_rankings
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import observations, paired_bootstrap
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import (
    code_manifest, dataset_digest, load_review_texts, load_snapshot, review_texts_path,
)
from rating_recsys.retrieval.review_embeddings import E5_MODEL, ProfileFormatter, model_embedding_config
from rating_recsys.retrieval.review_transformer import (
    VERSION, ReviewBlockCache, ReviewStore, ReviewTransformer, ReviewTransformerConfig, review_blocks,
)


def write_json(path, value):
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    staging.replace(path)


def log(value):
    print(value, file=sys.stderr, flush=True)


def alias_file(source, destination):
    """Publish a complete immutable file under the usual convenient name."""
    staging = destination.with_suffix(destination.suffix + ".tmp")
    staging.unlink(missing_ok=True)
    os.link(source, staging)
    staging.replace(destination)


def resume_state(run):
    saved = torch.load(run / "training_last.pt", map_location="cpu", weights_only=False)
    state = saved.get("context")
    if not state or len(state["epochs"]) != saved["epoch"]:
        raise ValueError("Training checkpoint has no consistent evaluation state")
    return state, saved["epoch"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=PROJECT_ROOT / "artifacts/snapshots/e7896add5b4b5939.jsonl")
    p.add_argument("--baseline-run", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/two_tower/20261008T002625755462Z-e7896add")
    p.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts/comparisons/review_transformer")
    p.add_argument("--cache-path", type=Path, default=PROJECT_ROOT / "artifacts/e5_review_blocks_cache.sqlite")
    p.add_argument("--local-model-cache", type=Path, default=PROJECT_ROOT / "artifacts/local_models")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=None,
                   help="Stop after N epochs without matching validation NDCG@10 improvement")
    p.add_argument("--min-epochs", type=int, default=1,
                   help="Earliest epoch at which patience can stop training")
    p.add_argument("--resume-run", type=Path, default=None,
                   help="Resume an interrupted run with saved optimizer and RNG state")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embedding-batch-size", type=int, default=16)
    p.add_argument("--candidate-k", type=int, default=200)
    p.add_argument("--max-examples", type=int, default=None, help="Prespecified pilot subset of training examples")
    p.add_argument("--max-users", type=int, default=None, help="Prespecified hash sample of validation users")
    p.add_argument("--embedding-pilot", type=int, default=None, help="Encode only N train reviews to measure local E5 cost; no training")
    p.add_argument("--plan", action="store_true", help="Inspect block counts without model inference/training")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for value in (args.max_examples, args.max_users, args.embedding_pilot):
        if value is not None and value < 1:
            raise ValueError("Pilot limits must be positive")
    if args.patience is not None and args.patience < 1:
        raise ValueError("patience must be positive")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("min-epochs must be within the epoch budget")
    if args.candidate_k < 100:
        raise ValueError("candidate-k must be at least 100 for the shared evaluation")
    cfg = ReviewTransformerConfig(epochs=args.epochs, threads=args.threads,
                                 batch_size=args.batch_size, candidate_k=args.candidate_k)
    prior = json.loads((args.baseline_run / "manifest.json").read_text())
    exp = ExperimentConfig(**prior["config"])
    rows = load_snapshot(args.snapshot)
    if dataset_digest(rows) != prior["snapshot"]["dataset_snapshot_id"]:
        raise ValueError("Baseline snapshot differs")
    texts, text_meta = load_review_texts(review_texts_path(args.snapshot), rows)
    split = build_global_temporal_split(rows, train_fraction=exp.train_fraction,
                                       validation_fraction=exp.validation_fraction)
    queries, _ = build_window_queries(split.train, split.validation, config=exp,
                                     phase="validation", cutoff=split.train_cutoff)
    if args.max_users:
        queries = tuple(sorted(queries, key=lambda q: hashlib.sha256(str(q.user_id).encode()).hexdigest())[:args.max_users])
    embedding = replace(model_embedding_config(E5_MODEL, str(args.cache_path.resolve())),
                        local_model_cache=str(args.local_model_cache.resolve()),
                        max_document_tokens=cfg.block_tokens, batch_size=args.embedding_batch_size,
                        cpu_threads=args.threads)
    formatter = ProfileFormatter(embedding, args.cache_path.parent / "tokenizers")
    if args.plan:
        chars, retained, blocks, capped, unique = 0, 0, 0, 0, set()
        for row in split.train:
            chunks, audit = review_blocks(texts.get(row.review_id), formatter, cfg)
            chars += audit["source_chars"]; retained += audit["retained_chars"]
            blocks += len(chunks); capped += audit["available_blocks"] > cfg.max_blocks
            for chunk in chunks:
                unique.update(formatter.format(chunk["text"], role) for role in ("query", "document"))
        result = {"training_rows": len(split.train), "validation_queries": len(queries),
                  "blocks": blocks, "unique_role_inputs": len(unique), "capped_reviews": capped,
                  "source_chars": chars, "retained_chars": retained, "config": cfg.to_dict(),
                  "api_requests": 0, "training_executed": False}
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + dataset_digest(rows)[:8]
    run = args.artifacts_dir / run_id
    if args.resume_run:
        run = args.resume_run.resolve()
        if not (run / "training_last.pt").is_file():
            raise ValueError("Resume requires training_last.pt with optimizer/RNG state")
        resumed_state, resume_epoch = resume_state(run)
        for field,value in (("config", cfg.to_dict()), ("experiment", exp.to_dict()),
                            ("snapshot_id",dataset_digest(rows)),("review_texts",text_meta),
                            ("cutoff",split.train_cutoff.isoformat()),("max_examples",args.max_examples),
                            ("baseline_run",str(args.baseline_run)),
                            ("max_users",args.max_users), ("validation_user_ids",[q.user_id for q in queries])):
            if resumed_state[field] != value:
                raise ValueError(f"Resume run differs: {field}")
        resume_finished = resume_epoch >= args.epochs or should_stop(
            resume_epoch,resumed_state["selected_epoch"],min_epochs=args.min_epochs,patience=args.patience)
        if (resumed_state["stopping"]["min_epochs"] != args.min_epochs or
                resumed_state["stopping"]["patience"] != args.patience):
            raise ValueError("Resume stopping policy differs")
    else:
        run.mkdir(parents=True)
    state = {"status": "preparing", "version": VERSION, "phase": "validation-only",
             "test_evaluated": False, "config": cfg.to_dict(), "experiment": exp.to_dict(),
             "snapshot_id": dataset_digest(rows), "review_texts": text_meta,
             "cutoff": split.train_cutoff.isoformat(), "validation_user_ids": [q.user_id for q in queries],
             "max_examples": args.max_examples, "max_users": args.max_users,
             "stopping": {"max_epochs": args.epochs, "min_epochs": args.min_epochs,
                          "patience": args.patience, "metric": "matching validation NDCG@10"},
             "embedding_pilot": args.embedding_pilot, "baseline_run": str(args.baseline_run),
             "command": [sys.executable, "-m", "rating_recsys.experiments.review_transformer_cli",
                         *(argv if argv is not None else sys.argv[1:])],
             "evidence_status": "exploratory-validation; previously used for model selection",
             "epochs": []}
    if args.resume_run:
        state = resumed_state
        state.update(status="preparing")
        state.setdefault("resumptions", []).append({"command": [sys.executable,*sys.argv[1:]],
                                                     "after_epoch": resume_epoch,
                                                     "at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(run / "progress.json", state)
    source = code_manifest(PROJECT_ROOT, allow_dirty=True)
    diff_name = "source.resume.diff" if args.resume_run else "source.diff"
    (run / diff_name).write_text(source.pop("git_diff"))
    state["resume_code" if args.resume_run else "code"] = source
    write_json(run / "progress.json", state)
    log(f"[review Transformer] {run}")
    try:
        with ReviewBlockCache(args.cache_path, embedding, cfg, progress=log) as cache:
            prepare_rows = split.train[:args.embedding_pilot] if args.embedding_pilot else split.train
            store = ReviewStore.prepare(prepare_rows, texts, formatter, cache, cfg, progress=log)
            write_json(run / "inputs.json", store.manifest)
            del cache.encoder
        gc.collect()
        if args.embedding_pilot:
            state.update(status="complete", kind="embedding-cost-pilot", wall_seconds=time.perf_counter() - started,
                         inputs=store.manifest["unique_inputs"], usage=store.manifest["cache_usage"],
                         training_executed=False)
            write_json(run / "progress.json", state)
            print(json.dumps({"run": str(run), "kind": state["kind"], "usage": state["usage"]}, ensure_ascii=False))
            return state
        catalog = {r.restaurant_id for r in split.train}
        old = {}
        for line in (args.baseline_run / "candidates_validation.jsonl").read_text().splitlines():
            row = json.loads(line)
            old[row["query_id"]] = tuple(row["c5_c1_lightgcn_rrf"])
        if any(q.query_id not in old for q in queries):
            raise ValueError("Baseline query IDs differ")
        def evaluate(ordered, cutoffs=None):
            return evaluate_rankings(observations(queries, ordered),
                cutoffs=cutoffs or tuple(sorted({10, 100, cfg.candidate_k})), catalog_ids=catalog)
        baseline = evaluate(old, (10, 100))
        state.update(status="training", baseline_validation=baseline)
        write_json(run / "progress.json", state)
        best, best_ranked, best_retrieval = -float("inf"), None, None
        if args.resume_run:
            best = max(e["matching_validation"]["ndcg_at_10"] for e in state["epochs"])
            records = [json.loads(line) for line in (run / state["candidates_file"]).read_text().splitlines()]
            best_ranked = {r["query_id"]:tuple(r["review_transformer"]) for r in records}
            best_retrieval = {r["query_id"]:tuple(r["retrieval"]) for r in records}
            alias_file(run / state["checkpoint"]["path"],run / "review_transformer_validation.pt")
            alias_file(run / state["candidates_file"],run / "candidates_validation.jsonl")
        def callback(stats, model):
            nonlocal best, best_ranked, best_retrieval
            retrieved, ranked = model.rankings([q.user_id for q in queries],
                {q.user_id: [r.restaurant_id for r in q.history] for q in queries}, cfg.candidate_k)
            retrieved = {q.query_id: retrieved[q.user_id] for q in queries}
            ranked = {q.query_id: ranked[q.user_id] for q in queries}
            retrieval_metrics, metrics = evaluate(retrieved), evaluate(ranked)
            for q in queries:
                chosen = ranked[q.query_id]
                assert not (set(chosen) & {r.restaurant_id for r in q.history})
                assert set(chosen) <= catalog and len(chosen) == len(set(chosen))
                assert set(chosen) <= set(retrieved[q.query_id])
            stats = {**stats, "retrieval_validation": retrieval_metrics, "matching_validation": metrics,
                     "retrieval_exposure_at10": exposure(retrieved),
                     "matching_exposure_at10": exposure(ranked),
                     "ranking_seconds": model.metadata["last_ranking_seconds"]}
            state["epochs"].append(stats)
            log(f"epoch {stats['epoch']}: loss={stats['loss']:.4f}; "
                f"Recall@{cfg.candidate_k}={retrieval_metrics[f'recall_at_{cfg.candidate_k}']:.4%}; "
                f"NDCG@10={metrics['ndcg_at_10']:.6f}; train={stats['seconds']:.1f}s, "
                f"rank={stats['ranking_seconds']:.1f}s")
            if metrics["ndcg_at_10"] > best:
                best = metrics["ndcg_at_10"]; best_ranked, best_retrieval = ranked, retrieved
                state["selected_epoch"] = stats["epoch"]
                checkpoint = run / f"review_transformer_validation_epoch_{stats['epoch']:03d}.pt"
                state["checkpoint"] = model.save(checkpoint)
                candidates = run / f"candidates_validation_epoch_{stats['epoch']:03d}.jsonl"
                staging = candidates.with_suffix(candidates.suffix + ".tmp")
                with staging.open("w") as stream:
                    for q in queries:
                        stream.write(json.dumps({"query_id": q.query_id, "user_id": q.user_id,
                            "retrieval": retrieved[q.query_id], "review_transformer": ranked[q.query_id],
                            "c5": old[q.query_id]}, ensure_ascii=False) + "\n")
                staging.replace(candidates)
                state["candidates_file"] = candidates.name
                alias_file(checkpoint,run / "review_transformer_validation.pt")
                alias_file(candidates,run / "candidates_validation.jsonl")
            stop = should_stop(stats["epoch"], state["selected_epoch"],
                               min_epochs=args.min_epochs, patience=args.patience)
            if stop:
                state["stopping"]["reason"] = "validation patience exhausted"
            write_json(run / "progress.json", state)
            return stop
        if args.resume_run and resume_finished:
            model = ReviewTransformer.load(run / state["checkpoint"]["path"])
        else:
            model = ReviewTransformer(cfg, store, exp).fit(split.train, texts, cutoff=split.train_cutoff,
                eval_users=[q.user_id for q in queries], callback=callback, max_examples=args.max_examples,
                progress=log, training_checkpoint=run / "training_last.pt",
                resume_from=run / "training_last.pt" if args.resume_run else None,
                checkpoint_context=lambda:state)
        state.update(status="complete", selection="maximum validation matching NDCG@10; earliest tie",
                     selected_validation=evaluate(best_ranked), selected_retrieval_validation=evaluate(best_retrieval),
                     comparisons={"matching_vs_c5_at10": paired_bootstrap(queries, best_ranked, old, cutoff=10,
                         samples=2000, seed=cfg.seed), "retrieval_vs_c5_at100": paired_bootstrap(
                         queries, best_retrieval, old, cutoff=100, samples=2000, seed=cfg.seed)},
                     training=model.metadata, wall_seconds=time.perf_counter() - started,
                     wall_seconds_scope="current execution session",
                     max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     parameters=sum(p.numel() for p in model.network.parameters()), api_requests=0)
        state["stopping"].setdefault("reason", "epoch budget exhausted")
        state["stopping"]["executed_epochs"] = len(state["epochs"])
        write_json(run / "progress.json", state)
        _report(run, state)
        print(json.dumps({"run": str(run), "selected_epoch": state["selected_epoch"],
                          "validation": state["selected_validation"]}, ensure_ascii=False))
        return state
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__, error=str(error), wall_seconds=time.perf_counter() - started)
        write_json(run / "progress.json", state)
        raise


def should_stop(epoch, selected_epoch, *, min_epochs, patience):
    return patience is not None and epoch >= min_epochs and epoch - selected_epoch >= patience


def exposure(rankings, k=10):
    counts = Counter(item for values in rankings.values() for item in values[:k])
    item, count = counts.most_common(1)[0] if counts else (None, 0)
    return {"unique_items": len(counts), "most_exposed_item": item,
            "most_exposed_users": count, "users": len(rankings),
            "most_exposed_user_fraction": count / len(rankings) if rankings else 0.0}


def _report(run, state):
    base, match, retrieval = state["baseline_validation"], state["selected_validation"], state["selected_retrieval_validation"]
    text = ["# 리뷰 Transformer: validation 실행", "", "Validation에서만 학습 설정을 확인한 탐색 결과다. Test 평가·test refit은 하지 않았다.", "",
            "| 단계 | Recall@100 | NDCG@10 |", "|---|---:|---:|"]
    for name, metrics in (("기존 C5", base), ("리뷰 Transformer 검색", retrieval), ("cross-attention 재정렬", match)):
        text.append(f"| {name} | {metrics['recall_at_100']:.4%} | {metrics['ndcg_at_10']:.6f} |")
    text += ["", f"선택 epoch: {state['selected_epoch']}. 후보 {state['config']['candidate_k']}개를 상세 비교했다.", "",
             f"실행 epoch: {state['stopping']['executed_epochs']} / 최대 {state['config']['epochs']}. 종료: {state['stopping']['reason']}.",
             f"학습 예제: {state['training']['training_examples']:,} / 가능한 {state['training']['available_training_examples']:,}.",
             f"평가 query: {len(state['validation_user_ids']):,}. API 호출: 0. 파라미터: {state['parameters']:,}.",
             f"현재 실행 세션 경과: {state['wall_seconds'] / 60:.1f}분. 해당 세션 최대 RSS: {state['max_rss_kib'] / 1024**2:.2f}GiB.",
             f"확정 epoch의 학습 합계: {sum(e['seconds'] for e in state['epochs']) / 60:.1f}분; validation 순위 계산 합계: {sum(e['ranking_seconds'] for e in state['epochs']) / 60:.1f}분. 중단된 epoch의 재계산 비용은 이 합계에서 제외된다.", "",
             "`progress.json`은 설정·epoch별 지표·bootstrap·cutoff audit, `inputs.json`은 블록 근거와 임베딩 provenance를 담는다.",
             "유료 호출·새 모델 다운로드·기존 E5 concat 캐시 변경은 없다. 리뷰 블록 캐시 사용량은 `inputs.json`에 기록했다."]
    if state["max_examples"] or state["max_users"]:
        text += ["", "학습 예제 또는 평가 사용자 수를 제한한 CPU pilot이다. 전체 학습의 모델 성능으로 일반화하지 않는다."]
    (run / "report.md").write_text("\n".join(text) + "\n")


def console_main():
    main()


if __name__ == "__main__":
    main()
