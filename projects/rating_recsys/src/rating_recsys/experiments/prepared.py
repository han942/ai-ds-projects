"""Persist cutoff-safe baseline rows and candidates, independent of ranker tuning."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from rating_recsys.experiments.artifacts import read_json, write_json
from rating_recsys.experiments.snapshot import canonical_json


def _digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def preprocessing_signature() -> str:
    """Ignore docs, new candidate adapters and ranker grid changes."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "datasets/models.py", "datasets/split.py", "experiments/queries.py",
        "experiments/models.py", "experiments/snapshot.py", "ranking/features.py",
        "retrieval/baselines.py", "retrieval/hybrid.py", "retrieval/lightgcn.py",
        "experiments/prepared.py",
    )
    sources = {name: _file_digest(root / name) for name in paths}
    pipeline_names = (
        "TrainingArrays", "WindowCandidates", "build_context", "checkpoint_start",
        "fit_lightgcn", "CheckpointedLightGCN", "prefix_training_arrays",
        "window_candidates", "window_training_arrays", "_generator",
    )
    text = (root / "experiments/pipeline.py").read_text()
    for node in ast.parse(text).body:
        if getattr(node, "name", None) in pipeline_names:
            sources[f"pipeline.{node.name}"] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "CANDIDATE_STAGES" for target in node.targets):
            sources["candidate_stages"] = ast.get_source_segment(text, node)
    text = (root / "experiments/config.py").read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == "ExperimentConfig":
            for method in node.body:
                if getattr(method, "name", None) in ("relevance", "satisfaction_profile", "training_label", "feature_names", "include_region", "lightgcn_config"):
                    sources[f"config.{method.name}"] = ast.get_source_segment(text, method)
    return _digest(sources)


def add_cache_arguments(parser) -> None:
    group = parser.add_argument_group("저장된 baseline 학습 데이터")
    choice = group.add_mutually_exclusive_group()
    choice.add_argument("--no-cache", action="store_true", help="저장된 데이터 사용·저장을 생략")
    choice.add_argument("--rebuild-cache", action="store_true", help="같은 조건의 저장 데이터도 다시 생성")


class PreparedData:
    def __init__(self, artifacts_root, snapshot_id, split, config, *, enabled=True, rebuild=False, log=None):
        if rebuild and not enabled:
            raise ValueError("--no-cache and --rebuild-cache cannot be combined")
        self.artifacts_root = Path(artifacts_root)
        self.root = self.artifacts_root / "prepared" / snapshot_id[:16]
        self.config = config
        self.enabled, self.rebuild = enabled, rebuild
        self.log = log or (lambda message: None)
        self.entries = []
        self.identity = {
            "schema": "prepared-baseline-v1", "snapshot_id": snapshot_id,
            "cutoffs": split.summary()["cutoffs"],
            "preprocessing": preprocessing_signature(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
            "candidate_k": config.candidate_k, "rrf_constant": config.rrf_constant,
            "region_mode": config.region_mode, "legacy_c3_quota": config.legacy_c3_quota,
            "lightgcn": config.lightgcn_config.to_dict(), "features": list(config.feature_names),
            "relevance_thresholds": [config.relevance_low_threshold, config.relevance_high_threshold],
            "satisfaction": {key: value for key, value in config.to_dict().items() if key.startswith("satisfaction_")},
            "rating_shrinkage_strength": config.rating_shrinkage_strength,
        }

    @property
    def manifest(self):
        return {"enabled": self.enabled, "entries": self.entries}

    def _get(self, kind, extra, build):
        identity = {**self.identity, "kind": kind, **extra}
        key = _digest(identity)
        folder = self.root / f"{kind}-{key[:16]}"
        entry = {"kind": kind, "key": key, "path": str(folder.relative_to(self.artifacts_root))}
        if not self.enabled:
            self.entries.append({**entry, "status": "disabled"})
            return (*build(), False)
        folder.mkdir(parents=True, exist_ok=True)
        # One writer per entry. Readers never consume an unfinished build.
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.rebuild and (folder / "manifest.json").exists():
                try:
                    manifest = read_json(folder / "manifest.json")
                    checksum = manifest.pop("metadata_sha256")
                    if _digest(manifest) != checksum or manifest["identity"] != identity:
                        raise ValueError("prepared metadata mismatch")
                    if _file_digest(folder / "data.npz") != manifest["arrays_sha256"]:
                        raise ValueError("prepared arrays checksum mismatch")
                    with np.load(folder / "data.npz", allow_pickle=False) as archive:
                        arrays = {name: archive[name] for name in archive.files}
                    self.entries.append({**entry, "status": "hit"})
                    self.log(f"prepared {kind}: HIT {folder}")
                    return arrays, manifest["data"], True
                except (OSError, ValueError, KeyError, EOFError, TypeError, AttributeError) as error:
                    self.log(f"prepared {kind}: invalid entry, rebuilding ({error})")
            self.log(f"prepared {kind}: BUILD {folder}")
            arrays, data = build()
            # Replace arrays first and the validated manifest last while holding the lock.
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, **arrays)
                manifest = {"identity": identity, "data": data, "arrays_sha256": _file_digest(temporary)}
                # JSON converts integer map keys to strings; hash the persisted representation.
                manifest = json.loads(canonical_json(manifest))
                manifest["metadata_sha256"] = _digest(manifest)
                os.replace(temporary, folder / "data.npz")
                with tempfile.NamedTemporaryFile(dir=folder, suffix=".json", delete=False) as handle:
                    metadata_path = Path(handle.name)
                try:
                    write_json(metadata_path, manifest)
                    os.replace(metadata_path, folder / "manifest.json")
                finally:
                    metadata_path.unlink(missing_ok=True)
            finally:
                temporary.unlink(missing_ok=True)
            self.entries.append({**entry, "status": "built"})
            return arrays, data, False

    def training(self, phase, reference, through, graphs):
        from rating_recsys.experiments.pipeline import (
            TrainingArrays, _generator, prefix_training_arrays, window_training_arrays,
        )
        from rating_recsys.experiments.queries import build_prefix_queries

        def build():
            before = {key: row["queries"] for key, row in graphs.summary.items()}
            if self.config.ranker_training_mode == "window":
                result = window_training_arrays(
                    reference, _generator(self.config), graphs, config=self.config,
                    through=through, phase=phase, log=self.log,
                )
            else:
                queries = build_prefix_queries(reference, config=self.config, phase=phase)
                if phase == "refit":
                    from datetime import date
                    t1 = date.fromisoformat(self.identity["cutoffs"]["train_through"])
                    queries = tuple(q for q in queries if q.target.event_date > t1)
                if any(q.target.event_date > through for q in queries):
                    raise ValueError("Prepared training target exceeds cutoff")
                result = prefix_training_arrays(
                    queries, reference, _generator(self.config), graphs,
                    feature_names=self.config.feature_names, config=self.config,
                    log=self.log, label=f"{phase}_prefix",
                )
            checkpoints = [
                {**row, "queries": row["queries"] - before.get(row["checkpoint"], 0)}
                for row in graphs.rows() if row["queries"] > before.get(row["checkpoint"], 0)
            ]
            return {
                "features": result.features, "labels": result.labels,
                "groups": np.asarray(result.groups, dtype=np.int64), "rating_priors": result.rating_priors,
            }, {"summary": result.summary, "checkpoints": checkpoints}

        arrays, data, hit = self._get(phase, {
            "through": through.isoformat(), "training_mode": self.config.ranker_training_mode,
            "label_mode": self.config.ranker_label_mode,
            "checkpoint_months": self.config.lightgcn_checkpoint_months,
        }, build)
        if hit:
            for row in data["checkpoints"]:
                previous = graphs.summary.get(row["checkpoint"])
                if previous and previous["edges"] != row["edges"]:
                    raise ValueError("Prepared graph checkpoint sees different edges")
                graphs.summary[row["checkpoint"]] = {
                    **row, "queries": row["queries"] + (previous["queries"] if previous else 0),
                }
        groups = arrays["groups"].tolist()
        if sum(groups) != len(arrays["labels"]) or arrays["features"].shape != (sum(groups), len(self.config.feature_names)):
            raise ValueError("Prepared training group/feature dimensions differ")
        return TrainingArrays(arrays["features"], arrays["labels"], groups, data["summary"], arrays["rating_priors"])

    def window(self, phase, history, queries):
        from rating_recsys.experiments.pipeline import WindowCandidates, _generator, build_context, fit_lightgcn, window_candidates

        def build():
            graph, info = fit_lightgcn(history, self.config.lightgcn_config)
            result = window_candidates(
                queries, build_context(history), _generator(self.config), graph=graph,
                feature_names=self.config.feature_names,
                rating_shrinkage_strength=self.config.rating_shrinkage_strength,
            )
            arrays = {name: getattr(result, name) for name in ("features", "labels", "row_restaurant_ids")}
            data = {
                name: getattr(result, name) for name in (
                    "ordered", "group_sizes", "item_popularity", "item_regions", "item_names",
                    "positive_availability", "rating_prior",
                )
            }
            data.update(catalog=sorted(result.catalog), graph=info)
            return arrays, data

        arrays, data, _ = self._get(phase, {}, build)
        data = dict(data)
        info = data.pop("graph")
        data["catalog"] = set(data["catalog"])
        for name in ("item_popularity", "item_regions", "item_names"):
            data[name] = {int(key): value for key, value in data[name].items()}
        data["ordered"] = {
            stage: {key: tuple(value) for key, value in rankings.items()}
            for stage, rankings in data["ordered"].items()
        }
        if set(data["ordered"]["c5_c1_lightgcn_rrf"]) != {q.query_id for q in queries}:
            raise ValueError("Prepared candidates have different queries")
        if sum(data["group_sizes"]) != len(arrays["labels"]):
            raise ValueError("Prepared candidate group dimensions differ")
        return WindowCandidates(**arrays, **data), info


def main(argv=None):
    from rating_recsys.config import PROJECT_ROOT
    from rating_recsys.datasets.split import build_global_temporal_split
    from rating_recsys.experiments.config import ExperimentConfig
    from rating_recsys.experiments.pipeline import CheckpointedLightGCN
    from rating_recsys.experiments.queries import build_window_queries
    from rating_recsys.experiments.snapshot import freeze_snapshot, load_snapshot

    parser = argparse.ArgumentParser(prog="rating-recsys-prepare", description="Baseline 후보·학습 데이터를 한 번 생성해 저장 (LTR 학습·평가 없음)")
    parser.add_argument("--snapshot", type=Path, required=True, help="고정된 interaction snapshot")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--scope", choices=("all", "candidates"), default="all")
    default = ExperimentConfig()
    fields = (
        "train_fraction", "validation_fraction", "candidate_k", "ranking_k", "rrf_constant",
        "random_seed", "region_mode", "ranker_training_mode", "ranker_label_mode",
        "rating_shrinkage_strength", "relevance_high_threshold", "relevance_low_threshold",
        "lightgcn_dimension", "lightgcn_layers", "lightgcn_epochs", "lightgcn_learning_rate",
        "lightgcn_regularization", "lightgcn_batch_size", "lightgcn_checkpoint_months", "legacy_c3_quota",
    )
    flags = {"random_seed": "seed", "relevance_high_threshold": "relevance-high", "relevance_low_threshold": "relevance-low"}
    choices = {"region_mode": ("with_region", "without_region"), "ranker_training_mode": ("prefix", "window"), "ranker_label_mode": ("relevance", "rating")}
    for name in fields:
        value = getattr(default, name)
        parser.add_argument(f"--{flags.get(name, name.replace('_', '-'))}", dest=name,
                            type=type(value), default=value, choices=choices.get(name))
    add_cache_arguments(parser)
    args = parser.parse_args(argv)
    config = ExperimentConfig(**{name: getattr(args, name) for name in fields})
    rows = tuple(load_snapshot(args.snapshot))
    split = build_global_temporal_split(rows, train_fraction=config.train_fraction, validation_fraction=config.validation_fraction)
    _, snapshot = freeze_snapshot(rows, args.artifacts_dir / "snapshots")
    import sys
    emit = lambda message: print(message, file=sys.stderr, flush=True)
    cache = PreparedData(args.artifacts_dir, snapshot["dataset_snapshot_id"], split, config,
                         enabled=not args.no_cache, rebuild=args.rebuild_cache, log=emit)
    history = split.train + split.validation
    if args.scope == "all":
        graphs = CheckpointedLightGCN(config.lightgcn_config, months=config.lightgcn_checkpoint_months, log=emit)
        cache.training("train", split.train, split.train_cutoff, graphs)
        cache.training("refit", history, split.validation_cutoff, graphs)
    for phase, past, future, cutoff in (
        ("validation", split.train, split.validation, split.train_cutoff),
        ("test", history, split.test, split.validation_cutoff),
    ):
        queries, _ = build_window_queries(past, future, config=config, phase=phase, cutoff=cutoff)
        cache.window(phase, past, queries)
    print(json.dumps(cache.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
