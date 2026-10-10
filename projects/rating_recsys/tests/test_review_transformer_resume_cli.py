"""CLI-level crash recovery with real tiny training and deterministic fake E5.

The tests exercise publication failures rather than orderly epoch boundaries.
Only the E5 formatter/cache and environment provenance are replaced; model
training, validation ranking, checkpoint selection and CLI persistence run.
"""
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments import review_transformer_cli as cli
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.snapshot import dataset_digest
from rating_recsys.retrieval.review_embeddings import ProfileFormatter
from rating_recsys.retrieval.review_transformer import ReviewTransformer, ReviewTransformerConfig


class DeterministicBlockCache:
    def __init__(self, *args, **kwargs):
        self.encoder = None
        self.usage = {"api_requests": 0}
        self.identity = {"backend": "deterministic-test-vectors"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def missing_count(self, documents):
        return len(set(documents))

    def embed(self, documents):
        output = {}
        for document in documents:
            seed = int.from_bytes(hashlib.sha256(document.encode()).digest()[:4], "little")
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            output[document] = vector / np.linalg.norm(vector)
        return output


@pytest.fixture
def tiny_cli(tmp_path, monkeypatch):
    def formatter(config, directory):
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        return ProfileFormatter(config, directory, tokenizer=tokenizer)

    def config(**kwargs):
        return ReviewTransformerConfig(width=16, heads=2, layers=1,
            max_user_reviews=3, max_item_reviews=4, max_blocks=2, block_tokens=64,
            negative_samples=2, dropout=.1, id_dropout=.2, **kwargs)

    monkeypatch.setattr(cli, "ReviewTransformerConfig", config)
    monkeypatch.setattr(cli, "ProfileFormatter", formatter)
    monkeypatch.setattr(cli, "ReviewBlockCache", DeterministicBlockCache)
    monkeypatch.setattr(cli, "code_manifest", lambda *a, **kw:
                        {"git_commit": "tiny-cli-fixture", "git_diff": "", "git_dirty": False})

    def prepare(*, validation_rating=5):
        def row(rid, user, item, month, rating=5):
            return Interaction(rid, user, item, date(2025, month, 10), rating,
                               "exact", "synthetic", "region")
        rows = [row(i, i, i * 10, 1) for i in range(1, 7)]
        rows += [row(7, 1, 20, 4), row(8, 1, 30, 4, 3), row(9, 1, 40, 4, 1),
                 row(10, 2, 50, 4), row(11, 3, 60, 4), row(12, 4, 10, 4),
                 row(13, 5, 20, 4), row(14, 6, 30, 4)]
        rows += [row(15 + i, i + 1, item, 7, validation_rating)
                 for i, item in enumerate((50, 60, 10, 20, 30, 40))]
        rows += [row(21, 7, 10, 10), row(22, 8, 20, 10)]
        experiment = ExperimentConfig(train_fraction=.6, validation_fraction=.3,
            ranker_training_mode="window", satisfaction_mode="history-aware")
        snapshot = tmp_path / "snapshot.jsonl"
        with snapshot.open("w") as stream:
            for r in rows:
                stream.write(json.dumps({"review_id": r.review_id, "user_id": r.user_id,
                    "restaurant_id": r.restaurant_id, "event_date": r.event_date.isoformat(),
                    "rating": r.rating, "reviewed_at_precision": r.reviewed_at_precision,
                    "restaurant_name": r.restaurant_name, "region": r.region}) + "\n")
        with snapshot.with_name("snapshot.reviews.jsonl").open("w") as stream:
            for r in rows:
                stream.write(json.dumps({"review_id": r.review_id,
                    "review_text": f"리뷰{r.review_id} 국물은 좋지만 가격은 비쌌다."}) + "\n")
        baseline = tmp_path / "baseline"
        baseline.mkdir(exist_ok=True)
        (baseline / "manifest.json").write_text(json.dumps({"config": experiment.to_dict(),
            "snapshot": {"dataset_snapshot_id": dataset_digest(rows)}}))
        split = build_global_temporal_split(rows, train_fraction=.6, validation_fraction=.3)
        queries, _ = build_window_queries(split.train, split.validation, config=experiment,
                                          phase="validation", cutoff=split.train_cutoff)
        catalog = sorted({r.restaurant_id for r in split.train})
        with (baseline / "candidates_validation.jsonl").open("w") as stream:
            for q in queries:
                seen = {r.restaurant_id for r in q.history}
                stream.write(json.dumps({"query_id": q.query_id,
                    "c5_c1_lightgcn_rrf": [item for item in catalog if item not in seen]}) + "\n")

        def arguments(artifacts, *, epochs=3, patience=None):
            args = ["--snapshot", str(snapshot), "--baseline-run", str(baseline),
                    "--artifacts-dir", str(artifacts), "--cache-path", str(tmp_path / "cache.sqlite"),
                    "--local-model-cache", str(tmp_path / "models"), "--epochs", str(epochs),
                    "--batch-size", "4", "--threads", "1", "--candidate-k", "100"]
            if patience is not None:
                args += ["--min-epochs", "2", "--patience", str(patience)]
            return args
        return arguments
    return prepare


def only_run(artifacts):
    runs = list(artifacts.iterdir())
    assert len(runs) == 1
    return runs[0]


def load_last(run):
    return torch.load(run / "training_last.pt", map_location="cpu", weights_only=False)


def test_cli_recovers_progress_ahead_of_atomic_checkpoint_and_matches_uninterrupted(
        tmp_path, monkeypatch, tiny_cli):
    arguments = tiny_cli()
    control_dir = tmp_path / "control"
    interrupted_dir = tmp_path / "interrupted"
    control = cli.main(arguments(control_dir))
    original_replace = Path.replace
    publications = 0

    def interrupt_second_training_publication(path, target):
        nonlocal publications
        if path.name == "training_last.pt.tmp":
            publications += 1
            if publications == 2:
                # Callback has already published evaluation progress. A partial
                # uncommitted candidate temporary file must also be ignored.
                path.with_name("candidates_validation_epoch_099.jsonl.tmp").write_text('{"partial":')
                raise KeyboardInterrupt("crash between validation and checkpoint commit")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_second_training_publication)
        with pytest.raises(KeyboardInterrupt):
            cli.main(arguments(interrupted_dir))
    run = only_run(interrupted_dir)
    progress = json.loads((run / "progress.json").read_text())
    assert len(progress["epochs"]) == 2 and progress["status"] == "interrupted"
    committed, epoch = cli.resume_state(run)
    assert epoch == 1 and len(committed["epochs"]) == 1
    assert (run / committed["candidates_file"]).is_file()

    recovered = cli.main(arguments(interrupted_dir) + ["--resume-run", str(run)])
    assert recovered["status"] == "complete" and recovered["stopping"]["executed_epochs"] == 3
    assert recovered["selected_epoch"] == control["selected_epoch"]
    control_last, resumed_last = load_last(only_run(control_dir)), load_last(run)
    assert control_last["epoch"] == resumed_last["epoch"] == 3
    for name, parameter in control_last["network"].items():
        assert torch.equal(parameter, resumed_last["network"][name]), name
    assert control_last["optimizer"]["param_groups"] == resumed_last["optimizer"]["param_groups"]
    for parameter, moments in control_last["optimizer"]["state"].items():
        for name, value in moments.items():
            assert torch.equal(value, resumed_last["optimizer"]["state"][parameter][name])
    for left, right in zip(control_last["history"], resumed_last["history"]):
        assert {k:v for k,v in left.items() if k != "seconds"} == {k:v for k,v in right.items() if k != "seconds"}
    assert torch.equal(control_last["torch_rng"], resumed_last["torch_rng"])
    assert control_last["numpy_rng"] == resumed_last["numpy_rng"]
    assert (only_run(control_dir) / "candidates_validation.jsonl").read_bytes() == (run / "candidates_validation.jsonl").read_bytes()
    assert recovered["selected_validation"] == control["selected_validation"]
    assert recovered["api_requests"] == 0 and not recovered["test_evaluated"]


@pytest.mark.parametrize("boundary", ["epoch_budget", "patience"])
def test_cli_finalizes_committed_stopping_epoch_without_another_fit(
        boundary, tmp_path, monkeypatch, tiny_cli):
    # All-zero validation relevance fixes NDCG ties for the patience case,
    # without altering actual training data, optimizer or model computation.
    arguments = tiny_cli(validation_rating=1 if boundary == "patience" else 5)
    artifacts = tmp_path / "stopping"
    args = arguments(artifacts, epochs=5 if boundary == "patience" else 2,
                     patience=1 if boundary == "patience" else None)
    original_write = cli.write_json
    failed = False

    def interrupt_final_complete(path, value):
        nonlocal failed
        if path.name == "progress.json" and value.get("status") == "complete" and not failed:
            failed = True
            raise KeyboardInterrupt("crash after committed stopping epoch")
        return original_write(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(cli, "write_json", interrupt_final_complete)
        with pytest.raises(KeyboardInterrupt):
            cli.main(args)
    run = only_run(artifacts)
    committed, epoch = cli.resume_state(run)
    assert epoch == len(committed["epochs"]) == 2
    before = (run / "training_last.pt").read_bytes()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A committed stopping epoch must finalize without additional training")
    with monkeypatch.context() as patch:
        patch.setattr(ReviewTransformer, "fit", forbidden_fit)
        result = cli.main(args + ["--resume-run", str(run)])
    assert result["status"] == "complete"
    assert result["stopping"]["executed_epochs"] == 2
    assert result["stopping"]["reason"] == (
        "validation patience exhausted" if boundary == "patience" else "epoch budget exhausted")
    assert (run / "training_last.pt").read_bytes() == before
    assert len(result["epochs"]) == 2 and not result["test_evaluated"]
