"""Completed-arm recovery validates evidence instead of overwriting it."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from rating_recsys.experiments.retrieval_resume import (
    atomic_write_json, check_completed_metrics, load_completed_arm, quarantine_uncommitted_bank,
)


def test_completed_metrics_must_match_reconstructed_scores():
    check_completed_metrics("arm", {"queries": 3, "recall_at_100": .5},
                            {"queries": 3, "recall_at_100": .5})
    for bad in ({"queries": 3, "recall_at_100": .9}, {"queries": 3},
                {"queries": 3, "recall_at_100": float("nan")}, None):
        with pytest.raises(ValueError, match="metrics differ"):
            check_completed_metrics("arm", bad, {"queries": 3, "recall_at_100": .5})


def test_progress_commit_preserves_old_json_if_replacement_fails(tmp_path, monkeypatch):
    import json
    from rating_recsys.experiments import retrieval_resume as resume

    path = tmp_path/"progress.json"
    atomic_write_json(path, {"arms": {"finished": True}})
    before = path.read_bytes()

    def interrupted_replace(*args):
        raise OSError("simulated interruption during commit")

    monkeypatch.setattr(resume.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(path, {"arms": {"finished": True, "new": True}})
    assert path.read_bytes() == before
    assert json.loads(path.read_text()) == {"arms": {"finished": True}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.json"]


def test_uncommitted_bank_is_preserved_but_finished_bank_cannot_be_replaced(tmp_path):
    bank = tmp_path/"review_bank.npz"
    bank.write_bytes(b"partially written bank")
    quarantined = quarantine_uncommitted_bank(tmp_path, {"arms": {}})
    assert not bank.exists() and quarantined.read_bytes() == b"partially written bank"
    assert quarantine_uncommitted_bank(tmp_path, {"arms": {}}) is None
    bank.write_bytes(b"committed bank")
    assert quarantine_uncommitted_bank(tmp_path, {"bank_file_sha256": "recorded"}) is None
    assert bank.read_bytes() == b"committed bank"
    with pytest.raises(ValueError, match="Completed arms lack"):
        quarantine_uncommitted_bank(tmp_path, {"arms": {"finished": {}}})
    assert bank.read_bytes() == b"committed bank"


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def completed(tmp_path):
    name = "graph_baseline_seed42"
    checkpoint = tmp_path / f"{name}.npz"
    checkpoint.write_bytes(b"saved checkpoint")
    users, items = np.asarray([10, 20]), np.asarray([1, 2, 3])
    scores = np.arange(6, dtype=np.float32).reshape(2, 3)
    score_path = tmp_path / f"{name}.scores.npz"
    np.savez_compressed(score_path, scores=scores, user_ids=users, item_ids=items)
    record = {"checkpoint": checkpoint.name, "checkpoint_sha256": _hash(checkpoint),
              "scores_sha256": _hash(score_path), "metadata": {"completed_epochs": 20},
              "curve": [{"epoch": 20}], "fit_seconds": 1.5}
    return tmp_path, {"arms": {name: record}}, name, users, items, scores


def test_unfinished_arm_returns_none_without_requiring_files(tmp_path):
    assert load_completed_arm(tmp_path, {"arms": {}}, "partial", [1], [2]) is None


def test_recovery_preserves_state_and_artifacts_and_returns_independent_record(completed):
    run, state, name, users, items, expected = completed
    original = deepcopy(state)
    files = {path.name: path.read_bytes() for path in run.iterdir()}
    scores, record = load_completed_arm(run, state, name, users, items)
    np.testing.assert_array_equal(scores, expected)
    assert record == state["arms"][name]
    record["metadata"]["completed_epochs"] = 0
    scores[:] = 0
    assert state == original
    assert files == {path.name: path.read_bytes() for path in run.iterdir()}


@pytest.mark.parametrize("artifact", ["checkpoint", "scores"])
def test_changed_or_missing_completed_artifact_is_rejected(completed, artifact):
    run, state, name, users, items, _ = completed
    path = run / (state["arms"][name]["checkpoint"] if artifact == "checkpoint"
                  else f"{name}.scores.npz")
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_completed_arm(run, state, name, users, items)
    path.unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        load_completed_arm(run, state, name, users, items)


@pytest.mark.parametrize("axis", ["user", "item"])
def test_changed_axis_order_is_rejected(completed, axis):
    run, state, name, users, items, _ = completed
    with pytest.raises(ValueError, match="score axes differ"):
        load_completed_arm(run, state, name, users[::-1] if axis == "user" else users,
                           items[::-1] if axis == "item" else items)


@pytest.mark.parametrize("scores", [np.zeros((3, 2)), np.full((2, 3), np.nan),
                                    np.full((2, 3), np.inf), np.ones((2, 3), dtype=complex)])
def test_rehashed_but_invalid_score_matrix_is_rejected(completed, scores):
    run, state, name, users, items, _ = completed
    path = run / f"{name}.scores.npz"
    np.savez_compressed(path, scores=scores, user_ids=users, item_ids=items)
    state["arms"][name]["scores_sha256"] = _hash(path)
    with pytest.raises(ValueError, match="score matrix is invalid"):
        load_completed_arm(run, state, name, users, items)


def test_rehashed_archive_with_missing_axes_is_rejected(completed):
    run, state, name, users, items, scores = completed
    path = run / f"{name}.scores.npz"
    np.savez_compressed(path, scores=scores)
    state["arms"][name]["scores_sha256"] = _hash(path)
    with pytest.raises(ValueError, match="archive is invalid"):
        load_completed_arm(run, state, name, users, items)


def test_incomplete_completion_record_is_rejected(completed):
    run, state, name, users, items, _ = completed
    del state["arms"][name]["scores_sha256"]
    with pytest.raises(ValueError, match="lacks checkpoint/score checksums"):
        load_completed_arm(run, state, name, users, items)


def test_rehashed_plain_npy_file_is_not_accepted_as_score_archive(completed):
    run, state, name, users, items, scores = completed
    path = run / f"{name}.scores.npz"
    with path.open("wb") as stream:
        np.save(stream, scores, allow_pickle=False)
    state["arms"][name]["scores_sha256"] = _hash(path)
    with pytest.raises(ValueError, match="archive is invalid"):
        load_completed_arm(run, state, name, users, items)


def test_checkpoint_outside_run_is_rejected(completed):
    run, state, name, users, items, _ = completed
    state["arms"][name]["checkpoint"] = "../other.npz"
    with pytest.raises(ValueError, match="local filename"):
        load_completed_arm(run, state, name, users, items)
