"""Validate completed retrieval arms before reusing their saved score matrices.

Partial arms have no entry in ``state['arms']`` and are restarted by the caller.
This helper never loads a model, changes artifacts, or updates progress. The
caller rebuilds rankings from the returned scores under its current checks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


def atomic_write_json(path, value):
    """Keep the last committed progress readable if saving is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def quarantine_uncommitted_bank(run, state):
    """Preserve an interrupted bank write and let the caller prepare it again."""
    run = Path(run)
    bank = run / "review_bank.npz"
    if not bank.exists() or "bank_file_sha256" in state:
        return None
    if state.get("arms"):
        raise ValueError("Completed arms lack a committed review bank checksum")
    from datetime import datetime, timezone
    target = run / ("review_bank.uncommitted-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".npz")
    bank.replace(target)
    return target


def check_completed_metrics(name, recorded, recomputed):
    """Reject stale progress metrics before combining them with new analyses."""
    if not isinstance(recorded, dict) or set(recorded) != set(recomputed):
        raise ValueError(f"Completed arm metrics differ from restored scores: {name}")
    for key, current in recomputed.items():
        previous = recorded[key]
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            agrees = (isinstance(previous, (int, float)) and not isinstance(previous, bool)
                      and bool(np.isclose(previous, current, rtol=1e-12, atol=1e-12)))
        else:
            agrees = previous == current
        if not agrees:
            raise ValueError(f"Completed arm metrics differ from restored scores: {name}/{key}")


def _file_hash(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"Completed arm artifact is missing or unreadable: {path.name}") from error
    return digest.hexdigest()


def load_completed_arm(run, state, name, user_ids, item_ids):
    """Return ``(scores, copied_arm_record)`` or ``None`` for an unfinished arm.

    Both files must match the checksums committed in progress, and score axes
    must match the caller's current population and catalog in the same order.
    A recorded completion with incomplete/corrupt artifacts raises ValueError;
    it must not silently turn into new training or overwrite existing evidence.
    """
    arms = state.get("arms", {})
    if not isinstance(arms, dict):
        raise ValueError("Completed arm registry must be a mapping")
    if name not in arms:
        return None
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Completed arm name must be a local filename stem")
    record = arms[name]
    if not isinstance(record, dict):
        raise ValueError(f"Completed arm record is invalid: {name}")
    required = ("checkpoint", "checkpoint_sha256", "scores_sha256")
    if any(not isinstance(record.get(key), str) or not record[key] for key in required):
        raise ValueError(f"Completed arm record lacks checkpoint/score checksums: {name}")
    checkpoint = record["checkpoint"]
    if Path(checkpoint).name != checkpoint or checkpoint in (".", ".."):
        raise ValueError(f"Completed arm checkpoint must be a local filename: {name}")
    run = Path(run)
    if _file_hash(run / checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"Completed arm checkpoint checksum mismatch: {name}")
    score_path = run / f"{name}.scores.npz"
    if _file_hash(score_path) != record["scores_sha256"]:
        raise ValueError(f"Completed arm score checksum mismatch: {name}")
    expected_users, expected_items = np.asarray(user_ids), np.asarray(item_ids)
    if expected_users.ndim != 1 or expected_items.ndim != 1:
        raise ValueError("Requested score axes must be one-dimensional")
    try:
        with np.load(score_path, allow_pickle=False) as saved:
            scores = saved["scores"].copy()
            saved_users, saved_items = saved["user_ids"], saved["item_ids"]
            if (not np.array_equal(saved_users, expected_users)
                    or not np.array_equal(saved_items, expected_items)):
                raise ValueError(f"Completed arm score axes differ: {name}")
    except (OSError, KeyError, BadZipFile, EOFError, TypeError) as error:
        raise ValueError(f"Completed arm score archive is invalid: {name}") from error
    if (scores.shape != (len(expected_users), len(expected_items))
            or scores.dtype.kind not in "iuf" or not np.isfinite(scores).all()):
        raise ValueError(f"Completed arm score matrix is invalid: {name}")
    return scores, deepcopy(record)
