"""Canonical dataset snapshots and run-environment manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.queries import global_interaction_key


SNAPSHOT_SCHEMA_VERSION = "first-interaction-v1"


def interaction_record(item: Interaction) -> dict[str, object]:
    record = asdict(item)
    record["event_date"] = item.event_date.isoformat()
    return record


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def dataset_digest(interactions: Iterable[Interaction]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{SNAPSHOT_SCHEMA_VERSION}\n".encode())
    for item in sorted(interactions, key=global_interaction_key):
        digest.update(canonical_json(interaction_record(item)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_snapshot(
    interactions: Iterable[Interaction],
    destination: Path,
) -> dict[str, object]:
    ordered = tuple(sorted(interactions, key=global_interaction_key))
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_digest = hashlib.sha256()
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            line = canonical_json(interaction_record(item)) + "\n"
            handle.write(line)
            file_digest.update(line.encode("utf-8"))

    dates = [item.event_date for item in ordered]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset_snapshot_id": dataset_digest(ordered),
        "artifact_sha256": file_digest.hexdigest(),
        "artifact": destination.name,
        "interactions": len(ordered),
        "users": len({item.user_id for item in ordered}),
        "restaurants": len({item.restaurant_id for item in ordered}),
        "minimum_event_date": min(dates).isoformat() if dates else None,
        "maximum_event_date": max(dates).isoformat() if dates else None,
    }


def _git_output(project_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def code_manifest(project_root: Path, *, allow_dirty: bool) -> dict[str, object]:
    commit = _git_output(project_root, "rev-parse", "HEAD")
    status = _git_output(project_root, "status", "--porcelain")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "Refusing a non-reproducible run from a dirty worktree; commit changes "
            "or call the pipeline with allow_dirty=True to capture its diff"
        )
    diff = ""
    if dirty:
        diff = (_git_output(project_root, "diff", "--binary") or "") + _untracked_diff(
            project_root
        )
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status": status or "",
        "git_diff": diff,
    }


def _untracked_diff(project_root: Path) -> str:
    """``git diff`` of new, not ignored files under the project, outside artifacts/.

    ``git diff`` alone skips untracked files, so a run using a new module would
    record a diff that cannot rebuild its code. Paths are relative to the
    repository root, like the tracked part of the diff.
    """

    top = _git_output(project_root, "rev-parse", "--show-toplevel")
    listed = _git_output(
        project_root, "ls-files", "--others", "--exclude-standard", "--full-name"
    )
    if not top or not listed:
        return ""
    prefix = _git_output(project_root, "rev-parse", "--show-prefix") or ""
    parts = []
    for path in sorted(listed.splitlines()):
        if path.removeprefix(prefix).startswith("artifacts/"):
            continue
        # --no-index exits 1 when the files differ, which is always the case here.
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "--", "/dev/null", path],
            cwd=top,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode in (0, 1) and result.stdout:
            parts.append(result.stdout if result.stdout.endswith("\n") else result.stdout + "\n")
    return ("\n" if parts else "") + "".join(parts)


def environment_manifest() -> dict[str, object]:
    packages: set[str] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.add(f"{name}=={distribution.version}")

    conda_packages: list[str] = []
    conda_metadata = Path(sys.prefix) / "conda-meta"
    if conda_metadata.is_dir():
        for package_file in sorted(conda_metadata.glob("*.json")):
            package = json.loads(package_file.read_text(encoding="utf-8"))
            conda_packages.append(
                f"{package['name']}={package['version']}={package.get('build', '')}"
            )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(packages, key=str.casefold),
        "conda_packages": conda_packages,
    }


def load_snapshot(path: Path) -> list[Interaction]:
    """Read a ``write_snapshot`` JSONL file back into interactions."""

    from datetime import date

    interactions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["event_date"] = date.fromisoformat(row["event_date"])
            interactions.append(Interaction(**row))
    return interactions


def freeze_snapshot(
    interactions: Iterable[Interaction],
    snapshots_dir: Path,
) -> tuple[Path, dict[str, object]]:
    """Store the input once as ``<snapshot_id[:16]>.jsonl`` and return its metadata.

    The file content is canonical, so the same interactions always produce the
    same file and rewriting an existing snapshot does not change it.
    """

    staging = snapshots_dir / ".staging.jsonl"
    meta = write_snapshot(interactions, staging)
    destination = snapshots_dir / f"{str(meta['dataset_snapshot_id'])[:16]}.jsonl"
    staging.replace(destination)
    meta["artifact"] = destination.name
    return destination, meta


# ---------------------------------------------------------------------------
# Review text side file (DeepCoNN)
# ---------------------------------------------------------------------------

REVIEW_TEXT_SCHEMA_VERSION = "review-text-v1"


def review_texts_path(snapshot_path: Path) -> Path:
    """``<snapshot>.reviews.jsonl`` next to the interaction snapshot."""

    return snapshot_path.with_name(f"{snapshot_path.stem}.reviews.jsonl")


def write_review_texts(
    texts: dict[int, str | None], destination: Path
) -> dict[str, object]:
    """Store review text by review id, sorted, so the file is canonical.

    Interactions stay text-free. The text is only read by models that build
    documents from reviews dated up to their own cutoff.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    staging = destination.with_name(destination.name + ".staging")
    with staging.open("w", encoding="utf-8", newline="\n") as handle:
        for review_id in sorted(texts):
            line = canonical_json({"review_id": review_id, "review_text": texts[review_id]}) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    staging.replace(destination)
    return _review_texts_meta(destination.name, texts, digest.hexdigest())


def load_review_texts(
    path: Path, interactions: Iterable[Interaction]
) -> tuple[dict[int, str | None], dict[str, object]]:
    """Read the side file and check it covers every interaction's review."""

    digest = hashlib.sha256()
    texts: dict[int, str | None] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            digest.update(line.encode("utf-8"))
            row = json.loads(line)
            texts[int(row["review_id"])] = row["review_text"]
    missing = {item.review_id for item in interactions} - set(texts)
    if missing:
        raise ValueError(f"{path} has no text row for {len(missing)} reviews, e.g. {sorted(missing)[:3]}")
    return texts, _review_texts_meta(path.name, texts, digest.hexdigest())


def _review_texts_meta(name: str, texts: dict[int, str | None], sha256: str) -> dict[str, object]:
    return {
        "schema_version": REVIEW_TEXT_SCHEMA_VERSION,
        "artifact": name,
        "artifact_sha256": sha256,
        "reviews": len(texts),
        "non_empty": sum(bool((text or "").strip()) for text in texts.values()),
    }
