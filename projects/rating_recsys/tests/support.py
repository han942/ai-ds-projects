"""Shared synthetic data and small experiment cases; no test module imports."""

from __future__ import annotations

import random
from dataclasses import replace
from datetime import date, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.candidate_models import CANDIDATE_MODELS
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import run_experiment


def _interaction(review_id: int, user: int, item: int, day: int, rating: float = 5.0):
    return Interaction(
        review_id=review_id,
        user_id=user,
        restaurant_id=item,
        event_date=date(2025, 1, 1) + timedelta(days=day),
        rating=rating,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{item}",
        region="서울" if item <= 15 else "부산",
    )


def synthetic_interactions() -> list[Interaction]:
    """Three taste clusters, staggered starts, and a few late-joining users."""

    rng = random.Random(7)
    clusters = {0: range(1, 11), 1: range(11, 21), 2: range(21, 31)}
    rows: list[Interaction] = []
    review_id = 1
    for user in range(1, 67):
        items = rng.sample(list(clusters[user % 3]), 6) + rng.sample(range(1, 31), 2)
        items = list(dict.fromkeys(items))
        day = rng.randint(0, 40) if user <= 60 else rng.randint(85, 95)
        for item in items:
            rows.append(
                _interaction(review_id, user, item, day, rng.choice([5, 5, 4, 4, 3, 2]))
            )
            review_id += 1
            day += rng.randint(3, 8)
    return rows


SMALL = ExperimentConfig(
    candidate_k=10,
    ranking_k=5,
    lightgcn_dimension=8,
    lightgcn_epochs=3,
    lightgcn_batch_size=64,
    lightgcn_checkpoint_months=1,
    num_leaves_grid=(4,),
    min_child_samples_grid=(1, 5),
    max_estimators=30,
    early_stopping_rounds=5,
    reference_estimators=10,
    reference_num_leaves=4,
    reference_min_child_samples=1,
    bootstrap_samples=50,
    n_jobs=1,
)


def run_small(rows, root: Path, **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    return run_experiment(
        rows,
        project_root=root,
        artifacts_root=root / "artifacts",
        config=kwargs.pop("config", SMALL),
        log=lambda _: None,
        **kwargs,
    )


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


CUISINE = {0: "국밥 순대", 1: "파스타 피자", 2: "초밥 회덮밥"}


def cuisine(item: int) -> int:
    return 0 if item <= 10 else 1 if item <= 20 else 2


def texts_for(rows) -> dict[int, str]:
    return {
        row.review_id: f"{CUISINE[cuisine(row.restaurant_id)]} 맛있어요 {row.restaurant_id}번"
        for row in rows
    }


# Every registered model needs one small config row, not a new test/CLI file.
# Parameters not listed here retain the model's default config values.
MODEL_CASES = {
    "lightgcn": {
        "dimension": 8, "layers": 2, "epochs": 12,
        "batch_size": 64, "learning_rate": 0.01,
    },
    "deepconn": {
        "embedding_dim": 8, "num_filters": 8, "latent_dim": 4, "fm_k": 2,
        "doc_length": 60, "max_review_length": 20, "batch_size": 32,
        "epochs": 3, "threads": 1, "dropout": 0.0, "learning_rate": 0.01,
    },
}


def available_models():
    models = []
    for model in CANDIDATE_MODELS.values():
        try:
            for package in model.packages:
                version(package)
        except PackageNotFoundError:
            continue
        models.append(model)
    return models


def tiny_grid(model):
    return tuple(dict.fromkeys(
        replace(config, **MODEL_CASES[model.name])
        for config in model.default_grid()
    ))


def tiny_cli_flags(model):
    flags = []
    values = dict(MODEL_CASES[model.name])
    first = tiny_grid(model)[0]
    values.update({name: getattr(first, name) for name in model.grid_parameters})
    for name, value in values.items():
        flag = "--max-epochs" if name == "epochs" else model.grid_parameters.get(
            name, (f"--{name.replace('_', '-')}", ())
        )[0]
        if isinstance(value, bool):
            flags.append(flag if value else f"--no-{flag[2:]}")
        else:
            flags.extend((flag, str(value)))
    # Keep the integration run to one model config; the full grid has its own tests.
    flags.extend({"deepconn": ("--variants", "bpr:relu")}.get(model.name, ()))
    return flags
