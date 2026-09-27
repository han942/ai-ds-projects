"""Global date-cutoff split for leakage-free recommendation evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from rating_recsys.datasets.models import Interaction


@dataclass(frozen=True, slots=True)
class GlobalTemporalSplit:
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]
    train_cutoff: date
    validation_cutoff: date
    validation_users: dict[str, int]
    test_users: dict[str, int]

    def summary(self) -> dict[str, object]:
        return {
            "protocol": "global-temporal",
            "cutoffs": {
                "train_through": self.train_cutoff.isoformat(),
                "validation_through": self.validation_cutoff.isoformat(),
            },
            "interactions": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            # seen = the user has history before the window; new = no history.
            "window_users": {
                "validation": self.validation_users,
                "test": self.test_users,
            },
        }


def _global_event_key(interaction: Interaction) -> tuple[date, int, int, int]:
    return (
        interaction.event_date,
        interaction.user_id,
        interaction.review_id,
        interaction.restaurant_id,
    )


def _quantile_date(sorted_interactions: list[Interaction], fraction: float) -> date:
    index = max(0, math.ceil(len(sorted_interactions) * fraction) - 1)
    return sorted_interactions[index].event_date


def _window_users(
    window: tuple[Interaction, ...],
    history: Iterable[Interaction],
) -> dict[str, int]:
    known = {item.user_id for item in history}
    users = {item.user_id for item in window}
    return {"seen": len(users & known), "new": len(users - known)}


def build_global_temporal_split(
    interactions: Iterable[Interaction],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> GlobalTemporalSplit:
    """Split every user at the same two dates taken from interaction quantiles.

    Train is ``<= T1``, validation is ``(T1, T2]`` and test is ``> T2``.
    Validation users are seen when they have train history; test users are
    seen when they have train or validation history.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    sorted_interactions = sorted(interactions, key=_global_event_key)
    if len(sorted_interactions) < 3:
        raise ValueError("At least three interactions are required")
    pairs: set[tuple[int, int]] = set()
    for item in sorted_interactions:
        pair = (item.user_id, item.restaurant_id)
        if pair in pairs:
            raise ValueError(
                "Interactions must contain only the first user/restaurant event; "
                f"duplicate pair found: {pair}"
            )
        pairs.add(pair)

    train_cutoff = _quantile_date(sorted_interactions, train_fraction)
    validation_cutoff = _quantile_date(
        sorted_interactions, train_fraction + validation_fraction
    )
    if train_cutoff >= validation_cutoff:
        raise ValueError(
            "Temporal cutoffs collapsed onto the same date; choose different "
            "fractions or provide a wider date range"
        )

    grouped: dict[str, list[Interaction]] = defaultdict(list)
    for item in sorted_interactions:
        if item.event_date <= train_cutoff:
            grouped["train"].append(item)
        elif item.event_date <= validation_cutoff:
            grouped["validation"].append(item)
        else:
            grouped["test"].append(item)
    train = tuple(grouped["train"])
    validation = tuple(grouped["validation"])
    test = tuple(grouped["test"])
    return GlobalTemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
        validation_users=_window_users(validation, train),
        test_users=_window_users(test, train + validation),
    )
