"""Leakage-aware recommendation dataset split builders."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable

from rating_recsys.datasets.models import Interaction


@dataclass(frozen=True, slots=True)
class SeenUserSplit:
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]
    source_user_count: int
    eligible_user_count: int
    excluded_user_count: int
    history_1_2_user_count: int
    history_3_plus_user_count: int
    minimum_user_items: int

    def summary(self) -> dict[str, object]:
        return {
            "protocol": "per-user-chronological-leave-last-two-out",
            "minimum_unique_restaurants": self.minimum_user_items,
            "source_users": self.source_user_count,
            "eligible_seen_users": self.eligible_user_count,
            "excluded_users": self.excluded_user_count,
            "interactions": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "history_diagnostics": {
                "train_history_1_2_users": self.history_1_2_user_count,
                "train_history_3_plus_users": self.history_3_plus_user_count,
            },
        }


@dataclass(frozen=True, slots=True)
class GlobalTemporalSplit:
    train: tuple[Interaction, ...]
    validation: tuple[Interaction, ...]
    test: tuple[Interaction, ...]
    train_cutoff: date
    validation_cutoff: date
    validation_cohorts: dict[str, dict[str, int]]
    test_cohorts: dict[str, dict[str, int]]

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
            "validation_cohorts": self.validation_cohorts,
            "test_cohorts": self.test_cohorts,
        }


def _event_key(interaction: Interaction) -> tuple[date, int, int]:
    return (
        interaction.event_date,
        interaction.review_id,
        interaction.restaurant_id,
    )


def _global_event_key(interaction: Interaction) -> tuple[date, int, int, int]:
    return (
        interaction.event_date,
        interaction.user_id,
        interaction.review_id,
        interaction.restaurant_id,
    )


def _group_unique_user_items(
    interactions: Iterable[Interaction],
) -> dict[int, list[Interaction]]:
    grouped: dict[int, list[Interaction]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()

    for interaction in interactions:
        pair = (interaction.user_id, interaction.restaurant_id)
        if pair in seen_pairs:
            raise ValueError(
                "Interactions must contain only the first user/restaurant event; "
                f"duplicate pair found: {pair}"
            )
        seen_pairs.add(pair)
        grouped[interaction.user_id].append(interaction)

    for user_interactions in grouped.values():
        user_interactions.sort(key=_event_key)
    return grouped


def build_seen_user_split(
    interactions: Iterable[Interaction],
    *,
    minimum_user_items: int = 3,
) -> SeenUserSplit:
    """Build the primary seen-user chronological leave-last-two-out split."""

    if minimum_user_items < 3:
        raise ValueError("minimum_user_items must be at least 3")

    grouped = _group_unique_user_items(interactions)
    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    history_1_2_users = 0
    history_3_plus_users = 0
    excluded_users = 0

    for user_interactions in grouped.values():
        if len(user_interactions) < minimum_user_items:
            excluded_users += 1
            continue

        user_train = user_interactions[:-2]
        train.extend(user_train)
        validation.append(user_interactions[-2])
        test.append(user_interactions[-1])

        if len(user_train) <= 2:
            history_1_2_users += 1
        else:
            history_3_plus_users += 1

    train.sort(key=_global_event_key)
    validation.sort(key=_global_event_key)
    test.sort(key=_global_event_key)
    eligible_users = history_1_2_users + history_3_plus_users

    return SeenUserSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        source_user_count=len(grouped),
        eligible_user_count=eligible_users,
        excluded_user_count=excluded_users,
        history_1_2_user_count=history_1_2_users,
        history_3_plus_user_count=history_3_plus_users,
        minimum_user_items=minimum_user_items,
    )


def _quantile_date(sorted_interactions: list[Interaction], fraction: float) -> date:
    index = max(0, math.ceil(len(sorted_interactions) * fraction) - 1)
    return sorted_interactions[index].event_date


def _cohort_summary(
    interactions: tuple[Interaction, ...],
    train_history: dict[int, int],
) -> dict[str, dict[str, int]]:
    cohort_users: dict[str, set[int]] = {
        "seen": set(),
        "new": set(),
        "history_1_2": set(),
        "history_3_plus": set(),
    }
    cohort_interactions = dict.fromkeys(cohort_users, 0)

    for interaction in interactions:
        history_count = train_history.get(interaction.user_id, 0)
        if history_count == 0:
            buckets = ("new",)
        elif history_count <= 2:
            buckets = ("seen", "history_1_2")
        else:
            buckets = ("seen", "history_3_plus")

        for bucket in buckets:
            cohort_users[bucket].add(interaction.user_id)
            cohort_interactions[bucket] += 1

    return {
        bucket: {
            "users": len(cohort_users[bucket]),
            "interactions": cohort_interactions[bucket],
        }
        for bucket in cohort_users
    }


def build_global_temporal_split(
    interactions: Iterable[Interaction],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> GlobalTemporalSplit:
    """Build a fixed-cutoff deployment benchmark and seen/new cohort audit."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    sorted_interactions = sorted(interactions, key=_global_event_key)
    if len(sorted_interactions) < 3:
        raise ValueError("At least three interactions are required")

    train_cutoff = _quantile_date(sorted_interactions, train_fraction)
    validation_cutoff = _quantile_date(
        sorted_interactions,
        train_fraction + validation_fraction,
    )
    if train_cutoff >= validation_cutoff:
        raise ValueError(
            "Temporal cutoffs collapsed onto the same date; choose different "
            "fractions or provide a wider date range"
        )

    train = tuple(
        item for item in sorted_interactions if item.event_date <= train_cutoff
    )
    validation = tuple(
        item
        for item in sorted_interactions
        if train_cutoff < item.event_date <= validation_cutoff
    )
    test = tuple(
        item for item in sorted_interactions if item.event_date > validation_cutoff
    )

    train_history: dict[int, int] = defaultdict(int)
    for interaction in train:
        train_history[interaction.user_id] += 1

    return GlobalTemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
        validation_cohorts=_cohort_summary(validation, train_history),
        test_cohorts=_cohort_summary(test, train_history),
    )
