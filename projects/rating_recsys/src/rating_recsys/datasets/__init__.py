"""DB-backed modeling datasets and leakage-aware split builders."""

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.datasets.split import (
    build_global_temporal_split,
    build_seen_user_split,
)

__all__ = [
    "Interaction",
    "InteractionRepository",
    "build_global_temporal_split",
    "build_seen_user_split",
]
