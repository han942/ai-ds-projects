"""Shared immutable records for recommendation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Interaction:
    """A user's first observed interaction with one restaurant.

    The target review text and its aspect ratings are deliberately absent. Rating
    is retained only as an outcome label for later relevance construction.
    """

    review_id: int
    user_id: int
    restaurant_id: int
    event_date: date
    rating: float
    reviewed_at_precision: str
    restaurant_name: str
    region: str
