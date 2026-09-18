"""PostgreSQL connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection


@contextmanager
def connect(database_url: str) -> Iterator["Connection"]:
    """Open one short-lived PostgreSQL session for a pipeline command."""

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Run `pip install -e '.[dev]'` first."
        ) from exc

    with psycopg.connect(
        database_url,
        autocommit=False,
        application_name="rating-recsys-pipeline",
        prepare_threshold=None,
    ) as connection:
        yield connection
