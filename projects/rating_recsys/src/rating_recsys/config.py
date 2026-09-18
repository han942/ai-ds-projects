"""Environment-backed application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_url: str
    user_hash_salt: str
    input_dir: Path


def load_dotenv_if_available() -> None:
    """Load the project-local .env without making it a hard import dependency."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_settings(*, require_database: bool = True) -> Settings:
    load_dotenv_if_available()

    database_url = os.getenv("DATABASE_URL", "").strip()
    user_hash_salt = os.getenv("USER_HASH_SALT", "").strip()
    input_dir_value = os.getenv(
        "RECSYS_INPUT_DIR",
        "legacy/v1_rating_prediction/crawled_data",
    )
    input_dir = Path(input_dir_value)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    errors: list[str] = []
    if require_database and not database_url:
        errors.append("DATABASE_URL is required")
    if require_database and not user_hash_salt:
        errors.append("USER_HASH_SALT is required")
    if require_database and user_hash_salt == "replace-with-a-long-random-secret":
        errors.append("USER_HASH_SALT must be changed from the example value")
    if errors:
        raise ValueError("; ".join(errors))

    return Settings(
        database_url=database_url,
        user_hash_salt=user_hash_salt,
        input_dir=input_dir,
    )
