"""Minimal ordered SQL migration runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.db.connection import connect


MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


def apply_migrations(database_url: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    migration_files = sorted(migrations_dir.glob("*.sql"))
    if not migration_files:
        raise FileNotFoundError(f"No SQL migrations found in {migrations_dir}")

    applied_now: list[str] = []
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS recsys")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recsys.schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.commit()

            cur.execute("SELECT version FROM recsys.schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

            for path in migration_files:
                if path.name in applied:
                    continue
                sql = path.read_text(encoding="utf-8")
                with conn.transaction():
                    cur.execute(sql, prepare=False)
                    cur.execute(
                        "INSERT INTO recsys.schema_migrations (version) VALUES (%s)",
                        (path.name,),
                    )
                applied_now.append(path.name)

    return applied_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=MIGRATIONS_DIR,
        help="Directory containing ordered .sql migrations",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings(
        require_database=True,
        require_user_hash_salt=False,
    )
    applied = apply_migrations(settings.database_url, args.migrations_dir)
    if applied:
        print("Applied migrations:")
        for version in applied:
            print(f"- {version}")
    else:
        print("Database schema is already up to date.")


if __name__ == "__main__":
    main()
