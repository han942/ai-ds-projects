from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from rating_recsys.config import get_settings


class SettingsTests(unittest.TestCase):
    @patch("rating_recsys.config.load_dotenv_if_available")
    def test_read_only_database_command_does_not_require_hash_salt(
        self,
        _load_dotenv,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.invalid/postgres",
                "USER_HASH_SALT": "replace-with-a-long-random-secret",
            },
            clear=True,
        ):
            settings = get_settings(
                require_database=True,
                require_user_hash_salt=False,
            )

        self.assertEqual(
            settings.database_url,
            "postgresql://example.invalid/postgres",
        )

    @patch("rating_recsys.config.load_dotenv_if_available")
    def test_ingestion_still_rejects_example_hash_salt(self, _load_dotenv) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.invalid/postgres",
                "USER_HASH_SALT": "replace-with-a-long-random-secret",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be changed"):
                get_settings(require_database=True)


if __name__ == "__main__":
    unittest.main()
