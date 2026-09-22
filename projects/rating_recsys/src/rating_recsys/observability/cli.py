"""Launch the Streamlit recommendation explorer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is required; install `pip install -e '.[experiment]'`"
        ) from exc

    environment = os.environ.copy()
    environment["RATING_RECSYS_ARTIFACTS_DIR"] = str(args.artifacts_dir.resolve())
    app_path = Path(__file__).with_name("app.py")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(args.port),
                "--server.address",
                args.address,
            ],
            env=environment,
        )
    )


if __name__ == "__main__":
    main()
