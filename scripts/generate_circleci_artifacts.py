#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_DIR = ROOT / "observability"
JSON_PATH = OUT_DIR / "circleci_run.json"
SQLITE_PATH = OUT_DIR / "circleci_run.sqlite"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.circleci_artifacts import build_circleci_payload, write_circleci_artifacts  # noqa: E402


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_circleci_payload()
    write_circleci_artifacts(JSON_PATH, SQLITE_PATH, payload)
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SQLITE_PATH}")


if __name__ == "__main__":
    main()
