#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.circleci_artifacts import build_circleci_payload, write_circleci_artifacts  # noqa: E402
from slp3_from_sutskever30.observability_paths import get_observability_dir  # noqa: E402


def main() -> None:
    out_dir = get_observability_dir(ROOT)
    json_path = out_dir / "circleci_run.json"
    sqlite_path = out_dir / "circleci_run.sqlite"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_circleci_payload()
    write_circleci_artifacts(json_path, sqlite_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {sqlite_path}")


if __name__ == "__main__":
    main()
