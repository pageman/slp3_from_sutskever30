#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_DIR = ROOT / "observability"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.artifacts import write_sqlite_table_with_backup, write_text_with_backup  # noqa: E402
from slp3_from_sutskever30.smoke_support import build_smoke_payload  # noqa: E402


def write_smoke_artifacts(report: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_text_with_backup(OUT_DIR / "smoke_test.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_sqlite_table_with_backup(
        OUT_DIR / "smoke_test.sqlite",
        table_name="results",
        payload=report,
        item_key="results",
        item_columns=("key", "title", "implementation_status", "source_papers", "payload_keys"),
    )


def main() -> None:
    report = build_smoke_payload()
    write_smoke_artifacts(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
