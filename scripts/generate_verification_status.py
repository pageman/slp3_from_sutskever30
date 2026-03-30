#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_DIR = ROOT / "observability"
JSON_PATH = OUT_DIR / "verification.json"
YAML_PATH = OUT_DIR / "verification.yaml"
SQLITE_PATH = OUT_DIR / "verification.sqlite"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.telemetry import build_telemetry_payload, write_telemetry_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repo-level verification telemetry artifacts.")
    parser.add_argument("--run-checks", action="store_true", help="Execute smoke, pytest, and survey before writing artifacts.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_telemetry_payload(run_live_checks=args.run_checks)
    write_telemetry_artifacts(JSON_PATH, YAML_PATH, SQLITE_PATH, payload)
    print(f"wrote {JSON_PATH}")
    print(f"wrote {YAML_PATH}")
    print(f"wrote {SQLITE_PATH}")


if __name__ == "__main__":
    main()
