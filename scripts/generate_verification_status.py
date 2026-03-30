#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.observability_paths import get_observability_dir  # noqa: E402
from slp3_from_sutskever30.telemetry import build_telemetry_payload, write_telemetry_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repo-level verification telemetry artifacts.")
    parser.add_argument("--run-checks", action="store_true", help="Execute smoke, pytest, and survey before writing artifacts.")
    args = parser.parse_args()

    out_dir = get_observability_dir(ROOT)
    json_path = out_dir / "verification.json"
    yaml_path = out_dir / "verification.yaml"
    sqlite_path = out_dir / "verification.sqlite"

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_telemetry_payload(run_live_checks=args.run_checks)
    write_telemetry_artifacts(json_path, yaml_path, sqlite_path, payload)
    print(f"wrote {json_path}")
    print(f"wrote {yaml_path}")
    print(f"wrote {sqlite_path}")


if __name__ == "__main__":
    main()
