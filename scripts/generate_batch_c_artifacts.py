#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.batch_c_artifacts import write_batch_c_artifacts  # noqa: E402


def main() -> None:
    manifest_path = write_batch_c_artifacts()
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
