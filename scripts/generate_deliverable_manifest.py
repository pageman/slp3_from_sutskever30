#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT_PATH = ROOT / "research" / "DELIVERABLE_MANIFEST.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.deliverable_manifest import build_deliverable_manifest, render_deliverable_manifest  # noqa: E402


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render_deliverable_manifest(build_deliverable_manifest()))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
