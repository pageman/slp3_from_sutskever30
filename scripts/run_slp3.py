#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.registry import get_chapter_map, get_chapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone NumPy SLP3 chapters.")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--chapter", type=str)
    args = parser.parse_args()
    if args.list:
        for spec in get_chapters():
            source = ",".join(str(value) for value in spec.source_papers) if spec.source_papers else "-"
            print(f"{spec.key:>2}  src={source:>5}  {spec.title}")
        return
    if not args.chapter:
        parser.error("pass --list or --chapter")
    spec = get_chapter_map()[args.chapter]
    print(json.dumps(spec.runner(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
