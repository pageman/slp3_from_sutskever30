#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.registry import get_chapters, get_orphaned_chapter_keys, get_unexpected_chapter_keys  # noqa: E402


def main() -> None:
    chapters = get_chapters()
    counts = Counter(spec.implementation_status for spec in chapters)
    payload = {
        "chapter_count": len(chapters),
        "counts": dict(counts),
        "chapters": [
            {
                "key": spec.key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "source_papers": list(spec.source_papers),
            }
            for spec in chapters
        ],
        "orphaned_chapters": get_orphaned_chapter_keys(),
        "unexpected_chapters": get_unexpected_chapter_keys(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
