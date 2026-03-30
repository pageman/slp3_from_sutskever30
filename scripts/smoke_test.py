#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slp3_from_sutskever30.registry import get_chapters, get_orphaned_chapter_keys, get_unexpected_chapter_keys  # noqa: E402


def main() -> None:
    chapters = get_chapters()
    results = []
    for spec in chapters:
        payload = spec.runner()
        results.append(
            {
                "key": spec.key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "payload_keys": sorted(payload.keys()),
            }
        )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chapter_count": len(chapters),
        "orphaned_chapters": get_orphaned_chapter_keys(),
        "unexpected_chapters": get_unexpected_chapter_keys(),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
