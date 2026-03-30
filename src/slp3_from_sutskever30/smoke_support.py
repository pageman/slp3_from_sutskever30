from __future__ import annotations

from datetime import datetime, timezone

from slp3_from_sutskever30.registry import get_chapters, get_orphaned_chapter_keys, get_unexpected_chapter_keys


def build_smoke_payload() -> dict[str, object]:
    chapters = get_chapters()
    results = []
    for spec in chapters:
        payload = spec.runner()
        results.append(
            {
                "key": spec.key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "source_papers": list(spec.source_papers),
                "payload_keys": sorted(payload.keys()),
            }
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chapter_count": len(chapters),
        "orphaned_chapters": get_orphaned_chapter_keys(),
        "unexpected_chapters": get_unexpected_chapter_keys(),
        "results": results,
    }
