from __future__ import annotations

from typing import Any


def build_chapter_payload(
    *,
    chapter: str,
    implementation_status: str,
    core_outputs: dict[str, Any],
    metrics: dict[str, Any],
    failure_modes: list[dict[str, Any]],
    chapter_notes: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chapter": chapter,
        "implementation_status": implementation_status,
        "core_outputs": core_outputs,
        "metrics": metrics,
        "failure_modes": failure_modes,
        "chapter_notes": chapter_notes,
        "sources": sources,
    }
