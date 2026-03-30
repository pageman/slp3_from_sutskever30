from __future__ import annotations

from typing import Any


REQUIRED_CHAPTER_FIELDS: tuple[str, ...] = (
    "chapter",
    "implementation_status",
    "core_outputs",
    "metrics",
    "failure_modes",
    "chapter_notes",
    "sources",
    "lesson_objectives",
    "core_algorithms",
    "minimal_dataset",
    "reference_experiments",
    "book_vs_repo_gap",
)


def _default_notes() -> dict[str, Any]:
    return {
        "batch": "",
        "counterintuitive_insight": "",
        "covered_claims": [],
        "omitted_claims": [],
    }


def _normalized_notes(notes: dict[str, Any] | None) -> dict[str, Any]:
    merged = _default_notes()
    if notes:
        merged.update(notes)
    return merged


def build_chapter_payload(
    *,
    chapter: str,
    implementation_status: str,
    core_outputs: dict[str, Any],
    metrics: dict[str, Any],
    failure_modes: list[dict[str, Any]],
    chapter_notes: dict[str, Any],
    sources: dict[str, Any],
    lesson_objectives: list[str] | None = None,
    core_algorithms: list[str] | None = None,
    minimal_dataset: dict[str, Any] | None = None,
    reference_experiments: list[dict[str, Any]] | None = None,
    book_vs_repo_gap: str = "",
) -> dict[str, Any]:
    payload = {
        "chapter": chapter,
        "implementation_status": implementation_status,
        "core_outputs": core_outputs,
        "metrics": metrics,
        "failure_modes": failure_modes,
        "chapter_notes": _normalized_notes(chapter_notes),
        "sources": sources,
        "lesson_objectives": lesson_objectives or [],
        "core_algorithms": core_algorithms or [],
        "minimal_dataset": minimal_dataset or {},
        "reference_experiments": reference_experiments or [],
        "book_vs_repo_gap": book_vs_repo_gap,
    }
    validate_chapter_payload(payload)
    return payload


def normalize_chapter_payload(
    *,
    chapter: str,
    implementation_status: str,
    title: str,
    source_papers: tuple[int, ...],
    payload: dict[str, Any],
) -> dict[str, Any]:
    if set(REQUIRED_CHAPTER_FIELDS) <= set(payload):
        normalized = dict(payload)
    else:
        core_outputs = {
            key: value
            for key, value in payload.items()
            if key not in {"chapter", "implementation_status"}
        }
        normalized = {
            "chapter": payload.get("chapter", chapter),
            "implementation_status": payload.get("implementation_status", implementation_status),
            "core_outputs": core_outputs,
            "metrics": payload.get("metrics", {}),
            "failure_modes": payload.get("failure_modes", []),
            "chapter_notes": payload.get(
                "chapter_notes",
                {
                    "batch": "legacy_or_adapted_runner",
                    "counterintuitive_insight": "",
                    "covered_claims": [f"Runnable chapter analog for {title}."],
                    "omitted_claims": ["Payload was normalized from a legacy or adapted runner."],
                },
            ),
            "sources": payload.get("sources", {"source_papers": list(source_papers)}),
            "lesson_objectives": payload.get("lesson_objectives", []),
            "core_algorithms": payload.get("core_algorithms", []),
            "minimal_dataset": payload.get("minimal_dataset", {}),
            "reference_experiments": payload.get("reference_experiments", []),
            "book_vs_repo_gap": payload.get(
                "book_vs_repo_gap",
                "Legacy/adapted runner normalized into the chapter contract; chapter-specific lesson metadata remains incomplete.",
            ),
        }
    normalized["chapter"] = chapter
    normalized["implementation_status"] = implementation_status
    normalized["chapter_notes"] = _normalized_notes(dict(normalized.get("chapter_notes", {})))
    sources = dict(normalized.get("sources", {}))
    sources.setdefault("source_papers", list(source_papers))
    normalized["sources"] = sources
    validate_chapter_payload(normalized)
    return normalized


def validate_chapter_payload(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_CHAPTER_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"chapter payload missing fields: {missing}")
    if not isinstance(payload["core_outputs"], dict):
        raise TypeError("core_outputs must be a dict")
    if not isinstance(payload["metrics"], dict):
        raise TypeError("metrics must be a dict")
    if not isinstance(payload["failure_modes"], list):
        raise TypeError("failure_modes must be a list")
    if not isinstance(payload["chapter_notes"], dict):
        raise TypeError("chapter_notes must be a dict")
    if not isinstance(payload["sources"], dict):
        raise TypeError("sources must be a dict")
    if not isinstance(payload["lesson_objectives"], list):
        raise TypeError("lesson_objectives must be a list")
    if not isinstance(payload["core_algorithms"], list):
        raise TypeError("core_algorithms must be a list")
    if not isinstance(payload["minimal_dataset"], dict):
        raise TypeError("minimal_dataset must be a dict")
    if not isinstance(payload["reference_experiments"], list):
        raise TypeError("reference_experiments must be a list")
    if not isinstance(payload["book_vs_repo_gap"], str):
        raise TypeError("book_vs_repo_gap must be a string")
