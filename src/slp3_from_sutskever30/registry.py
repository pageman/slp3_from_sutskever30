from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from slp3_from_sutskever30.chapters import ALL_CHAPTER_SPECS


@dataclass(frozen=True)
class ChapterSpec:
    key: str
    title: str
    implementation_status: str
    source_papers: tuple[int, ...]
    runner: Callable[[], dict[str, object]]


EXPECTED_CHAPTER_KEYS: tuple[str, ...] = tuple([str(index) for index in range(2, 26)] + ["A", "B", "C", "D"])


def get_chapters() -> list[ChapterSpec]:
    return [ChapterSpec(**spec) for spec in ALL_CHAPTER_SPECS]


def get_chapter_map() -> dict[str, ChapterSpec]:
    return {spec.key: spec for spec in get_chapters()}


def get_orphaned_chapter_keys() -> list[str]:
    present = {spec.key for spec in get_chapters()}
    return [key for key in EXPECTED_CHAPTER_KEYS if key not in present]


def get_unexpected_chapter_keys() -> list[str]:
    expected = set(EXPECTED_CHAPTER_KEYS)
    return [spec.key for spec in get_chapters() if spec.key not in expected]
