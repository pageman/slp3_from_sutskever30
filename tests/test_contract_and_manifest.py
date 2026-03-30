from __future__ import annotations

import json
from pathlib import Path

from slp3_from_sutskever30.chapter_contract import REQUIRED_CHAPTER_FIELDS, normalize_chapter_payload
from slp3_from_sutskever30.deliverable_manifest import build_deliverable_manifest, render_deliverable_manifest
from slp3_from_sutskever30.observability_paths import get_observability_dir
from slp3_from_sutskever30.registry import get_chapters


def test_all_chapters_normalize_to_contract() -> None:
    for spec in get_chapters():
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert set(REQUIRED_CHAPTER_FIELDS) <= set(payload)


def test_deliverable_manifest_covers_all_registered_chapters() -> None:
    payload = build_deliverable_manifest()
    assert payload["chapter_count"] == 28
    assert len(payload["chapters"]) == 28
    assert any(item["batch"] == "batch_a_classical_foundations" for item in payload["chapters"])
    rendered = render_deliverable_manifest(payload)
    assert json.loads(rendered)["chapter_count"] == 28


def test_local_observability_dir_is_separate_from_ci_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CIRCLECI", raising=False)
    assert get_observability_dir(tmp_path) == tmp_path / "observability" / "local"
    monkeypatch.setenv("CIRCLECI", "true")
    assert get_observability_dir(tmp_path) == tmp_path / "observability"


def test_batch_a_folder_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "research" / "batches" / "batch_a_classical_foundations" / "README.md").exists()
