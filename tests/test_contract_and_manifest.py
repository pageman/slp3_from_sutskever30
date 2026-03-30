from __future__ import annotations

import json
from pathlib import Path

from slp3_from_sutskever30.batch_a_artifacts import build_batch_a_payload
from slp3_from_sutskever30.batch_b_artifacts import build_batch_b_payload
from slp3_from_sutskever30.batch_c_artifacts import build_batch_c_payload
from slp3_from_sutskever30.batch_d_artifacts import build_batch_d_payload
from slp3_from_sutskever30.batch_e_artifacts import build_batch_e_payload
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
    assert payload["chapter_count"] == 29
    assert len(payload["chapters"]) == 29
    assert any(item["batch"] == "introductory_overview" for item in payload["chapters"])
    assert any(item["batch"] == "batch_a_classical_foundations" for item in payload["chapters"])
    rendered = render_deliverable_manifest(payload)
    assert json.loads(rendered)["chapter_count"] == 29


def test_local_observability_dir_is_separate_from_ci_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CIRCLECI", raising=False)
    assert get_observability_dir(tmp_path) == tmp_path / "observability" / "local"
    monkeypatch.setenv("CIRCLECI", "true")
    assert get_observability_dir(tmp_path) == tmp_path / "observability" / "ci_latest"


def test_batch_a_folder_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "research" / "batches" / "batch_a_classical_foundations" / "README.md").exists()
    assert (root / "research" / "batches" / "batch_b_lm_and_seq_models" / "README.md").exists()
    assert (root / "research" / "batches" / "batch_c_speech" / "README.md").exists()
    assert (root / "research" / "batches" / "batch_d_structure_and_ie" / "README.md").exists()
    assert (root / "research" / "batches" / "batch_e_discourse_and_dialogue" / "README.md").exists()


def test_batch_a_chapters_populate_rich_contract_fields() -> None:
    batch_a_keys = {"1", "2", "3", "4", "5", "6", "A", "B", "C", "D"}
    for spec in get_chapters():
        if spec.key not in batch_a_keys:
            continue
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert payload["lesson_objectives"]
        assert payload["core_algorithms"]
        assert payload["minimal_dataset"]
        assert payload["reference_experiments"]
        assert payload["book_vs_repo_gap"]


def test_batch_a_payload_contains_real_fixture_and_eval_pack_entries() -> None:
    payload = build_batch_a_payload()
    assert payload["chapter_count"] == 9
    assert len(payload["chapters"]) == 9
    assert sorted(payload["fixtures"]) == ["2", "3", "4", "5", "6", "A", "B", "C", "D"]
    assert sorted(payload["eval_packs"]) == ["2", "3", "4", "5", "6", "A", "B", "C", "D"]
    assert payload["eval_packs"]["2"]["lesson_objectives"]
    assert payload["eval_packs"]["A"]["reference_experiments"]


def test_batch_b_chapters_populate_rich_contract_fields() -> None:
    batch_b_keys = {"7", "8", "9", "10", "11", "12", "13"}
    for spec in get_chapters():
        if spec.key not in batch_b_keys:
            continue
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert payload["lesson_objectives"]
        assert payload["core_algorithms"]
        assert payload["minimal_dataset"]
        assert payload["reference_experiments"]
        assert payload["book_vs_repo_gap"]


def test_batch_b_payload_contains_real_fixture_and_eval_pack_entries() -> None:
    payload = build_batch_b_payload()
    assert payload["chapter_count"] == 7
    assert len(payload["chapters"]) == 7
    assert sorted(payload["fixtures"]) == ["10", "11", "12", "13", "7", "8", "9"]
    assert sorted(payload["eval_packs"]) == ["10", "11", "12", "13", "7", "8", "9"]
    assert payload["eval_packs"]["7"]["lesson_objectives"]
    assert payload["eval_packs"]["11"]["reference_experiments"]


def test_batch_c_chapters_populate_rich_contract_fields() -> None:
    batch_c_keys = {"14", "15", "16"}
    for spec in get_chapters():
        if spec.key not in batch_c_keys:
            continue
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert payload["lesson_objectives"]
        assert payload["core_algorithms"]
        assert payload["minimal_dataset"]
        assert payload["reference_experiments"]
        assert payload["book_vs_repo_gap"]


def test_batch_c_payload_contains_real_fixture_and_eval_pack_entries() -> None:
    payload = build_batch_c_payload()
    assert payload["chapter_count"] == 3
    assert len(payload["chapters"]) == 3
    assert sorted(payload["fixtures"]) == ["14", "15", "16"]
    assert sorted(payload["eval_packs"]) == ["14", "15", "16"]
    assert payload["eval_packs"]["14"]["lesson_objectives"]
    assert payload["eval_packs"]["16"]["reference_experiments"]


def test_batch_d_chapters_populate_rich_contract_fields() -> None:
    batch_d_keys = {"17", "18", "19", "20", "21"}
    for spec in get_chapters():
        if spec.key not in batch_d_keys:
            continue
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert payload["lesson_objectives"]
        assert payload["core_algorithms"]
        assert payload["minimal_dataset"]
        assert payload["reference_experiments"]
        assert payload["book_vs_repo_gap"]


def test_batch_d_payload_contains_real_fixture_and_eval_pack_entries() -> None:
    payload = build_batch_d_payload()
    assert payload["chapter_count"] == 5
    assert len(payload["chapters"]) == 5
    assert sorted(payload["fixtures"]) == ["17", "18", "19", "20", "21"]
    assert sorted(payload["eval_packs"]) == ["17", "18", "19", "20", "21"]
    assert payload["eval_packs"]["17"]["lesson_objectives"]
    assert payload["eval_packs"]["20"]["reference_experiments"]


def test_batch_e_chapters_populate_rich_contract_fields() -> None:
    batch_e_keys = {"22", "23", "24", "25"}
    for spec in get_chapters():
        if spec.key not in batch_e_keys:
            continue
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        assert payload["lesson_objectives"]
        assert payload["core_algorithms"]
        assert payload["minimal_dataset"]
        assert payload["reference_experiments"]
        assert payload["book_vs_repo_gap"]


def test_batch_e_payload_contains_real_fixture_and_eval_pack_entries() -> None:
    payload = build_batch_e_payload()
    assert payload["chapter_count"] == 4
    assert len(payload["chapters"]) == 4
    assert sorted(payload["fixtures"]) == ["22", "23", "24", "25"]
    assert sorted(payload["eval_packs"]) == ["22", "23", "24", "25"]
    assert payload["eval_packs"]["22"]["lesson_objectives"]
    assert payload["eval_packs"]["25"]["reference_experiments"]
