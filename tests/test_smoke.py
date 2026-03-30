from __future__ import annotations

from pathlib import Path

from slp3_from_sutskever30.chapters import ALL_CHAPTER_SPECS
from slp3_from_sutskever30.registry import (
    EXPECTED_CHAPTER_KEYS,
    get_chapters,
    get_orphaned_chapter_keys,
    get_unexpected_chapter_keys,
)


def test_all_chapters_run_and_return_chapter_key() -> None:
    chapters = get_chapters()
    assert len(chapters) == 36
    for spec in chapters:
        payload = spec.runner()
        assert payload["chapter"] == spec.key


def test_selected_shapes_exist_for_neural_chapters() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    assert chapter_map["8"].runner()["core_outputs"]["logits_shape"] == (4, 3)
    assert chapter_map["11"].runner()["core_outputs"]["rag_probs_shape"] == (4, 3)
    assert chapter_map["16"].runner()["core_outputs"]["mel_frames_shape"][0] == 2
    assert chapter_map["25"].runner()["core_outputs"]["act_logits_shape"] == (6, 5)


def test_no_orphaned_or_unexpected_slp3_chapters() -> None:
    chapters = get_chapters()
    assert tuple(spec.key for spec in chapters) == EXPECTED_CHAPTER_KEYS
    assert get_orphaned_chapter_keys() == []
    assert get_unexpected_chapter_keys() == []


def test_only_supported_status_labels_are_used() -> None:
    supported = {"FULL", "DIRECT", "ADAPTED", "SCAFFOLDED"}
    assert {spec.implementation_status for spec in get_chapters()} <= supported


def test_each_chapter_has_a_physical_module() -> None:
    chapters_dir = Path(__file__).resolve().parents[1] / "src" / "slp3_from_sutskever30" / "chapters"
    expected_files = [f"chapter_{spec['key'].zfill(2)}.py" if spec["key"].isdigit() else f"chapter_{spec['key']}.py" for spec in ALL_CHAPTER_SPECS]
    assert sorted(path.name for path in chapters_dir.glob("chapter_*.py")) == sorted(expected_files)


def test_batch_one_full_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("1", "2", "3", "4", "5", "6"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_two_full_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("9", "10"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_three_speech_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("14", "15", "16"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_four_structured_prediction_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("17", "21"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_four_parsing_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("18", "19"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_four_span_graph_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("20", "23"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_batch_five_discourse_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("22", "24", "25"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }


def test_appendix_chapters_expose_standard_contract_payload() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    for key in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"):
        payload = chapter_map[key].runner()
        assert chapter_map[key].implementation_status == "FULL"
        assert set(payload) >= {
            "chapter",
            "implementation_status",
            "core_outputs",
            "metrics",
            "failure_modes",
            "chapter_notes",
            "sources",
        }
