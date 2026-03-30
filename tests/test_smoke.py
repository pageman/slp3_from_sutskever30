from __future__ import annotations

from slp3_from_sutskever30.registry import get_chapters


def test_all_chapters_run_and_return_chapter_key() -> None:
    chapters = get_chapters()
    assert len(chapters) == 28
    for spec in chapters:
        payload = spec.runner()
        assert payload["chapter"] == spec.key


def test_selected_shapes_exist_for_neural_chapters() -> None:
    chapter_map = {spec.key: spec for spec in get_chapters()}
    assert chapter_map["8"].runner()["logits_shape"] == (4, 3)
    assert chapter_map["11"].runner()["rag_probs_shape"] == (4, 3)
    assert chapter_map["16"].runner()["mel_frames_shape"][0] == 4
    assert chapter_map["25"].runner()["dialogue_act_logits_shape"] == (4, 5, 8)
