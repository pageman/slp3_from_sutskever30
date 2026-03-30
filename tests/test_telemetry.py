from __future__ import annotations

from slp3_from_sutskever30.telemetry import build_telemetry_payload, render_json, render_yaml


def test_telemetry_payload_has_expected_shape() -> None:
    payload = build_telemetry_payload(run_live_checks=False)
    assert payload["chapter_count"] == 28
    assert payload["orphaned_chapters"] == []
    assert payload["unexpected_chapters"] == []
    assert len(payload["chapters"]) == 28


def test_telemetry_renderers_emit_core_fields() -> None:
    payload = build_telemetry_payload(run_live_checks=False)
    json_text = render_json(payload)
    yaml_text = render_yaml(payload)
    assert '"chapter_count": 28' in json_text
    assert "repo_checks:" in yaml_text
    assert "chapters:" in yaml_text
