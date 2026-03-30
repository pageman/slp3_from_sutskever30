from __future__ import annotations

import sqlite3
from pathlib import Path

from slp3_from_sutskever30.telemetry import build_telemetry_payload, render_json, render_yaml, write_telemetry_artifacts


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
    assert '"ci": {' in json_text
    assert "stdout_preview" in json_text
    assert '"stdout":' not in json_text
    assert "repo_checks:" in yaml_text
    assert "ci:" in yaml_text
    assert "stdout_truncated:" in yaml_text
    assert "chapters:" in yaml_text
    assert "source_papers:" in yaml_text
    assert "payload_keys:" in yaml_text


def test_telemetry_writes_sqlite_mirror(tmp_path: Path) -> None:
    payload = build_telemetry_payload(run_live_checks=False)
    json_path = tmp_path / "verification.json"
    yaml_path = tmp_path / "verification.yaml"
    sqlite_path = tmp_path / "verification.sqlite"
    write_telemetry_artifacts(json_path, yaml_path, sqlite_path, payload)
    assert json_path.exists()
    assert yaml_path.exists()
    assert sqlite_path.exists()
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chapters")
        assert cur.fetchone()[0] == 28
