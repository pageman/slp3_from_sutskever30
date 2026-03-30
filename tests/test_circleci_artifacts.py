from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from slp3_from_sutskever30.circleci_artifacts import build_circleci_payload, write_circleci_artifacts


def test_circleci_payload_has_expected_shape(monkeypatch) -> None:
    monkeypatch.setenv("CIRCLE_BRANCH", "main")
    monkeypatch.setenv("CIRCLE_SHA1", "abc123")
    monkeypatch.setenv("CIRCLE_BUILD_NUM", "7")
    monkeypatch.setenv("CIRCLE_WORKFLOW_ID", "workflow-123")
    payload = build_circleci_payload()
    assert payload["run_count"] == 1
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["branch"] == "main"
    assert payload["runs"][0]["sha1"] == "abc123"
    assert payload["runs"][0]["pipeline_number"] == "7"
    assert payload["runs"][0]["workflow_url"] == "https://app.circleci.com/pipelines/workflows/workflow-123"


def test_circleci_artifacts_write_json_and_sqlite(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CIRCLE_BRANCH", "main")
    monkeypatch.setenv("CIRCLE_SHA1", "abc123")
    monkeypatch.setenv("CIRCLE_BUILD_NUM", "7")
    monkeypatch.setenv("CIRCLE_WORKFLOW_ID", "workflow-123")
    json_path = tmp_path / "circleci_run.json"
    sqlite_path = tmp_path / "circleci_run.sqlite"
    payload = build_circleci_payload()
    write_circleci_artifacts(json_path, sqlite_path, payload)
    assert json.loads(json_path.read_text())["run_count"] == 1
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM runs")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT pipeline_number, workflow_url FROM runs")
        assert cur.fetchone() == ("7", "https://app.circleci.com/pipelines/workflows/workflow-123")
