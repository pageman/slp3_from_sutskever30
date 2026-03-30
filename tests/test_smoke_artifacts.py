from __future__ import annotations

import sqlite3
from pathlib import Path

from slp3_from_sutskever30.artifacts import write_sqlite_table_with_backup
from slp3_from_sutskever30.smoke_support import build_smoke_payload


def test_smoke_payload_writes_sqlite_mirror(tmp_path: Path) -> None:
    payload = build_smoke_payload()
    sqlite_path = tmp_path / "smoke_test.sqlite"
    write_sqlite_table_with_backup(
        sqlite_path,
        table_name="results",
        payload=payload,
        item_key="results",
        item_columns=("key", "title", "implementation_status", "source_papers", "payload_keys"),
    )
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM results")
        assert cur.fetchone()[0] == 29
