from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backup_if_exists(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{path.name}.{timestamp}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def write_text_with_backup(path: Path, text: str) -> Path | None:
    backup = backup_if_exists(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return backup


def write_sqlite_table_with_backup(
    sqlite_path: Path,
    *,
    table_name: str,
    payload: dict[str, object],
    item_key: str,
    item_columns: tuple[str, ...],
) -> Path | None:
    backup = backup_if_exists(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    items = payload[item_key]
    with sqlite3.connect(sqlite_path) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS metadata")
        cur.execute("DROP TABLE IF EXISTS checks")
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
        cur.execute("CREATE TABLE checks (name TEXT PRIMARY KEY, command TEXT NOT NULL, passed INTEGER NOT NULL, exit_code INTEGER NOT NULL, stdout TEXT, stderr TEXT)")
        cur.execute(f"CREATE TABLE {table_name} ({', '.join(f'{column} TEXT NOT NULL' for column in item_columns)})")

        for key, value in payload.items():
            if key in {item_key, "repo_checks"}:
                continue
            cur.execute("INSERT INTO metadata (key, value_json) VALUES (?, ?)", (key, json.dumps(value, sort_keys=True)))

        for name, check in payload.get("repo_checks", {}).items():
            cur.execute(
                "INSERT INTO checks (name, command, passed, exit_code, stdout, stderr) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    name,
                    str(check.get("command", "")),
                    1 if bool(check.get("passed", False)) else 0,
                    int(check.get("exit_code", 0)),
                    str(check.get("stdout", "")),
                    str(check.get("stderr", "")),
                ),
            )

        insert_sql = f"INSERT INTO {table_name} ({', '.join(item_columns)}) VALUES ({', '.join('?' for _ in item_columns)})"
        for item in items:
            row = []
            for column in item_columns:
                value = item[column]
                row.append(json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else str(value))
            cur.execute(insert_sql, tuple(row))
        conn.commit()
    return backup
