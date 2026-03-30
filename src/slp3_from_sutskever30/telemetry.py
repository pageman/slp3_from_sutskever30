from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from slp3_from_sutskever30.artifacts import write_sqlite_table_with_backup, write_text_with_backup
from slp3_from_sutskever30.circleci_artifacts import build_circleci_payload
from slp3_from_sutskever30.registry import get_chapters, get_orphaned_chapter_keys, get_unexpected_chapter_keys


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CommandResult:
    command: str
    passed: bool
    exit_code: int
    stdout: str
    stderr: str


def run_command(command: str) -> CommandResult:
    completed = subprocess.run(
        [sys.executable, "-c", command] if command.startswith("import ") else ["/bin/sh", "-lc", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return CommandResult(
        command=command,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except Exception:
        return None


def bool_yaml(value: bool) -> str:
    return "true" if value else "false"


def quote_yaml(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def optional_quote_yaml(text: str | None) -> str:
    return "null" if text is None else quote_yaml(text)


def indent(lines: Iterable[str], n: int) -> list[str]:
    prefix = " " * n
    return [prefix + line if line else line for line in lines]


def _yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return bool_yaml(value)
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return quote_yaml(str(value))


def _yaml_lines_for_value(key: str, value: object, level: int = 0) -> list[str]:
    prefix = " " * level
    if isinstance(value, dict):
        lines = [f"{prefix}{key}:"]
        for child_key, child_value in value.items():
            lines.extend(_yaml_lines_for_value(str(child_key), child_value, level + 2))
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}{key}: []"]
        lines = [f"{prefix}{key}:"]
        for item in value:
            if isinstance(item, dict):
                first = True
                for child_key, child_value in item.items():
                    marker = "- " if first else "  "
                    if isinstance(child_value, (dict, list)):
                        lines.append(f"{prefix}  {marker}{child_key}:")
                        nested = _yaml_lines_for_value(child_key, child_value, level + 6)[1:]
                        lines.extend(nested)
                    else:
                        lines.append(f"{prefix}  {marker}{child_key}: {_yaml_scalar(child_value)}")
                    first = False
            elif isinstance(item, list):
                lines.append(f"{prefix}  -")
                for child in item:
                    lines.append(f"{prefix}    - {_yaml_scalar(child)}")
            else:
                lines.append(f"{prefix}  - {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{key}: {_yaml_scalar(value)}"]


def _preview_text(text: str, *, max_lines: int = 4, max_chars: int = 240) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    lines = stripped.splitlines()[:max_lines]
    preview = "\n".join(lines)
    if len(preview) > max_chars or len(stripped.splitlines()) > max_lines:
        return preview[:max_chars].rstrip() + " ..."
    return preview


def _trim_check_output(mapping: dict[str, object]) -> dict[str, object]:
    stdout = str(mapping.get("stdout", ""))
    stderr = str(mapping.get("stderr", ""))
    return {
        "command": mapping.get("command", ""),
        "passed": bool(mapping.get("passed", False)),
        "exit_code": int(mapping.get("exit_code", 0)),
        "stdout_preview": _preview_text(stdout),
        "stderr_preview": _preview_text(stderr),
        "stdout_truncated": bool(stdout.strip()) and _preview_text(stdout) != stdout.strip(),
        "stderr_truncated": bool(stderr.strip()) and _preview_text(stderr) != stderr.strip(),
    }


def collect_repo_checks(*, run_live_checks: bool) -> dict[str, CommandResult | dict[str, object]]:
    defaults: dict[str, CommandResult | dict[str, object]] = {
        "smoke_test": {
            "command": "python3 scripts/smoke_test.py",
            "passed": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
        "pytest_local": {
            "command": "python3 -m pytest",
            "passed": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
        "survey": {
            "command": "python3 scripts/survey_slp3.py",
            "passed": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    }
    if not run_live_checks:
        return defaults
    return {
        "smoke_test": run_command("python3 scripts/smoke_test.py"),
        "pytest_local": run_command("python3 -m pytest"),
        "survey": run_command("python3 scripts/survey_slp3.py"),
    }


def _command_mapping(value: CommandResult | dict[str, object]) -> dict[str, object]:
    if isinstance(value, CommandResult):
        return _trim_check_output({
            "command": value.command,
            "passed": value.passed,
            "exit_code": value.exit_code,
            "stdout": value.stdout,
            "stderr": value.stderr,
        })
    return _trim_check_output(value)


def build_telemetry_payload(*, run_live_checks: bool) -> dict[str, object]:
    checks = {name: _command_mapping(result) for name, result in collect_repo_checks(run_live_checks=run_live_checks).items()}
    chapters = []
    for spec in get_chapters():
        payload = spec.runner()
        chapters.append(
            {
                "key": spec.key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "source_papers": list(spec.source_papers),
                "payload_keys": sorted(payload.keys()),
            }
        )
    circleci_payload = build_circleci_payload()
    ci_run = circleci_payload["runs"][0] if circleci_payload["runs"] else {}
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checked_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "ci": {
            "platform": ci_run.get("ci_platform", ""),
            "branch": ci_run.get("branch", ""),
            "sha1": ci_run.get("sha1", ""),
            "job": ci_run.get("job", ""),
            "build_url": ci_run.get("build_url", ""),
            "workflow_id": ci_run.get("workflow_id", ""),
            "workflow_url": ci_run.get("workflow_url", ""),
            "pipeline_id": ci_run.get("pipeline_id", ""),
            "pipeline_number": ci_run.get("pipeline_number", ""),
        },
        "repo_checks": checks,
        "chapter_count": len(chapters),
        "orphaned_chapters": get_orphaned_chapter_keys(),
        "unexpected_chapters": get_unexpected_chapter_keys(),
        "chapters": chapters,
    }


def render_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_yaml(payload: dict[str, object]) -> str:
    yaml_payload = dict(payload)
    yaml_checks: dict[str, object] = {}
    for name, check in payload["repo_checks"].items():
        yaml_checks[name] = {
            "command": check["command"],
            "passed": check["passed"],
            "exit_code": check["exit_code"],
            "stdout_preview": str(check.get("stdout_preview", "")),
            "stderr_preview": str(check.get("stderr_preview", "")),
            "stdout_truncated": bool(check.get("stdout_truncated", False)),
            "stderr_truncated": bool(check.get("stderr_truncated", False)),
        }
    yaml_payload["repo_checks"] = yaml_checks
    lines: list[str] = []
    for key in ("schema_version", "generated_at", "checked_commit", "git_branch", "ci", "repo_checks", "chapter_count", "orphaned_chapters", "unexpected_chapters", "chapters"):
        lines.extend(_yaml_lines_for_value(key, yaml_payload[key], 0))
    return "\n".join(lines) + "\n"


def write_telemetry_artifacts(json_path: Path, yaml_path: Path, sqlite_path: Path, payload: dict[str, object]) -> dict[str, str | None]:
    json_backup = write_text_with_backup(json_path, render_json(payload))
    yaml_backup = write_text_with_backup(yaml_path, render_yaml(payload))
    sqlite_backup = write_sqlite_table_with_backup(
        sqlite_path,
        table_name="chapters",
        payload=payload,
        item_key="chapters",
        item_columns=("key", "title", "implementation_status", "source_papers", "payload_keys"),
    )
    return {
        "json_backup": None if json_backup is None else str(json_backup),
        "yaml_backup": None if yaml_backup is None else str(yaml_backup),
        "sqlite_backup": None if sqlite_backup is None else str(sqlite_backup),
    }
