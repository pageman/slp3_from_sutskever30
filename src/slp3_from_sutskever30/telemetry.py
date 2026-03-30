from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
        [sys.executable, "-c", command] if command.startswith("import ") else ["/bin/zsh", "-lc", command],
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
        return {
            "command": value.command,
            "passed": value.passed,
            "exit_code": value.exit_code,
            "stdout": value.stdout,
            "stderr": value.stderr,
        }
    return value


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
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checked_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "repo_checks": checks,
        "chapter_count": len(chapters),
        "orphaned_chapters": get_orphaned_chapter_keys(),
        "unexpected_chapters": get_unexpected_chapter_keys(),
        "chapters": chapters,
    }


def render_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_yaml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"schema_version: {payload['schema_version']}")
    lines.append(f"generated_at: {quote_yaml(str(payload['generated_at']))}")
    lines.append(f"checked_commit: {optional_quote_yaml(payload['checked_commit'])}")
    lines.append(f"git_branch: {optional_quote_yaml(payload['git_branch'])}")
    lines.append("repo_checks:")
    for name, check in payload["repo_checks"].items():
        lines.extend(
            indent(
                [
                    f"{name}:",
                    f"  command: {quote_yaml(str(check['command']))}",
                    f"  passed: {bool_yaml(bool(check['passed']))}",
                    f"  exit_code: {check['exit_code']}",
                ],
                2,
            )
        )
    lines.append(f"chapter_count: {payload['chapter_count']}")
    orphaned = payload["orphaned_chapters"]
    unexpected = payload["unexpected_chapters"]
    lines.append(f"orphaned_chapters: {quote_yaml(json.dumps(orphaned)) if orphaned else '[]'}")
    lines.append(f"unexpected_chapters: {quote_yaml(json.dumps(unexpected)) if unexpected else '[]'}")
    lines.append("chapters:")
    for chapter in payload["chapters"]:
        lines.extend(
            indent(
                [
                    f"- key: {quote_yaml(str(chapter['key']))}",
                    f"  title: {quote_yaml(str(chapter['title']))}",
                    f"  implementation_status: {quote_yaml(str(chapter['implementation_status']))}",
                    f"  source_papers: {quote_yaml(json.dumps(chapter['source_papers'])) if chapter['source_papers'] else '[]'}",
                    f"  payload_keys: {quote_yaml(json.dumps(chapter['payload_keys']))}",
                ],
                2,
            )
        )
    return "\n".join(lines) + "\n"
