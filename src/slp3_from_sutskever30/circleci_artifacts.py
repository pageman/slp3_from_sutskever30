from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from slp3_from_sutskever30.artifacts import write_sqlite_table_with_backup, write_text_with_backup


def build_circleci_payload() -> dict[str, object]:
    env = os.environ
    workflow_id = env.get("CIRCLE_WORKFLOW_ID", "")
    workflow_url = f"https://app.circleci.com/pipelines/workflows/{workflow_id}" if workflow_id else ""
    build_num = env.get("CIRCLE_BUILD_NUM", "")
    pipeline_number = env.get("CIRCLE_PIPELINE_NUMBER", "") or build_num
    run = {
        "vcs_type": env.get("CIRCLE_VCS_TYPE", ""),
        "project_username": env.get("CIRCLE_PROJECT_USERNAME", ""),
        "project_reponame": env.get("CIRCLE_PROJECT_REPONAME", ""),
        "branch": env.get("CIRCLE_BRANCH", ""),
        "sha1": env.get("CIRCLE_SHA1", ""),
        "build_num": build_num,
        "build_url": env.get("CIRCLE_BUILD_URL", ""),
        "job": env.get("CIRCLE_JOB", ""),
        "workflow_id": workflow_id,
        "workflow_url": workflow_url,
        "workflow_job_id": env.get("CIRCLE_WORKFLOW_JOB_ID", ""),
        "pipeline_id": env.get("CIRCLE_PIPELINE_ID", ""),
        "pipeline_number": pipeline_number,
        "pull_request": env.get("CIRCLE_PULL_REQUEST", ""),
        "pull_requests": [value for value in env.get("CIRCLE_PULL_REQUESTS", "").split(",") if value],
        "runner_os": env.get("RUNNER_OS", ""),
        "ci_platform": "circleci",
    }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_count": 1,
        "runs": [run],
    }


def write_circleci_artifacts(json_path: Path, sqlite_path: Path, payload: dict[str, object]) -> dict[str, str | None]:
    json_backup = write_text_with_backup(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sqlite_backup = write_sqlite_table_with_backup(
        sqlite_path,
        table_name="runs",
        payload=payload,
        item_key="runs",
        item_columns=(
            "ci_platform",
            "vcs_type",
            "project_username",
            "project_reponame",
            "branch",
            "sha1",
            "build_num",
            "build_url",
            "job",
            "workflow_id",
            "workflow_url",
            "workflow_job_id",
            "pipeline_id",
            "pipeline_number",
            "pull_request",
            "pull_requests",
            "runner_os",
        ),
    )
    return {
        "json_backup": None if json_backup is None else str(json_backup),
        "sqlite_backup": None if sqlite_backup is None else str(sqlite_backup),
    }
