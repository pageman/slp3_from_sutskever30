from __future__ import annotations

import datetime as dt
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
from typing import Any

from slp3_from_sutskever30.chapter_contract import normalize_chapter_payload
from slp3_from_sutskever30.registry import get_chapter_map


ROOT = Path(__file__).resolve().parents[2]
BATCH_F_DIR = ROOT / "research" / "batches" / "batch_f_web_appendices"
BATCH_F_KEYS: tuple[str, ...] = ("E", "F", "G", "H", "I", "J", "K")


def _module_name_for_key(key: str) -> str:
    suffix = key.zfill(2) if key.isdigit() else key
    return f"slp3_from_sutskever30.chapters.chapter_{suffix}"


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_batch_f_payload() -> dict[str, object]:
    chapter_map = get_chapter_map()
    chapters: list[dict[str, object]] = []
    for key in BATCH_F_KEYS:
        spec = chapter_map[key]
        module = importlib.import_module(_module_name_for_key(key))
        fixture = module.build_fixture()
        payload = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        suffix = key.zfill(2) if key.isdigit() else key
        chapters.append(
            {
                "key": key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "fixture_filename": f"chapter_{suffix}_fixture.json",
                "eval_pack_filename": f"chapter_{suffix}_eval_pack.json",
                "fixture": _to_jsonable(fixture),
                "eval_pack": {
                    "chapter": key,
                    "title": spec.title,
                    "implementation_status": spec.implementation_status,
                    "lesson_objectives": _to_jsonable(payload["lesson_objectives"]),
                    "core_algorithms": _to_jsonable(payload["core_algorithms"]),
                    "minimal_dataset": _to_jsonable(payload["minimal_dataset"]),
                    "reference_experiments": _to_jsonable(payload["reference_experiments"]),
                    "metrics": _to_jsonable(payload["metrics"]),
                    "failure_modes": _to_jsonable(payload["failure_modes"]),
                    "book_vs_repo_gap": _to_jsonable(payload["book_vs_repo_gap"]),
                    "chapter_notes": _to_jsonable(payload["chapter_notes"]),
                },
            }
        )
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch": "batch_f_web_appendices",
        "chapter_count": len(chapters),
        "chapters": [
            {
                "key": chapter["key"],
                "title": chapter["title"],
                "implementation_status": chapter["implementation_status"],
                "fixture_path": f"research/batches/batch_f_web_appendices/fixtures/{chapter['fixture_filename']}",
                "eval_pack_path": f"research/batches/batch_f_web_appendices/eval_packs/{chapter['eval_pack_filename']}",
            }
            for chapter in chapters
        ],
        "fixtures": {chapter["key"]: chapter["fixture"] for chapter in chapters},
        "eval_packs": {chapter["key"]: chapter["eval_pack"] for chapter in chapters},
    }


def render_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_batch_f_artifacts(base_dir: Path = BATCH_F_DIR) -> Path:
    payload = build_batch_f_payload()
    fixtures_dir = base_dir / "fixtures"
    eval_packs_dir = base_dir / "eval_packs"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    eval_packs_dir.mkdir(parents=True, exist_ok=True)
    for key, fixture in payload["fixtures"].items():
        suffix = key.zfill(2) if str(key).isdigit() else str(key)
        (fixtures_dir / f"chapter_{suffix}_fixture.json").write_text(render_json(fixture))
    for key, eval_pack in payload["eval_packs"].items():
        suffix = key.zfill(2) if str(key).isdigit() else str(key)
        (eval_packs_dir / f"chapter_{suffix}_eval_pack.json").write_text(render_json(eval_pack))
    manifest_path = base_dir / "BATCH_F_MANIFEST.json"
    manifest_path.write_text(
        render_json(
            {
                "schema_version": payload["schema_version"],
                "generated_at": payload["generated_at"],
                "batch": payload["batch"],
                "chapter_count": payload["chapter_count"],
                "chapters": payload["chapters"],
            }
        )
    )
    return manifest_path
