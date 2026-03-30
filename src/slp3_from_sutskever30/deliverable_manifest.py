from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from slp3_from_sutskever30.chapter_contract import normalize_chapter_payload
from slp3_from_sutskever30.registry import get_chapters


ROOT = Path(__file__).resolve().parents[2]

AUDIT_CLASS_BY_KEY = {
    "1": "not yet aligned",
    "2": "method match",
    "3": "method match",
    "4": "method match",
    "5": "method match",
    "6": "method match",
    "7": "topic match",
    "8": "method match",
    "9": "method match",
    "10": "method match",
    "11": "method match",
    "12": "topic match",
    "13": "method match",
    "14": "method match",
    "15": "method match",
    "16": "method match",
    "17": "method match",
    "18": "method match",
    "19": "method match",
    "20": "topic match",
    "21": "method match",
    "22": "topic match",
    "23": "topic match",
    "24": "method match",
    "25": "topic match",
    "A": "method match",
    "B": "method match",
    "C": "method match",
    "D": "method match",
}

FEASIBILITY_BY_KEY = {
    "1": "fully feasible in NumPy",
    "2": "fully feasible in NumPy",
    "3": "fully feasible in NumPy",
    "4": "fully feasible in NumPy",
    "5": "fully feasible in NumPy",
    "6": "fully feasible in NumPy",
    "7": "not realistically faithful without non-NumPy infrastructure",
    "8": "faithful only in miniature",
    "9": "faithful only in miniature",
    "10": "faithful only in miniature",
    "11": "faithful only in miniature",
    "12": "faithful only in miniature",
    "13": "fully feasible in NumPy",
    "14": "fully feasible in NumPy",
    "15": "faithful only in miniature",
    "16": "faithful only in miniature",
    "17": "fully feasible in NumPy",
    "18": "fully feasible in NumPy",
    "19": "fully feasible in NumPy",
    "20": "faithful only in miniature",
    "21": "fully feasible in NumPy",
    "22": "fully feasible in NumPy",
    "23": "faithful only in miniature",
    "24": "fully feasible in NumPy",
    "25": "faithful only in miniature",
    "A": "fully feasible in NumPy",
    "B": "fully feasible in NumPy",
    "C": "fully feasible in NumPy",
    "D": "fully feasible in NumPy",
}

BATCH_BY_KEY = {
    "2": "batch_a_classical_foundations",
    "3": "batch_a_classical_foundations",
    "4": "batch_a_classical_foundations",
    "5": "batch_a_classical_foundations",
    "6": "batch_a_classical_foundations",
    "A": "batch_a_classical_foundations",
    "B": "batch_a_classical_foundations",
    "C": "batch_a_classical_foundations",
    "D": "batch_a_classical_foundations",
    "7": "batch_b_lm_and_seq_models",
    "8": "batch_b_lm_and_seq_models",
    "9": "batch_b_lm_and_seq_models",
    "10": "batch_b_lm_and_seq_models",
    "11": "batch_b_lm_and_seq_models",
    "12": "batch_b_lm_and_seq_models",
    "13": "batch_b_lm_and_seq_models",
    "14": "batch_c_speech",
    "15": "batch_c_speech",
    "16": "batch_c_speech",
    "17": "batch_d_structure_and_ie",
    "18": "batch_d_structure_and_ie",
    "19": "batch_d_structure_and_ie",
    "20": "batch_d_structure_and_ie",
    "21": "batch_d_structure_and_ie",
    "22": "batch_e_discourse_and_dialogue",
    "23": "batch_e_discourse_and_dialogue",
    "24": "batch_e_discourse_and_dialogue",
    "25": "batch_e_discourse_and_dialogue",
}


def _chapter_module_path(key: str) -> str:
    suffix = key.zfill(2) if key.isdigit() else key
    return f"src/slp3_from_sutskever30/chapters/chapter_{suffix}.py"


def build_deliverable_manifest() -> dict[str, object]:
    chapters = []
    for spec in get_chapters():
        normalized = normalize_chapter_payload(
            chapter=spec.key,
            implementation_status=spec.implementation_status,
            title=spec.title,
            source_papers=spec.source_papers,
            payload=spec.runner(),
        )
        chapters.append(
            {
                "key": spec.key,
                "title": spec.title,
                "implementation_status": spec.implementation_status,
                "audit_class": AUDIT_CLASS_BY_KEY.get(spec.key, "not yet aligned"),
                "feasibility": FEASIBILITY_BY_KEY.get(spec.key, "not yet aligned"),
                "batch": BATCH_BY_KEY.get(spec.key, ""),
                "source_papers": list(spec.source_papers),
                "module_path": _chapter_module_path(spec.key),
                "contract_fields": sorted(normalized.keys()),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "chapter_count": len(chapters),
        "batches": {
            "batch_a_classical_foundations": {
                "chapters": ["2", "3", "4", "5", "6", "A", "B", "C", "D"],
                "folder": "research/batches/batch_a_classical_foundations",
            },
            "batch_b_lm_and_seq_models": {
                "chapters": ["7", "8", "9", "10", "11", "12", "13"],
                "folder": "research/batches/batch_b_lm_and_seq_models",
            },
            "batch_c_speech": {
                "chapters": ["14", "15", "16"],
                "folder": "research/batches/batch_c_speech",
            },
            "batch_d_structure_and_ie": {"chapters": ["17", "18", "19", "20", "21"]},
            "batch_e_discourse_and_dialogue": {"chapters": ["22", "23", "24", "25"]},
        },
        "chapters": chapters,
    }


def render_deliverable_manifest(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
