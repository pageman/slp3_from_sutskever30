from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import frame_state_tracker, slot_accuracy


def build_fixture() -> dict[str, object]:
    turns = [
        {"speaker": "user", "inform": {"food": "ramen"}},
        {"speaker": "system", "confirm": True},
        {"speaker": "user", "inform": {"area": "downtown"}},
        {"speaker": "user", "repair": ("food", "sushi")},
        {"speaker": "system", "confirm": True},
    ]
    gold_state = {"food": "sushi", "area": "downtown"}
    return {"turns": turns, "gold_state": gold_state}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    tracked = frame_state_tracker(fixture["turns"])
    return {"tracked": tracked}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "slot_accuracy": slot_accuracy(outputs["tracked"]["state"], fixture["gold_state"]),
        "repair_count": float(len(outputs["tracked"]["repairs"])),
        "confirmation_count": float(outputs["tracked"]["confirmations"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "frame_systems_depend_on_ontology", "note": "Anything outside the predefined slot set is invisible to a frame tracker."},
        {"case": "repairs_need_memory", "state": outputs["tracked"]["state"], "note": "A frame system fails if it cannot overwrite stale slot values after repair."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "Frame systems look simple until repair happens; then the core problem is memory overwriting, not slot extraction.",
        "covered_claims": ["This appendix complements chapter 25 with explicit frame-state tracking.", "Repairs and confirmations are handled as state transitions."],
        "omitted_claims": ["No learned NLU front-end.", "No policy optimization or database querying layer."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="K",
        implementation_status="FULL",
        core_outputs={"state": outputs["tracked"]["state"], "repairs": outputs["tracked"]["repairs"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["chapter 25 dialogue materials", "repo-native frame-dialogue appendix"]},
        lesson_objectives=["Track slot-value state across dialogue turns.", "Handle confirmation and repair explicitly.", "Show how frame-based systems differ from broader dialogue structure modeling."],
        core_algorithms=["slot-state tracking", "repair overwrite handling", "confirmation counting", "frame-state evaluation"],
        minimal_dataset={"turn_count": len(fixture["turns"]), "gold_slot_count": len(fixture["gold_state"])},
        reference_experiments=[
            {"name": "frame_state_accuracy", "metric": "slot_accuracy", "expected_signal": "repairs update the final slot state correctly"},
        ],
        book_vs_repo_gap="This appendix captures the core frame-state tracking story, but omits NLU parsing, action policy, and external knowledge-base interaction.",
    )


SPEC = {"key": "K", "title": "Frame-based Dialogue Systems", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
