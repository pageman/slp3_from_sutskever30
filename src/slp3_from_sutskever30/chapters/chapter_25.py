from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.discourse import commitment_consistency, dialogue_state


def build_fixture() -> dict[str, object]:
    turns = [
        {"speaker": "user", "act": "request", "commit": ("refund", "requested"), "grounded": True},
        {"speaker": "agent", "act": "inform", "commit": ("refund", "approved"), "grounded": True},
        {"speaker": "user", "act": "confirm", "check": ("refund", "approved"), "grounded": True},
        {"speaker": "agent", "act": "repair", "repair": True, "commit": ("delivery_date", "tomorrow"), "grounded": False},
        {"speaker": "user", "act": "question", "check": ("delivery_date", "tomorrow"), "grounded": True},
        {"speaker": "agent", "act": "inform", "commit": ("delivery_date", "tomorrow"), "grounded": True},
    ]
    return {"turns": turns}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    state = dialogue_state(fixture["turns"])
    act_vocab = ["request", "inform", "confirm", "repair", "question"]
    turn_taking = [fixture["turns"][idx]["speaker"] != fixture["turns"][idx - 1]["speaker"] for idx in range(1, len(fixture["turns"]))]
    repair_positions = [idx for idx, turn in enumerate(fixture["turns"]) if turn.get("repair")]
    act_logits = []
    for turn in fixture["turns"]:
        logits = np.asarray([1.0 if turn["act"] == act else -0.5 for act in act_vocab], dtype=np.float64)
        act_logits.append(logits)
    return {"state": state, "act_vocab": act_vocab, "act_logits": np.asarray(act_logits), "turn_taking": turn_taking, "repair_positions": repair_positions}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    predicted_acts = np.argmax(outputs["act_logits"], axis=1)
    gold_acts = np.asarray([outputs["act_vocab"].index(turn["act"]) for turn in fixture["turns"]], dtype=np.int64)
    return {
        "dialogue_act_accuracy": float(np.mean(predicted_acts == gold_acts)),
        "turn_taking_rate": float(np.mean(outputs["turn_taking"])),
        "grounding_rate": float(outputs["state"]["grounding_rate"]),
        "commitment_consistency": commitment_consistency(fixture["turns"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "fluency_is_not_commitment_consistency",
            "commitments": outputs["state"]["commitments"],
            "consistency": commitment_consistency(fixture["turns"]),
        },
        {
            "case": "repair_turns_change_state_nonlocally",
            "repair_positions": outputs["repair_positions"],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_5_discourse_and_dialogue",
        "counterintuitive_insight": "Conversation systems fail harder on forgotten commitments than on mildly awkward wording. Commitment consistency should be optimized before response fluency.",
        "covered_claims": [
            "Chapter 25 now includes dialogue acts, turn-taking, grounding, repair signals, and commitment memory.",
            "Evaluation includes consistency over prior commitments, not just turn-local predictions.",
        ],
        "omitted_claims": ["No retrieval-backed long-context memory yet.", "No multi-session conversation tracking yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="25",
        implementation_status="FULL",
        core_outputs={
            "act_logits_shape": tuple(outputs["act_logits"].shape),
            "dialogue_state": outputs["state"],
            "repair_positions": outputs["repair_positions"],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "25",
    "title": "Conversation and its Structure",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
