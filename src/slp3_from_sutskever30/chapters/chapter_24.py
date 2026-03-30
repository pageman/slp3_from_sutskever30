from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.discourse import coherence_score, entity_grid, perturb_order


def build_fixture() -> dict[str, object]:
    document = [
        ["The", "agent", "opened", "the", "case"],
        ["The", "customer", "described", "the", "delay"],
        ["The", "agent", "offered", "a", "refund"],
        ["The", "customer", "accepted", "the", "refund"],
    ]
    tracked_entities = ["agent", "customer", "refund", "delay"]
    return {"document": document, "tracked_entities": tracked_entities}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(24)
    sentence_repr = rng.normal(scale=0.2, size=(len(fixture["document"]), 12))
    perturbed_doc = perturb_order(fixture["document"])
    perturbed_repr = sentence_repr[[0, 2, 1, 3]]
    grid = entity_grid(fixture["document"], fixture["tracked_entities"])
    perturbed_grid = entity_grid(perturbed_doc, fixture["tracked_entities"])
    ordering_logits = np.asarray(
        [
            [coherence_score(sentence_repr, grid), coherence_score(perturbed_repr, perturbed_grid)],
            [coherence_score(perturbed_repr, perturbed_grid), coherence_score(sentence_repr, grid)],
        ],
        dtype=np.float64,
    )
    return {"grid": grid, "perturbed_grid": perturbed_grid, "ordering_logits": ordering_logits, "coherent_score": coherence_score(sentence_repr, grid), "perturbed_score": coherence_score(perturbed_repr, perturbed_grid)}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "coherent_score": float(outputs["coherent_score"]),
        "perturbed_score": float(outputs["perturbed_score"]),
        "margin": float(outputs["coherent_score"] - outputs["perturbed_score"]),
        "entity_grid_shape": tuple(outputs["grid"].shape),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "near_correct_documents_are_the_real_test",
            "coherent_score": float(outputs["coherent_score"]),
            "perturbed_score": float(outputs["perturbed_score"]),
        },
        {
            "case": "binary_wellformedness_is_too_easy",
            "note": "The main challenge is ranking minimally corrupted discourse variants, not separating perfect text from nonsense.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_e_discourse_and_dialogue",
        "counterintuitive_insight": "Discourse models should be judged on near-correct documents, not obvious incoherence. Robustness to tiny perturbations matters more than easy binary wins.",
        "covered_claims": [
            "Chapter 24 now includes entity-grid and sentence-ordering style coherence evaluation.",
            "A minimally perturbed document is compared directly against a coherent baseline.",
        ],
        "omitted_claims": ["No discourse relation classifier yet.", "No hierarchical transformer encoder yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="24",
        implementation_status="FULL",
        core_outputs={
            "entity_grid": outputs["grid"].tolist(),
            "ordering_logits": outputs["ordering_logits"].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Model coherence with both entity-grid structure and sentence-order signals.",
            "Compare coherent and minimally perturbed documents directly.",
            "Treat small coherence margins as the central difficulty, not easy nonsense detection.",
        ],
        core_algorithms=["entity-grid construction", "sentence-order perturbation", "coherence scoring", "margin-based discourse comparison"],
        minimal_dataset={"sentence_count": len(fixture["document"]), "tracked_entity_count": len(fixture["tracked_entities"])},
        reference_experiments=[
            {"name": "coherence_margin", "metric": "margin", "expected_signal": "coherent document should outscore perturbed ordering by a positive but not trivial margin"},
            {"name": "entity_grid_shape", "metric": "entity_grid_shape", "expected_signal": "tracked-entity structure stays explicit in the evaluation object"},
        ],
        book_vs_repo_gap="This chapter is faithful in entity-grid and perturbation-style coherence testing, but still omits discourse relation parsing and larger document encoders.",
    )


SPEC = {
    "key": "24",
    "title": "Discourse Coherence",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
