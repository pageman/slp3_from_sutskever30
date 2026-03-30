from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.span_graph import pair_features, propose_spans, schema_constrained_relations


ENTITY_TYPES = ["O", "PER", "ORG", "LOC", "EVENT"]
RELATION_TYPES = ["none", "works_for", "located_in", "before"]


def build_fixture() -> dict[str, object]:
    token_ids = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
        ],
        dtype=np.int64,
    )
    gold_entity_types = np.asarray(
        [
            [1, 2, 0, 3, 4, 0],
            [1, 2, 0, 3, 0, 4],
            [2, 1, 0, 4, 3, 0],
            [1, 0, 2, 3, 4, 0],
        ],
        dtype=np.int64,
    )
    return {"token_ids": token_ids, "gold_entity_types": gold_entity_types}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(20)
    token_emb = rng.normal(scale=0.2, size=(25, 10))
    entity_head = rng.normal(scale=0.2, size=(len(ENTITY_TYPES), 10))
    relation_head = rng.normal(scale=0.2, size=(len(RELATION_TYPES), 20))
    event_head = rng.normal(scale=0.2, size=(2, 10))
    time_head = rng.normal(scale=0.2, size=(2, 20))
    token_repr = token_emb[fixture["token_ids"]]
    span_repr, span_bounds = propose_spans(token_repr, max_width=1)
    entity_logits = np.einsum("cf,bsf->bsc", entity_head, span_repr)
    entity_types = np.argmax(entity_logits, axis=2)
    relation_repr = pair_features(span_repr)
    relation_logits = np.einsum("cf,bsrf->bsrc", relation_head, relation_repr)
    constrained_relations = schema_constrained_relations(entity_types, relation_logits, ENTITY_TYPES, RELATION_TYPES)
    relation_types = np.argmax(constrained_relations, axis=3)
    event_logits = np.einsum("cf,btf->btc", event_head, token_repr)
    time_logits = np.einsum("cf,bsrf->bsrc", time_head, relation_repr)
    return {
        "span_bounds": span_bounds,
        "entity_logits": entity_logits,
        "entity_types": entity_types,
        "relation_types": relation_types,
        "event_logits": event_logits,
        "time_logits": time_logits,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    gold = fixture["gold_entity_types"]
    pred = outputs["entity_types"]
    entity_accuracy = float(np.mean(pred == gold))
    non_null = gold != 0
    cascade_consistency = float(np.mean((outputs["relation_types"] == 0) | (pred[:, :, None] != 0)))
    return {
        "entity_accuracy": entity_accuracy,
        "non_null_entity_accuracy": float(np.mean(pred[non_null] == gold[non_null])),
        "cascade_consistency": cascade_consistency,
        "relation_tensor_shape": tuple(outputs["relation_types"].shape),
        "event_tensor_shape": tuple(outputs["event_logits"].shape),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "local_classifiers_without_constraints_produce_illegal_graphs",
            "note": "Schema-constrained decoding suppresses relations between incompatible entity types.",
        },
        {
            "case": "cascading_errors_dominate_joint_ie",
            "entity_types_first_sentence": outputs["entity_types"][0].tolist(),
            "relation_types_first_sentence": outputs["relation_types"][0].tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_4_span_graphs",
        "counterintuitive_insight": "Joint IE gains more from consistency propagation than from stronger local heads. The system should make impossible graphs hard to express.",
        "covered_claims": [
            "Chapter 20 now uses span proposals plus entity, relation, event, and time heads.",
            "Relation decoding is schema-constrained instead of unconstrained head-by-head argmax.",
        ],
        "omitted_claims": ["No external schema retrieval yet.", "No trigger-argument decoding loop yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="20",
        implementation_status="FULL",
        core_outputs={
            "span_bounds_shape": tuple(outputs["span_bounds"].shape),
            "entity_logits_shape": tuple(outputs["entity_logits"].shape),
            "relation_types_shape": tuple(outputs["relation_types"].shape),
            "event_logits_shape": tuple(outputs["event_logits"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "20",
    "title": "Information Extraction: Relations, Events, and Time",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
