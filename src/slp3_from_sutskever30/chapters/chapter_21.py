from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.structured_labeling import role_constrained_decode, role_metrics


ROLE_LABELS = ["NULL", "ARG0", "ARG1", "ARGM-TMP", "ARGM-LOC", "ARG2"]


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
    predicate_indices = np.asarray([1, 2, 3, 4], dtype=np.int64)
    span_starts = np.asarray([[0, 2, 4], [0, 3, 4], [1, 3, 4], [0, 2, 4]], dtype=np.int64)
    span_ends = np.asarray([[1, 3, 5], [2, 4, 5], [2, 4, 5], [1, 3, 5]], dtype=np.int64)
    gold_roles = [
        ["ARG0", "ARG1", "ARGM-TMP"],
        ["ARG0", "ARG1", "NULL"],
        ["ARGM-LOC", "ARG1", "ARG0"],
        ["ARG0", "NULL", "ARG1"],
    ]
    return {
        "token_ids": token_ids,
        "predicate_indices": predicate_indices,
        "span_starts": span_starts,
        "span_ends": span_ends,
        "gold_roles": gold_roles,
    }


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(21)
    token_emb = rng.normal(scale=0.2, size=(25, 10))
    predicate_gate = rng.normal(scale=0.2, size=(10, 10))
    role_head = rng.normal(scale=0.2, size=(len(ROLE_LABELS), 20))
    token_repr = token_emb[fixture["token_ids"]]
    predicate_repr = token_repr[np.arange(token_repr.shape[0]), fixture["predicate_indices"]]
    conditioned_predicate = np.tanh(predicate_repr @ predicate_gate)
    span_repr = []
    for batch_idx in range(token_repr.shape[0]):
        spans = []
        for start, end in zip(fixture["span_starts"][batch_idx], fixture["span_ends"][batch_idx]):
            spans.append(np.mean(token_repr[batch_idx, start : end + 1], axis=0))
        span_repr.append(spans)
    span_repr_arr = np.asarray(span_repr)
    features = np.concatenate([np.repeat(conditioned_predicate[:, None, :], span_repr_arr.shape[1], axis=1), span_repr_arr], axis=2)
    role_logits = np.einsum("cf,bsf->bsc", role_head, features)
    decoded_roles = role_constrained_decode(role_logits, ROLE_LABELS)
    raw_roles = [[ROLE_LABELS[int(idx)] for idx in np.argmax(seq_logits, axis=1)] for seq_logits in role_logits]
    return {"role_logits": role_logits, "decoded_roles": decoded_roles, "raw_roles": raw_roles}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "role_metrics": role_metrics(outputs["decoded_roles"], fixture["gold_roles"]),
        "predicate_centric_accuracy": float(
            np.mean([all(pred == gold for pred, gold in zip(pred_seq, gold_seq)) for pred_seq, gold_seq in zip(outputs["decoded_roles"], fixture["gold_roles"])])
        ),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    duplicate_core = []
    for seq in outputs["raw_roles"]:
        cores = [role for role in seq if role.startswith("ARG") and role != "ARGM-TMP" and role != "ARGM-LOC"]
        if len(cores) != len(set(cores)):
            duplicate_core.append(seq)
    return [
        {
            "case": "raw_argmax_reuses_core_roles",
            "duplicate_examples": duplicate_core,
        },
        {
            "case": "predicate_conditioning_changes_role_decisions",
            "decoded_roles_first_sequence": outputs["decoded_roles"][0],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_4_structured_prediction_a",
        "counterintuitive_insight": "Semantic role labeling is less about tagging spans independently and more about enforcing predicate-centric role economy. Wrong but self-consistent role sets are the real failure mode.",
        "covered_claims": [
            "Chapter 21 now uses predicate-conditioned span scoring.",
            "Role decoding enforces simple core-role uniqueness constraints.",
        ],
        "omitted_claims": ["No null-argument recovery yet.", "No PropBank-style frame lexicon yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="21",
        implementation_status="FULL",
        core_outputs={
            "role_logits_shape": tuple(outputs["role_logits"].shape),
            "decoded_roles": outputs["decoded_roles"],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "21",
    "title": "Semantic Role Labeling and Argument Structure",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
