from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.structured_labeling import bio_constrained_decode, boundary_f1, segmentation_vs_label_errors, span_f1, token_accuracy


LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def build_fixture() -> dict[str, object]:
    token_ids = np.asarray(
        [
            [1, 2, 3, 4, 5],
            [2, 6, 7, 8, 0],
            [9, 10, 11, 12, 0],
            [13, 14, 15, 16, 17],
        ],
        dtype=np.int64,
    )
    char_ids = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6], [7, 3, 2], [8, 4, 3], [2, 1, 0]],
            [[5, 6, 7], [3, 7, 1], [4, 4, 2], [8, 1, 0], [0, 0, 0]],
            [[7, 1, 2], [5, 5, 3], [6, 4, 2], [3, 2, 1], [0, 0, 0]],
            [[3, 3, 7], [6, 1, 2], [4, 7, 3], [2, 5, 6], [1, 4, 7]],
        ],
        dtype=np.int64,
    )
    gold_tags = [
        ["B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
        ["B-LOC", "I-LOC", "O", "O", "O"],
        ["B-ORG", "I-ORG", "I-ORG", "O", "O"],
        ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"],
    ]
    return {"token_ids": token_ids, "char_ids": char_ids, "gold_tags": gold_tags}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(17)
    token_emb = rng.normal(scale=0.2, size=(18, 8))
    char_emb = rng.normal(scale=0.2, size=(9, 4))
    transition = rng.normal(scale=0.12, size=(8, 8))
    fusion = rng.normal(scale=0.18, size=(12, 8))
    classifier = rng.normal(scale=0.2, size=(len(LABELS), 12))
    token_features = token_emb[fixture["token_ids"]]
    char_features = np.mean(char_emb[fixture["char_ids"]], axis=2)
    features = np.concatenate([token_features, char_features], axis=2)
    token_context = np.einsum("ij,btj->bti", transition, token_features)
    contextual = np.tanh(features + np.einsum("ij,btj->bti", fusion, token_context))
    logits = np.einsum("cf,btf->btc", classifier, contextual)
    constrained_tags = bio_constrained_decode(logits, LABELS)
    raw_tags = [[LABELS[int(idx)] for idx in np.argmax(seq_logits, axis=1)] for seq_logits in logits]
    return {"logits": logits, "constrained_tags": constrained_tags, "raw_tags": raw_tags}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    gold = fixture["gold_tags"]
    predicted = outputs["constrained_tags"]
    return {
        "token_accuracy": token_accuracy(predicted, gold),
        "span_metrics": span_f1(predicted, gold),
        "boundary_f1": boundary_f1(predicted, gold),
        "error_breakdown": segmentation_vs_label_errors(predicted, gold),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "boundary_errors_cost_more_than_label_swaps",
            **segmentation_vs_label_errors(outputs["constrained_tags"], fixture["gold_tags"]),
        },
        {
            "case": "raw_argmax_breaks_bio_constraints",
            "raw_tags_first_sequence": outputs["raw_tags"][0],
            "constrained_tags_first_sequence": outputs["constrained_tags"][0],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_d_structure_and_ie",
        "counterintuitive_insight": "Boundary certainty matters more than tag identity. A span with the wrong edges is usually more damaging than a semantically nearby label confusion.",
        "covered_claims": [
            "Chapter 17 now uses constrained BIO decoding instead of raw argmax tags.",
            "Evaluation separates token accuracy from span and boundary behavior.",
        ],
        "omitted_claims": ["No CRF training loop yet.", "No external corpus loader yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="17",
        implementation_status="FULL",
        core_outputs={
            "tag_logits_shape": tuple(outputs["logits"].shape),
            "decoded_tags": outputs["constrained_tags"],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Decode BIO tags with explicit structural constraints instead of raw argmax.",
            "Separate token accuracy from span and boundary behavior.",
            "Show why segmentation errors dominate many sequence-labeling failures.",
        ],
        core_algorithms=["token and character feature fusion", "contextual sequence scoring", "BIO-constrained decoding", "span and boundary evaluation"],
        minimal_dataset={"sentence_count": int(fixture["token_ids"].shape[0]), "tokens_per_sentence": int(fixture["token_ids"].shape[1]), "label_count": len(LABELS)},
        reference_experiments=[
            {"name": "span_vs_token_metrics", "metric": ["token_accuracy", "span_metrics", "boundary_f1"], "expected_signal": "token accuracy overstates quality when boundaries drift"},
            {"name": "segmentation_error_breakdown", "metric": "error_breakdown", "expected_signal": "boundary mistakes dominate label swaps in structured tagging"},
        ],
        book_vs_repo_gap="This chapter is faithful in constrained decoding and evaluation, but still omits CRF training, corpus-scale feature extraction, and broader tag-set coverage.",
    )


SPEC = {
    "key": "17",
    "title": "Sequence Labeling for Parts of Speech and Named Entities",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
