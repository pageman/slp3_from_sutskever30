from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.parsing import attachment_scores, crossing_count, mst_decode, projective_arc_decode


RELATIONS = ["root", "nsubj", "obj", "det", "obl"]


def build_fixture() -> dict[str, object]:
    token_ids = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 6, 7, 8, 9, 10],
            [0, 11, 12, 13, 14, 15],
            [0, 16, 17, 18, 19, 20],
        ],
        dtype=np.int64,
    )
    gold_heads = np.asarray(
        [
            [0, 2, 0, 4, 2, 2],
            [0, 2, 0, 2, 3, 2],
            [0, 2, 0, 2, 5, 3],
            [0, 2, 0, 4, 2, 4],
        ],
        dtype=np.int64,
    )
    gold_labels = np.asarray(
        [
            [0, 1, 0, 3, 2, 4],
            [0, 1, 0, 2, 4, 4],
            [0, 1, 0, 2, 3, 4],
            [0, 1, 0, 3, 2, 4],
        ],
        dtype=np.int64,
    )
    return {"token_ids": token_ids, "gold_heads": gold_heads, "gold_labels": gold_labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(19)
    token_emb = rng.normal(scale=0.2, size=(21, 8))
    head_proj = rng.normal(scale=0.2, size=(8, 8))
    dep_proj = rng.normal(scale=0.2, size=(8, 8))
    rel_head = rng.normal(scale=0.2, size=(len(RELATIONS), 16))
    states = token_emb[fixture["token_ids"]]
    heads = states @ head_proj
    deps = states @ dep_proj
    arc_scores = np.einsum("bih,bjh->bij", heads, deps)
    for batch_idx in range(arc_scores.shape[0]):
        np.fill_diagonal(arc_scores[batch_idx], -1e9)
    proj_heads = projective_arc_decode(arc_scores)
    mst_heads = mst_decode(arc_scores)
    pair_repr = np.concatenate([heads[np.arange(heads.shape[0])[:, None], mst_heads], deps], axis=2)
    relation_logits = np.einsum("cf,btf->btc", rel_head, pair_repr)
    pred_labels = np.argmax(relation_logits, axis=2)
    pred_labels[:, 0] = 0
    return {"arc_scores": arc_scores, "projective_heads": proj_heads, "mst_heads": mst_heads, "relation_logits": relation_logits, "pred_labels": pred_labels}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    proj_scores = attachment_scores(outputs["projective_heads"], fixture["gold_heads"], outputs["pred_labels"], fixture["gold_labels"])
    mst_scores = attachment_scores(outputs["mst_heads"], fixture["gold_heads"], outputs["pred_labels"], fixture["gold_labels"])
    return {
        "projective_scores": proj_scores,
        "mst_scores": mst_scores,
        "crossing_counts": [crossing_count(heads) for heads in outputs["mst_heads"]],
        "arc_scores_shape": tuple(outputs["arc_scores"].shape),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "wrong_but_self_consistent_trees_exist",
            "mst_heads_first_sentence": outputs["mst_heads"][0].tolist(),
            "gold_heads_first_sentence": fixture["gold_heads"][0].tolist(),
        },
        {
            "case": "projective_and_nonprojective_decoders_diverge",
            "projective_first_sentence": outputs["projective_heads"][0].tolist(),
            "mst_first_sentence": outputs["mst_heads"][0].tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_4_structured_prediction_b",
        "counterintuitive_insight": "The dangerous dependency error is not just a wrong arc. It is a wrong tree that remains globally coherent and therefore looks plausible unless you inspect structure explicitly.",
        "covered_claims": [
            "Chapter 19 now compares projective and MST-style arc decoding.",
            "Evaluation includes UAS/LAS and crossing diagnostics.",
        ],
        "omitted_claims": ["No transition-based parser yet.", "No length-bucketed error analysis yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="19",
        implementation_status="FULL",
        core_outputs={
            "arc_scores_shape": tuple(outputs["arc_scores"].shape),
            "projective_heads": outputs["projective_heads"].tolist(),
            "mst_heads": outputs["mst_heads"].tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "19",
    "title": "Dependency Parsing",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
