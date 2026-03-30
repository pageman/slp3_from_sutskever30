from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import cross_entropy_from_probs, seeded_rng, stable_softmax


def build_fixture() -> dict[str, object]:
    return {"batch_size": 4, "seq_len": 3, "model_dim": 4, "num_classes": 3}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(13)
    seqs = rng.normal(size=(fixture["batch_size"], fixture["seq_len"], fixture["model_dim"]))
    targets = np.asarray([0, 1, 2, 1], dtype=np.int64)
    wq = rng.normal(scale=0.2, size=(fixture["model_dim"], fixture["model_dim"]))
    wk = rng.normal(scale=0.2, size=(fixture["model_dim"], fixture["model_dim"]))
    wv = rng.normal(scale=0.2, size=(fixture["model_dim"], fixture["model_dim"]))
    wo = rng.normal(scale=0.2, size=(fixture["num_classes"], fixture["model_dim"]))
    q = seqs @ wq.T
    k = seqs @ wk.T
    v = seqs @ wv.T
    scores = np.einsum("btd,bsd->bts", q, k) / np.sqrt(q.shape[-1])
    weights = stable_softmax(scores, axis=2)
    attended = weights @ v
    pooled = np.mean(attended, axis=1)
    logits = pooled @ wo.T
    probs = stable_softmax(logits, axis=1)
    return {"seqs": seqs, "targets": targets, "attention": weights, "attended": attended, "logits": logits, "probs": probs}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    preds = np.argmax(outputs["probs"], axis=1)
    diagonal_attention = np.mean(np.diagonal(outputs["attention"], axis1=1, axis2=2))
    return {
        "loss": cross_entropy_from_probs(outputs["probs"], outputs["targets"]),
        "accuracy": float(np.mean(preds == outputs["targets"])),
        "mean_diagonal_attention": float(diagonal_attention),
        "attention_row_sums": np.sum(outputs["attention"], axis=2).round(6).mean(axis=0).tolist(),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "attention_is_not_explanation_by_default",
            "note": "Attention weights are useful mechanics, but they do not automatically justify a model decision.",
        },
        {
            "case": "missing_positional_information",
            "note": "This toy chapter omits explicit positional encoding, so permutation sensitivity is under-modeled relative to textbook transformer presentations.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "The core transformer lesson is not multi-head scale but the separation between content mixing, pooling, and output prediction.",
        "covered_claims": [
            "Scaled dot-product attention is implemented directly in NumPy.",
            "Attention normalization can be inspected separately from classification loss.",
            "A tiny transformer block already shows the content-mixing mechanics that matter pedagogically.",
        ],
        "omitted_claims": ["No multi-head split.", "No positional encoding.", "No residual/FFN stack depth."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="8",
        implementation_status="DIRECT",
        core_outputs={
            "attention_shape": tuple(outputs["attention"].shape),
            "logits_shape": tuple(outputs["logits"].shape),
            "attention_preview": outputs["attention"][0].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [13], "derivation_lineage": ["pageman/sutskever-30-implementations", "local direct NumPy wrapper"]},
        lesson_objectives=[
            "Implement scaled dot-product attention directly in NumPy.",
            "Inspect attention normalization and pooled predictions separately.",
            "Make explicit which transformer components are present and which are omitted in the toy reference.",
        ],
        core_algorithms=["query-key-value projections", "scaled dot-product attention", "sequence pooling", "softmax classification"],
        minimal_dataset={"batch_size": int(fixture["batch_size"]), "seq_len": int(fixture["seq_len"]), "model_dim": int(fixture["model_dim"]), "num_classes": int(fixture["num_classes"])},
        reference_experiments=[
            {"name": "attention_normalization", "metric": "attention_row_sums", "expected_signal": "attention rows stay normalized"},
            {"name": "classification_accuracy", "metric": "accuracy", "expected_signal": "toy transformer separates simple sequence classes"},
        ],
        book_vs_repo_gap="This chapter is faithful in method but still miniature: no multi-head composition, positional encodings, residual stack, or decoder-only generation path.",
    )


SPEC = {
    "key": "8",
    "title": "Transformers",
    "implementation_status": "DIRECT",
    "source_papers": (13,),
    "runner": run_chapter,
}
