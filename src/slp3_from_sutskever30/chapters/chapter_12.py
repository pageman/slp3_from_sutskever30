from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import cross_entropy_from_probs, seeded_rng, stable_softmax


def build_fixture() -> dict[str, object]:
    src = np.asarray([[0, 1, 2, 1], [1, 2, 0, 2], [2, 1, 1, 0], [0, 2, 1, 2]], dtype=np.int64)
    decoder = np.asarray([1, 0, 2, 1], dtype=np.int64)
    targets = np.asarray([2, 1, 0, 2], dtype=np.int64)
    return {"src": src, "decoder": decoder, "targets": targets, "vocab_size": 3, "hidden_dim": 5}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(12)
    src = fixture["src"]
    decoder = fixture["decoder"]
    targets = fixture["targets"]
    emb = rng.normal(scale=0.2, size=(fixture["vocab_size"], fixture["hidden_dim"]))
    dec_emb = emb[decoder]
    enc = emb[src]
    wa = rng.normal(scale=0.2, size=(fixture["hidden_dim"], fixture["hidden_dim"]))
    attention_scores = np.einsum("bth,bh->bt", enc @ wa.T, dec_emb)
    attention = stable_softmax(attention_scores, axis=1)
    context = np.einsum("bt,bth->bh", attention, enc)
    wout = rng.normal(scale=0.2, size=(fixture["vocab_size"], fixture["hidden_dim"] * 2))
    logits = np.concatenate([context, dec_emb], axis=1) @ wout.T
    probs = stable_softmax(logits, axis=1)
    return {"attention": attention, "context": context, "logits": logits, "probs": probs, "targets": targets}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    preds = np.argmax(outputs["probs"], axis=1)
    return {
        "loss": cross_entropy_from_probs(outputs["probs"], outputs["targets"]),
        "accuracy": float(np.mean(preds == outputs["targets"])),
        "attention_row_sums": np.sum(outputs["attention"], axis=1).round(6).tolist(),
        "context_norm": float(np.mean(np.linalg.norm(outputs["context"], axis=1))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "attention_without_search_is_incomplete_mt",
            "note": "Machine translation quality depends on decoding policy; this toy chapter scores only a single decoder step.",
        },
        {
            "case": "single_context_vector_limits_reordering",
            "attention_preview": outputs["attention"].round(4).tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "The pedagogical heart of neural MT is not BLEU-scale evaluation but making alignment pressure visible as a distinct object from token prediction.",
        "covered_claims": [
            "Encoder-decoder attention is exposed directly.",
            "A context vector and decoder state are combined for target prediction.",
            "Alignment normalization can be checked independently of translation accuracy.",
        ],
        "omitted_claims": ["No beam search.", "No multi-step decoder rollout.", "No BLEU or corpus-scale evaluation."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="12",
        implementation_status="ADAPTED",
        core_outputs={
            "attention_shape": tuple(outputs["attention"].shape),
            "attention_preview": outputs["attention"].round(4).tolist(),
            "logits_shape": tuple(outputs["logits"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [14], "derivation_lineage": ["pageman/sutskever-30-implementations", "adapted local NumPy seq2seq wrapper"]},
        lesson_objectives=[
            "Make encoder-decoder attention visible as an alignment object.",
            "Separate alignment normalization from target-token prediction quality.",
            "Show what is gained and lost when translation is reduced to a single-step toy decoder.",
        ],
        core_algorithms=["embedding lookup", "encoder-decoder attention", "context-vector construction", "softmax translation prediction"],
        minimal_dataset={"batch_size": int(fixture["src"].shape[0]), "source_length": int(fixture["src"].shape[1]), "vocab_size": int(fixture["vocab_size"])},
        reference_experiments=[
            {"name": "attention_normalization", "metric": "attention_row_sums", "expected_signal": "alignment rows sum to one"},
            {"name": "single_step_translation_accuracy", "metric": "accuracy", "expected_signal": "toy seq2seq setup separates simple target classes"},
        ],
        book_vs_repo_gap="This chapter is adapted and only faithful in miniature: it demonstrates alignment mechanics but omits decoder rollout, beam search, and corpus-level MT evaluation.",
    )


SPEC = {
    "key": "12",
    "title": "Machine Translation",
    "implementation_status": "ADAPTED",
    "source_papers": (14,),
    "runner": run_chapter,
}
