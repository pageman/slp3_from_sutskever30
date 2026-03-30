from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import cross_entropy_from_probs, seeded_rng, stable_softmax


def build_fixture() -> dict[str, object]:
    contexts = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float64)
    targets = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    return {"contexts": contexts, "targets": targets, "hidden_dim": 5, "vocab_size": 4, "prediction_horizon": 2}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(27)
    contexts = fixture["contexts"]
    targets = fixture["targets"]
    hidden_dim = int(fixture["hidden_dim"])
    vocab_size = int(fixture["vocab_size"])
    horizon = int(fixture["prediction_horizon"])
    w_hidden = rng.normal(scale=0.18, size=(hidden_dim, contexts.shape[1]))
    b_hidden = np.zeros((hidden_dim,), dtype=np.float64)
    w_out = rng.normal(scale=0.18, size=(horizon, vocab_size, hidden_dim))
    b_out = np.zeros((horizon, vocab_size), dtype=np.float64)
    hidden = np.tanh(contexts @ w_hidden.T + b_hidden)
    logits = np.einsum("tvh,bh->btv", w_out, hidden) + b_out[None, :, :]
    probs = stable_softmax(logits, axis=2)
    losses = [cross_entropy_from_probs(probs[:, idx, :], targets[:, idx]) for idx in range(targets.shape[1])]
    greedy = np.argmax(probs, axis=2)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=2)
    return {
        "hidden": hidden,
        "logits": logits,
        "probs": probs,
        "losses": losses,
        "greedy": greedy,
        "entropy": entropy,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    targets = fixture["targets"]
    greedy = outputs["greedy"]
    return {
        "loss": float(np.mean(outputs["losses"])),
        "per_step_accuracy": [float(np.mean(greedy[:, idx] == targets[:, idx])) for idx in range(targets.shape[1])],
        "mean_entropy_per_step": outputs["entropy"].mean(axis=0).round(6).tolist(),
        "context_separability": float(np.linalg.norm(outputs["hidden"][0] - outputs["hidden"][1])),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "scale_mismatch_with_real_llms",
            "note": "This chapter is method-faithful only in miniature: there is no tokenizer, pretraining corpus, or long-context causal decoder stack here.",
        },
        {
            "case": "next_token_loss_hides_search_behavior",
            "greedy_predictions": outputs["greedy"].tolist(),
            "note": "The tiny model can score next tokens but does not expose beam-search or long-horizon decoding tradeoffs found in real LLM systems.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "For language models, the educational win is not parameter count but preserving the causal factorization and per-step uncertainty structure.",
        "covered_claims": [
            "A tiny autoregressive objective can be modeled with causal next-token heads in NumPy.",
            "Per-step entropy and accuracy expose prediction confidence beyond a single loss scalar.",
            "Context separation in hidden space matters even in a minimal causal model.",
        ],
        "omitted_claims": ["No large vocabulary tokenizer.", "No scaled self-attention decoder stack.", "No long-context generation loop."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="7",
        implementation_status="DIRECT",
        core_outputs={
            "logits_shape": tuple(outputs["logits"].shape),
            "greedy_predictions": outputs["greedy"].tolist(),
            "hidden_preview": outputs["hidden"][:2].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [27], "derivation_lineage": ["pageman/sutskever-30-implementations", "local direct NumPy wrapper"]},
        lesson_objectives=[
            "Preserve autoregressive next-token factorization in a minimal NumPy model.",
            "Measure per-step uncertainty and accuracy instead of only average loss.",
            "Show what an educational LLM analog can and cannot faithfully capture.",
        ],
        core_algorithms=["causal next-token prediction", "softmax language modeling loss", "greedy decoding", "hidden-state separability check"],
        minimal_dataset={
            "context_count": int(fixture["contexts"].shape[0]),
            "context_dim": int(fixture["contexts"].shape[1]),
            "prediction_horizon": int(fixture["prediction_horizon"]),
            "vocab_size": int(fixture["vocab_size"]),
        },
        reference_experiments=[
            {"name": "per_step_accuracy", "metric": "per_step_accuracy", "expected_signal": "causal heads can separate early and late token difficulty"},
            {"name": "entropy_profile", "metric": "mean_entropy_per_step", "expected_signal": "uncertainty stays visible even in a toy autoregressive setup"},
        ],
        book_vs_repo_gap="This chapter remains faithful only in miniature. It preserves causal prediction structure, but not realistic scale, tokenizer infrastructure, or transformer-decoder depth.",
    )


SPEC = {
    "key": "7",
    "title": "Large Language Models",
    "implementation_status": "DIRECT",
    "source_papers": (27,),
    "runner": run_chapter,
}
