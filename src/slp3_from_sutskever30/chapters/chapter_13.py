from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import cross_entropy_from_probs, one_hot, seeded_rng, stable_softmax


def build_fixture() -> dict[str, object]:
    inputs = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    targets = np.asarray([1, 2, 1, 0, 1], dtype=np.int64)
    return {"inputs": inputs, "targets": targets, "vocab_size": 5, "hidden_dim": 6}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(13)
    vocab = int(fixture["vocab_size"])
    inputs = fixture["inputs"]
    targets = fixture["targets"]
    xs = one_hot(inputs, vocab)
    hidden_dim = int(fixture["hidden_dim"])
    wxh = rng.normal(scale=0.2, size=(hidden_dim, vocab))
    whh = rng.normal(scale=0.2, size=(hidden_dim, hidden_dim))
    why = rng.normal(scale=0.2, size=(vocab, hidden_dim))
    h = np.zeros((hidden_dim,), dtype=np.float64)
    rnn_hidden = []
    rnn_logits = []
    for x_t in xs:
        h = np.tanh(wxh @ x_t + whh @ h)
        rnn_hidden.append(h.copy())
        rnn_logits.append(why @ h)
    rnn_hidden = np.stack(rnn_hidden)
    rnn_probs = stable_softmax(np.stack(rnn_logits), axis=1)
    wf = rng.normal(scale=0.2, size=(hidden_dim, vocab + hidden_dim))
    wi = rng.normal(scale=0.2, size=(hidden_dim, vocab + hidden_dim))
    wo = rng.normal(scale=0.2, size=(hidden_dim, vocab + hidden_dim))
    wc = rng.normal(scale=0.2, size=(hidden_dim, vocab + hidden_dim))
    h = np.zeros((hidden_dim,), dtype=np.float64)
    c = np.zeros((hidden_dim,), dtype=np.float64)
    lstm_hidden = []
    lstm_cell = []
    lstm_logits = []
    for x_t in xs:
        concat = np.concatenate([x_t, h])
        f = 1.0 / (1.0 + np.exp(-(wf @ concat)))
        i = 1.0 / (1.0 + np.exp(-(wi @ concat)))
        o = 1.0 / (1.0 + np.exp(-(wo @ concat)))
        g = np.tanh(wc @ concat)
        c = f * c + i * g
        h = o * np.tanh(c)
        lstm_hidden.append(h.copy())
        lstm_cell.append(c.copy())
        lstm_logits.append(why @ h)
    lstm_hidden = np.stack(lstm_hidden)
    lstm_cell = np.stack(lstm_cell)
    lstm_probs = stable_softmax(np.stack(lstm_logits), axis=1)
    return {
        "targets": targets,
        "rnn_probs": rnn_probs,
        "lstm_probs": lstm_probs,
        "rnn_hidden": rnn_hidden,
        "lstm_hidden": lstm_hidden,
        "lstm_cell": lstm_cell,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    targets = outputs["targets"]
    rnn_preds = np.argmax(outputs["rnn_probs"], axis=1)
    lstm_preds = np.argmax(outputs["lstm_probs"], axis=1)
    return {
        "rnn_loss": cross_entropy_from_probs(outputs["rnn_probs"], targets),
        "lstm_loss": cross_entropy_from_probs(outputs["lstm_probs"], targets),
        "rnn_accuracy": float(np.mean(rnn_preds == targets)),
        "lstm_accuracy": float(np.mean(lstm_preds == targets)),
        "state_norm_gap": float(np.mean(np.linalg.norm(outputs["lstm_hidden"], axis=1) - np.linalg.norm(outputs["rnn_hidden"], axis=1))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "rnn_memory_decay",
            "note": "A vanilla RNN can mix current evidence and past state, but it has no explicit cell state to preserve information under repeated updates.",
        },
        {
            "case": "lstm_not_magic",
            "lstm_cell_preview": outputs["lstm_cell"][:3].round(4).tolist(),
            "note": "The LSTM improves memory control, but still depends on training signal and task structure.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "The key conceptual jump from RNN to LSTM is not accuracy but controllable state persistence. The cell state is the lesson.",
        "covered_claims": [
            "Vanilla recurrent updates and gated recurrent updates are both explicit.",
            "RNN and LSTM losses can be compared on the same toy sequence task.",
            "Hidden-state and cell-state trajectories are inspectable as first-class objects.",
        ],
        "omitted_claims": ["No BPTT training loop.", "No long-sequence benchmark.", "No GRU comparison."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="13",
        implementation_status="ADAPTED",
        core_outputs={
            "rnn_hidden_preview": outputs["rnn_hidden"][:3].round(4).tolist(),
            "lstm_hidden_preview": outputs["lstm_hidden"][:3].round(4).tolist(),
            "lstm_cell_preview": outputs["lstm_cell"][:3].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [2, 3], "derivation_lineage": ["pageman/sutskever-30-implementations", "adapted local NumPy recurrent wrapper"]},
        lesson_objectives=[
            "Contrast vanilla recurrent updates with gated memory updates.",
            "Track hidden and cell dynamics directly instead of only comparing losses.",
            "Explain why state persistence is the real conceptual gain of LSTMs.",
        ],
        core_algorithms=["vanilla RNN recurrence", "LSTM gating", "sequence softmax prediction", "state trajectory comparison"],
        minimal_dataset={"sequence_length": int(fixture["inputs"].shape[0]), "vocab_size": int(fixture["vocab_size"]), "hidden_dim": int(fixture["hidden_dim"])},
        reference_experiments=[
            {"name": "rnn_vs_lstm_loss", "metric": ["rnn_loss", "lstm_loss"], "expected_signal": "gated memory is usually more stable"},
            {"name": "state_norm_gap", "metric": "state_norm_gap", "expected_signal": "hidden-state dynamics differ between plain and gated recurrence"},
        ],
        book_vs_repo_gap="This chapter is adapted and method-faithful in miniature, but it still omits training-through-time, longer sequence stress tests, and broader recurrent architecture comparisons.",
    )


SPEC = {
    "key": "13",
    "title": "RNNs and LSTMs",
    "implementation_status": "ADAPTED",
    "source_papers": (2, 3),
    "runner": run_chapter,
}
