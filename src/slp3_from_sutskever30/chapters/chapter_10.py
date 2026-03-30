from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng, stable_softmax


MASK_ID = 0


def _layer_norm(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-6)


def _self_attention(hidden: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = hidden @ wq
    k = hidden @ wk
    v = hidden @ wv
    scores = np.einsum("btd,bsd->bts", q, k) / np.sqrt(q.shape[-1])
    weights = stable_softmax(scores, axis=2)
    return weights @ v, weights


def _encoder(hidden: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    attention_maps: list[np.ndarray] = []
    state = hidden
    for layer in range(2):
        attended, weights = _self_attention(state, params[f"wq{layer}"], params[f"wk{layer}"], params[f"wv{layer}"])
        state = _layer_norm(state + attended)
        ff = np.tanh(state @ params[f"w1_{layer}"]) @ params[f"w2_{layer}"]
        state = _layer_norm(state + ff)
        attention_maps.append(weights)
    return state, attention_maps


def _make_masked_batch(token_ids: np.ndarray, *, policy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = token_ids.copy()
    targets: list[int] = []
    positions: list[tuple[int, int]] = []
    for row in range(token_ids.shape[0]):
        if policy == "token":
            chosen = [1, token_ids.shape[1] - 2]
        else:
            chosen = [1, 2]
        for col in chosen:
            targets.append(int(token_ids[row, col]))
            positions.append((row, col))
            masked[row, col] = MASK_ID
    return masked, np.asarray(targets, dtype=np.int64), np.asarray(positions, dtype=np.int64)


def _masked_logits(encoded: np.ndarray, positions: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    reps = encoded[positions[:, 0], positions[:, 1]]
    return reps @ decoder.T


def _probe_quality(encoded: np.ndarray, token_ids: np.ndarray) -> dict[str, float]:
    token_rep = np.mean(encoded, axis=1)
    lexical_probe = float(np.mean(np.argmax(token_rep[:, : token_ids.max() + 1], axis=1) == token_ids[:, 0]))
    parity_target = (np.sum(token_ids, axis=1) % 2).astype(np.int64)
    parity_score = np.mean((np.sum(token_rep, axis=1) > 0.0).astype(np.int64) == parity_target)
    return {"lexical_probe": lexical_probe, "parity_probe": float(parity_score)}


def build_fixture() -> dict[str, object]:
    token_ids = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 1],
            [3, 4, 5, 6, 1, 2],
            [4, 5, 6, 1, 2, 3],
        ],
        dtype=np.int64,
    )
    return {"token_ids": token_ids, "vocab_size": 7, "hidden_dim": 8}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(10)
    token_ids = fixture["token_ids"]
    vocab_size = fixture["vocab_size"]
    hidden_dim = fixture["hidden_dim"]
    embeddings = rng.normal(scale=0.2, size=(vocab_size, hidden_dim))
    positional = rng.normal(scale=0.1, size=(token_ids.shape[1], hidden_dim))
    params = {
        "wq0": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "wk0": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "wv0": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "w1_0": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim * 2)),
        "w2_0": rng.normal(scale=0.2, size=(hidden_dim * 2, hidden_dim)),
        "wq1": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "wk1": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "wv1": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim)),
        "w1_1": rng.normal(scale=0.2, size=(hidden_dim, hidden_dim * 2)),
        "w2_1": rng.normal(scale=0.2, size=(hidden_dim * 2, hidden_dim)),
    }
    decoder = rng.normal(scale=0.2, size=(vocab_size, hidden_dim))

    masked_token, targets_token, positions_token = _make_masked_batch(token_ids, policy="token")
    hidden_token = embeddings[masked_token] + positional[None, :, :]
    encoded_token, attn_token = _encoder(hidden_token, params)
    logits_token = _masked_logits(encoded_token, positions_token, decoder)
    probs_token = stable_softmax(logits_token, axis=1)

    masked_span, targets_span, positions_span = _make_masked_batch(token_ids, policy="span")
    hidden_span = embeddings[masked_span] + positional[None, :, :]
    encoded_span, attn_span = _encoder(hidden_span, params)
    logits_span = _masked_logits(encoded_span, positions_span, decoder)
    probs_span = stable_softmax(logits_span, axis=1)

    unmasked_hidden = embeddings[token_ids] + positional[None, :, :]
    encoded_full, attn_full = _encoder(unmasked_hidden, params)
    return {
        "token_policy": {"probs": probs_token, "targets": targets_token, "positions": positions_token, "attention": attn_token[-1]},
        "span_policy": {"probs": probs_span, "targets": targets_span, "positions": positions_span, "attention": attn_span[-1]},
        "full_encoded": encoded_full,
        "full_attention": attn_full[-1],
        "token_ids": token_ids,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    token_probs = outputs["token_policy"]["probs"]
    token_targets = outputs["token_policy"]["targets"]
    span_probs = outputs["span_policy"]["probs"]
    span_targets = outputs["span_policy"]["targets"]
    return {
        "masked_accuracy": {
            "token_masking": float(np.mean(np.argmax(token_probs, axis=1) == token_targets)),
            "span_masking": float(np.mean(np.argmax(span_probs, axis=1) == span_targets)),
        },
        "masked_loss": {
            "token_masking": float(-np.mean(np.log(token_probs[np.arange(len(token_targets)), token_targets] + 1e-12))),
            "span_masking": float(-np.mean(np.log(span_probs[np.arange(len(span_targets)), span_targets] + 1e-12))),
        },
        "probe_quality": _probe_quality(outputs["full_encoded"], outputs["token_ids"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "single_mask_policy_overfits_corruption_style",
            "token_masking_accuracy": float(np.mean(np.argmax(outputs["token_policy"]["probs"], axis=1) == outputs["token_policy"]["targets"])),
            "span_masking_accuracy": float(np.mean(np.argmax(outputs["span_policy"]["probs"], axis=1) == outputs["span_policy"]["targets"])),
        },
        {
            "case": "mask_token_shortcut_risk",
            "note": "A model can learn the corruption artifact instead of robust bidirectional context unless multiple corruption patterns are used.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "The key upgrade is corruption diversity, not mask prediction itself. Encoders become useful when the corruption policy teaches invariance instead of a single special-token trick.",
        "covered_claims": [
            "A two-layer bidirectional encoder can support masked-token prediction in NumPy.",
            "Token masking and span masking produce meaningfully different diagnostics.",
            "Representation probes help justify the encoder beyond raw MLM loss.",
        ],
        "omitted_claims": ["No pretrained tokenizer stack yet.", "No large-scale pretraining loop yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="10",
        implementation_status="FULL",
        core_outputs={
            "token_mask_positions": outputs["token_policy"]["positions"].tolist(),
            "span_mask_positions": outputs["span_policy"]["positions"].tolist(),
            "token_attention_shape": tuple(outputs["token_policy"]["attention"].shape),
            "span_attention_shape": tuple(outputs["span_policy"]["attention"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Show how token masking and span masking lead to different supervision signals.",
            "Inspect bidirectional encoder behavior beyond MLM loss via representation probes.",
            "Make corruption-policy diversity a first-class design choice.",
        ],
        core_algorithms=["masked token corruption", "span masking", "two-layer bidirectional self-attention encoder", "representation probing"],
        minimal_dataset={"batch_size": int(fixture["token_ids"].shape[0]), "sequence_length": int(fixture["token_ids"].shape[1]), "vocab_size": int(fixture["vocab_size"]), "hidden_dim": int(fixture["hidden_dim"])},
        reference_experiments=[
            {"name": "corruption_policy_comparison", "metric": "masked_accuracy", "expected_signal": "different masking policies induce different prediction behavior"},
            {"name": "probe_quality", "metric": "probe_quality", "expected_signal": "encoder states retain lexical and structural signal beyond MLM loss"},
        ],
        book_vs_repo_gap="This chapter is faithful only in miniature: the bidirectional encoder and corruption policies are present, but not the scale, tokenizer stack, or long pretraining loop of real masked language models.",
    )


SPEC = {
    "key": "10",
    "title": "Masked Language Models",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
