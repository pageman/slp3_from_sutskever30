from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng, stable_softmax
from slp3_from_sutskever30.speech import alignment_entropy, cmvn, frame_signal, log_mel_spectrogram, monotonic_attention, power_spectrum, pre_emphasis


def _build_waves() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 8000, endpoint=False)
    return np.stack(
        [
            np.sin(2 * np.pi * 160 * t) + 0.3 * np.sin(2 * np.pi * 320 * t),
            np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t),
        ]
    ).astype(np.float64)


def _ctc_like_loss(frame_probs: np.ndarray, targets: np.ndarray, blank_id: int = 0) -> float:
    losses: list[float] = []
    for probs, target_seq in zip(frame_probs, targets):
        state_seq = [blank_id]
        for token in target_seq:
            state_seq.extend([int(token), blank_id])
        dp = np.full((probs.shape[0], len(state_seq)), -np.inf, dtype=np.float64)
        dp[0, 0] = np.log(probs[0, blank_id] + 1e-12)
        if len(state_seq) > 1:
            dp[0, 1] = np.log(probs[0, state_seq[1]] + 1e-12)
        for t in range(1, probs.shape[0]):
            for s, symbol in enumerate(state_seq):
                candidates = [dp[t - 1, s]]
                if s - 1 >= 0:
                    candidates.append(dp[t - 1, s - 1])
                if s - 2 >= 0 and symbol != blank_id and symbol != state_seq[s - 2]:
                    candidates.append(dp[t - 1, s - 2])
                score = np.max(candidates)
                dp[t, s] = score + np.log(probs[t, symbol] + 1e-12)
        losses.append(-float(np.max(dp[-1, -2:])))
    return float(np.mean(losses))


def build_fixture() -> dict[str, object]:
    waves = _build_waves()
    targets = np.asarray([[1, 2], [2, 1]], dtype=np.int64)
    return {"waves": waves, "targets": targets}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(15)
    emphasized = pre_emphasis(fixture["waves"])
    frames = frame_signal(emphasized, frame_length=320, hop_length=160)
    features = cmvn(log_mel_spectrogram(power_spectrum(frames), num_filters=24))
    acoustic_head = rng.normal(scale=0.2, size=(3, features.shape[2]))
    logits = np.einsum("cf,btf->btc", acoustic_head, features)
    probs = stable_softmax(logits, axis=2)
    queries = np.tile(np.mean(features, axis=1, keepdims=True), (1, fixture["targets"].shape[1], 1))
    attention = monotonic_attention(features, queries)
    beam_paths = np.argsort(-np.mean(probs[:, :, 1:], axis=1), axis=1)[:, :2] + 1
    return {
        "features": features,
        "logits": logits,
        "probs": probs,
        "attention": attention,
        "beam_paths": beam_paths,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    probs = outputs["probs"]
    targets = fixture["targets"]
    frame_entropy = alignment_entropy(probs)
    top_tokens = np.argmax(np.mean(probs[:, :, 1:], axis=1), axis=1) + 1
    return {
        "ctc_like_loss": _ctc_like_loss(probs, targets),
        "alignment_entropy": {
            "mean": float(np.mean(frame_entropy)),
            "std": float(np.std(frame_entropy)),
        },
        "beam_hit_rate": float(np.mean([int(target[0] in beam) for target, beam in zip(targets, outputs["beam_paths"])])),
        "top_token_accuracy": float(np.mean(top_tokens == targets[:, 0])),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "acoustic_model_and_ctc_are_distinct_objects",
            "note": "Low frame entropy does not automatically imply low sequence loss; the dynamic program and the acoustic model must be inspected separately.",
            "ctc_like_loss": _ctc_like_loss(outputs["probs"], fixture["targets"]),
        },
        {
            "case": "beam_search_can_hide_alignment_uncertainty",
            "beam_paths": outputs["beam_paths"].tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_3_speech_stack",
        "counterintuitive_insight": "ASR quality is limited as much by alignment uncertainty as by acoustic classification. Separating the acoustic model from the CTC dynamic program makes the real bottleneck visible.",
        "covered_claims": [
            "Chapter 15 now hooks into a chapter-14-style DSP frontend.",
            "A CTC-like loss and beam diagnostics can be exposed separately from acoustic logits.",
        ],
        "omitted_claims": ["No language model fusion yet.", "No true beam decoder with path merging yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="15",
        implementation_status="FULL",
        core_outputs={
            "feature_shape": tuple(outputs["features"].shape),
            "logits_shape": tuple(outputs["logits"].shape),
            "beam_paths": outputs["beam_paths"].tolist(),
            "attention_shape": tuple(outputs["attention"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [21], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "15",
    "title": "Automatic Speech Recognition",
    "implementation_status": "FULL",
    "source_papers": (21,),
    "runner": run_chapter,
}
