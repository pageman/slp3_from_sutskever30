from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng, stable_softmax
from slp3_from_sutskever30.speech import alignment_entropy, cmvn, delta_features, frame_signal, log_mel_spectrogram, power_spectrum, pre_emphasis


def _build_waves() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    base = np.stack(
        [
            np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 240 * t),
            np.sin(2 * np.pi * 180 * t) + 0.20 * np.sin(2 * np.pi * 360 * t),
            np.sin(2 * np.pi * 240 * t) + 0.15 * np.sin(2 * np.pi * 480 * t),
            np.sin(2 * np.pi * 300 * t) + 0.10 * np.sin(2 * np.pi * 600 * t),
        ]
    )
    envelope = np.linspace(1.0, 0.6, t.shape[0])[None, :]
    return (base * envelope).astype(np.float64)


def build_fixture() -> dict[str, object]:
    waves = _build_waves()
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    return {"waves": waves, "labels": labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(14)
    waves = fixture["waves"]
    emphasized = pre_emphasis(waves)
    frames = frame_signal(emphasized, frame_length=400, hop_length=160)
    power = power_spectrum(frames)
    log_mel = log_mel_spectrogram(power, num_filters=40)
    deltas = delta_features(log_mel)
    cmvn_mel = cmvn(log_mel)
    stacked = np.concatenate([cmvn_mel, deltas], axis=2)
    raw_energy = np.mean(frames**2, axis=2)
    spectral_energy = np.mean(power, axis=2)
    acoustic_centroids = np.stack(
        [
            np.mean(raw_energy, axis=1),
            np.mean(spectral_energy, axis=1),
            np.mean(cmvn_mel[:, :, :10], axis=(1, 2)),
            np.mean(deltas[:, :, :10], axis=(1, 2)),
        ],
        axis=1,
    )
    probe_weights = rng.normal(scale=0.25, size=(4, acoustic_centroids.shape[1]))
    probe_logits = acoustic_centroids @ probe_weights.T
    probe_probs = stable_softmax(probe_logits, axis=1)
    frame_posteriors = stable_softmax(log_mel[:, :, :4], axis=2)
    return {
        "frames": frames,
        "power": power,
        "log_mel": log_mel,
        "deltas": deltas,
        "cmvn_mel": cmvn_mel,
        "stacked": stacked,
        "probe_probs": probe_probs,
        "frame_posteriors": frame_posteriors,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    labels = fixture["labels"]
    probe_probs = outputs["probe_probs"]
    entropy = alignment_entropy(outputs["frame_posteriors"])
    return {
        "feature_shapes": {
            "frames": tuple(outputs["frames"].shape),
            "log_mel": tuple(outputs["log_mel"].shape),
            "stacked": tuple(outputs["stacked"].shape),
        },
        "probe_accuracy": float(np.mean(np.argmax(probe_probs, axis=1) == labels)),
        "alignment_entropy": {
            "mean": float(np.mean(entropy)),
            "std": float(np.std(entropy)),
        },
        "cmvn_mean_abs": float(np.mean(np.abs(np.mean(outputs["cmvn_mel"], axis=1)))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "spectrogram_appearance_is_not_enough",
            "note": "Features that look smooth can still have high frame-level alignment entropy and be poor for downstream alignment.",
            "alignment_entropy_mean": float(np.mean(alignment_entropy(outputs["frame_posteriors"]))),
        },
        {
            "case": "raw_energy_collapses_phonetic_contrast",
            "note": "Raw frame energy alone loses much of the useful contrast preserved by mel plus delta features.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_3_speech_stack",
        "counterintuitive_insight": "Speech features should be judged by alignment entropy, not by how visually plausible the spectrogram looks to humans.",
        "covered_claims": [
            "A NumPy DSP stack can include pre-emphasis, framing, spectrum, mel, deltas, and CMVN.",
            "Feature pipelines can be compared through downstream probe behavior and alignment entropy.",
        ],
        "omitted_claims": ["No MFCC/DCT path yet.", "No real phone recognizer yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="14",
        implementation_status="FULL",
        core_outputs={
            "log_mel_shape": tuple(outputs["log_mel"].shape),
            "stacked_feature_shape": tuple(outputs["stacked"].shape),
            "sample_probe_probs": outputs["probe_probs"].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "14",
    "title": "Phonetics and Speech Feature Extraction",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
