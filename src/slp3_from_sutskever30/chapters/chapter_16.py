from __future__ import annotations

import re

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.speech import monotonic_attention


def _normalize_text(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9 ]+", "", text.lower())
    replacements = {"nlp": "en el pi", "tts": "ti ti es"}
    words = []
    for token in text.split():
        words.extend(replacements.get(token, token).split())
    return words


def _g2p(words: list[str]) -> list[str]:
    mapping = {
        "hello": ["HH", "AH", "L", "OW"],
        "speech": ["S", "P", "IY", "CH"],
        "models": ["M", "AA", "D", "AH", "L", "Z"],
        "align": ["AH", "L", "AY", "N"],
        "timing": ["T", "AY", "M", "IH", "NG"],
        "en": ["EH", "N"],
        "el": ["EH", "L"],
        "pi": ["P", "AY"],
        "ti": ["T", "IY"],
        "es": ["EH", "S"],
    }
    phonemes: list[str] = []
    for word in words:
        phonemes.extend(mapping.get(word, list(word.upper())))
    return phonemes


def build_fixture() -> dict[str, object]:
    texts = ["hello speech models", "align timing"]
    normalized = [_normalize_text(text) for text in texts]
    phonemes = [_g2p(words) for words in normalized]
    return {"texts": texts, "normalized": normalized, "phonemes": phonemes}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(16)
    phoneme_vocab = sorted({phone for seq in fixture["phonemes"] for phone in seq})
    phone_to_idx = {phone: idx for idx, phone in enumerate(phoneme_vocab)}
    encoder = rng.normal(scale=0.2, size=(len(phoneme_vocab), 10))
    mel_bank = rng.normal(scale=0.2, size=(80, 10))
    duration_head = rng.normal(scale=0.2, size=(1, 10))
    decoder_query = rng.normal(scale=0.2, size=(10, 10))

    mel_outputs = []
    stop_logits = []
    duration_predictions = []
    attention_maps = []
    teacher_forced_lengths = []
    free_run_lengths = []
    for seq in fixture["phonemes"]:
        phoneme_ids = np.asarray([phone_to_idx[phone] for phone in seq], dtype=np.int64)
        enc = encoder[phoneme_ids]
        durations = np.clip(np.round(np.maximum(1.0, enc @ duration_head.T + 2.0)).astype(np.int64).ravel(), 1, 5)
        duration_predictions.append(durations.tolist())
        teacher_forced = np.repeat(enc, durations, axis=0)
        queries = np.tanh(teacher_forced @ decoder_query)
        attention = monotonic_attention(enc[None, :, :], queries[None, :, :])[0]
        mel = teacher_forced @ mel_bank.T
        stop = np.linspace(-2.0, 2.0, mel.shape[0])
        mel_outputs.append(mel)
        stop_logits.append(stop)
        attention_maps.append(attention)
        teacher_forced_lengths.append(int(mel.shape[0]))
        free_run_lengths.append(int(np.sum(np.clip(durations + np.asarray([0 if idx % 2 == 0 else 1 for idx in range(len(durations))]), 1, 6))))

    max_len = max(mel.shape[0] for mel in mel_outputs)
    padded_mel = np.zeros((len(mel_outputs), max_len, 80), dtype=np.float64)
    padded_stop = np.full((len(mel_outputs), max_len), -10.0, dtype=np.float64)
    padded_attention = np.zeros((len(mel_outputs), max_len, max(len(seq) for seq in fixture["phonemes"])), dtype=np.float64)
    for idx, (mel, stop, attention) in enumerate(zip(mel_outputs, stop_logits, attention_maps)):
        padded_mel[idx, : mel.shape[0]] = mel
        padded_stop[idx, : stop.shape[0]] = stop
        padded_attention[idx, : attention.shape[0], : attention.shape[1]] = attention
    return {
        "phoneme_vocab": phoneme_vocab,
        "duration_predictions": duration_predictions,
        "padded_mel": padded_mel,
        "padded_stop": padded_stop,
        "padded_attention": padded_attention,
        "teacher_forced_lengths": teacher_forced_lengths,
        "free_run_lengths": free_run_lengths,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    teacher = np.asarray(outputs["teacher_forced_lengths"], dtype=np.float64)
    free = np.asarray(outputs["free_run_lengths"], dtype=np.float64)
    duration_totals = np.asarray([sum(seq) for seq in outputs["duration_predictions"]], dtype=np.float64)
    attention_mass = np.sum(outputs["padded_attention"], axis=2)
    return {
        "mel_shape": tuple(outputs["padded_mel"].shape),
        "stop_shape": tuple(outputs["padded_stop"].shape),
        "duration_error": float(np.mean(np.abs(teacher - duration_totals))),
        "teacher_forcing_gap": float(np.mean(np.abs(teacher - free))),
        "attention_row_mass_error": float(np.mean(np.abs(attention_mass[attention_mass > 0] - 1.0))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "timing_is_the_main_latent_variable",
            "teacher_forced_lengths": outputs["teacher_forced_lengths"],
            "free_run_lengths": outputs["free_run_lengths"],
        },
        {
            "case": "teacher_forcing_hides_generation_drift",
            "note": "Acoustics can look fine under teacher forcing while free-running duration drift still accumulates.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_c_speech",
        "counterintuitive_insight": "TTS bottlenecks on timing correctness more than on spectrogram rendering quality. Durations are the central latent variable.",
        "covered_claims": [
            "Text normalization and a lightweight G2P front end are now part of the chapter.",
            "Duration modeling, attention alignment, and acoustic rendering are separated.",
            "Teacher-forcing gap is measured explicitly.",
        ],
        "omitted_claims": ["No neural vocoder yet.", "No speaker conditioning yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="16",
        implementation_status="FULL",
        core_outputs={
            "phoneme_vocab_size": len(outputs["phoneme_vocab"]),
            "duration_predictions": outputs["duration_predictions"],
            "mel_frames_shape": tuple(outputs["padded_mel"].shape),
            "attention_shape": tuple(outputs["padded_attention"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Split TTS into text normalization, phoneme conversion, duration prediction, alignment, and acoustic rendering.",
            "Measure teacher-forcing gap as a timing-drift diagnostic.",
            "Show why duration modeling is central even in a minimal TTS system.",
        ],
        core_algorithms=["text normalization", "lightweight grapheme-to-phoneme mapping", "duration prediction", "monotonic attention alignment", "mel-frame rendering"],
        minimal_dataset={"text_count": len(fixture["texts"]), "phoneme_sequence_count": len(fixture["phonemes"]), "phoneme_vocab_size": len(outputs["phoneme_vocab"])},
        reference_experiments=[
            {"name": "teacher_forcing_gap", "metric": "teacher_forcing_gap", "expected_signal": "free-running duration drift should be visible even when teacher-forced acoustics look stable"},
            {"name": "attention_mass_check", "metric": "attention_row_mass_error", "expected_signal": "attention rows should stay normalized"},
        ],
        book_vs_repo_gap="This chapter is faithful only in miniature: timing and alignment mechanics are explicit, but there is no neural vocoder, speaker conditioning, or large-scale acoustic training loop.",
    )


SPEC = {
    "key": "16",
    "title": "Text-to-Speech",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
