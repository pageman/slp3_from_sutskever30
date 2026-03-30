from __future__ import annotations

import numpy as np


def pre_emphasis(waveforms: np.ndarray, *, coeff: float = 0.97) -> np.ndarray:
    emphasized = waveforms.copy()
    emphasized[:, 1:] = waveforms[:, 1:] - coeff * waveforms[:, :-1]
    return emphasized


def frame_signal(waveforms: np.ndarray, *, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
    num_frames = 1 + max(0, (waveforms.shape[1] - frame_length) // hop_length)
    frames = np.zeros((waveforms.shape[0], num_frames, frame_length), dtype=np.float64)
    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        stop = start + frame_length
        frames[:, frame_idx, :] = waveforms[:, start:stop]
    return frames


def hann_window(frame_length: int) -> np.ndarray:
    return np.hanning(frame_length).astype(np.float64)


def power_spectrum(frames: np.ndarray) -> np.ndarray:
    windowed = frames * hann_window(frames.shape[-1])[None, None, :]
    spectrum = np.fft.rfft(windowed, axis=2)
    return np.abs(spectrum) ** 2


def mel_filterbank(num_bins: int, num_filters: int) -> np.ndarray:
    centers = np.linspace(0, num_bins - 1, num_filters + 2)
    filters = np.zeros((num_filters, num_bins), dtype=np.float64)
    for filt in range(num_filters):
        left, center, right = centers[filt : filt + 3]
        for bin_idx in range(num_bins):
            if left <= bin_idx <= center:
                filters[filt, bin_idx] = (bin_idx - left) / max(center - left, 1e-12)
            elif center < bin_idx <= right:
                filters[filt, bin_idx] = (right - bin_idx) / max(right - center, 1e-12)
    return filters


def log_mel_spectrogram(power: np.ndarray, *, num_filters: int = 40) -> np.ndarray:
    bank = mel_filterbank(power.shape[-1], num_filters)
    mel = np.einsum("mf,btf->btm", bank, power)
    return np.log1p(mel)


def delta_features(features: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(features)
    delta[:, 1:-1, :] = 0.5 * (features[:, 2:, :] - features[:, :-2, :])
    delta[:, 0, :] = features[:, 1, :] - features[:, 0, :]
    delta[:, -1, :] = features[:, -1, :] - features[:, -2, :]
    return delta


def cmvn(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=1, keepdims=True)
    std = np.std(features, axis=1, keepdims=True) + 1e-6
    return (features - mean) / std


def alignment_entropy(frame_posteriors: np.ndarray) -> np.ndarray:
    probs = np.clip(frame_posteriors, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def monotonic_attention(encoder_states: np.ndarray, decoder_queries: np.ndarray) -> np.ndarray:
    scores = np.einsum("bth,bsh->bts", decoder_queries, encoder_states)
    weights = np.exp(scores - np.max(scores, axis=2, keepdims=True))
    weights = np.cumsum(weights, axis=2)
    return weights / np.sum(weights, axis=2, keepdims=True)
