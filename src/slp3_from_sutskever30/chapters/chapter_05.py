from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import build_ppmi_embeddings, tokenize_words
from slp3_from_sutskever30.common import seeded_rng


def _cooccurrence_pairs(sentences: list[str], *, window_size: int = 2) -> tuple[list[tuple[int, int]], tuple[str, ...]]:
    vocab = tuple(sorted({token for sentence in sentences for token in tokenize_words(sentence)}))
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    pairs: list[tuple[int, int]] = []
    for sentence in sentences:
        tokens = tokenize_words(sentence)
        for center_idx, center in enumerate(tokens):
            left = max(0, center_idx - window_size)
            right = min(len(tokens), center_idx + window_size + 1)
            for ctx_idx in range(left, right):
                if ctx_idx != center_idx:
                    pairs.append((vocab_index[center], vocab_index[tokens[ctx_idx]]))
    return pairs, vocab


def _train_sgns(sentences: list[str], *, embedding_dim: int = 8, steps: int = 200, negatives: int = 3, seed: int = 5) -> tuple[tuple[str, ...], np.ndarray]:
    pairs, vocab = _cooccurrence_pairs(sentences)
    rng = seeded_rng(seed)
    target = rng.normal(scale=0.2, size=(len(vocab), embedding_dim))
    context = rng.normal(scale=0.2, size=(len(vocab), embedding_dim))
    unigram = np.ones((len(vocab),), dtype=np.float64)
    unigram /= unigram.sum()
    lr = 0.08
    for _ in range(steps):
        for center_idx, ctx_idx in pairs:
            pos_score = float(target[center_idx] @ context[ctx_idx])
            pos_grad = 1.0 / (1.0 + np.exp(pos_score)) - 1.0
            target_vec = target[center_idx].copy()
            context_vec = context[ctx_idx].copy()
            target[center_idx] -= lr * pos_grad * context_vec
            context[ctx_idx] -= lr * pos_grad * target_vec
            for neg_idx in rng.choice(len(vocab), size=negatives, p=unigram):
                neg_score = float(target[center_idx] @ context[neg_idx])
                neg_grad = 1.0 / (1.0 + np.exp(-neg_score))
                neg_context_vec = context[neg_idx].copy()
                target[center_idx] -= lr * neg_grad * neg_context_vec
                context[neg_idx] -= lr * neg_grad * target[center_idx]
    return vocab, target


def _nearest_neighbors(vocab: tuple[str, ...], embeddings: np.ndarray, token: str, *, top_k: int = 3) -> list[str]:
    idx = vocab.index(token)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normalized = embeddings / norms
    sims = normalized @ normalized[idx]
    order = np.argsort(-sims)
    return [vocab[i] for i in order if vocab[i] != token][:top_k]


def _isotropy_score(embeddings: np.ndarray) -> float:
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return float(singular_values[-1] / (singular_values[0] + 1e-12))


def build_fixture() -> dict[str, object]:
    corpus = [
        "language models use context windows",
        "speech models use acoustic context",
        "language systems learn embeddings",
        "context windows stabilize embeddings",
        "speech systems learn acoustic structure",
    ]
    perturbed = corpus[:-1] + ["language systems learn structure"]
    probes = ["language", "context", "speech"]
    return {"corpus": corpus, "perturbed_corpus": perturbed, "probes": probes}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    vocab_ppmi, ppmi, ppmi_emb = build_ppmi_embeddings(fixture["corpus"], window_size=2, embedding_dim=4)
    vocab_sgns, sgns_emb = _train_sgns(fixture["corpus"], embedding_dim=8, steps=160)
    _, _, perturbed_ppmi = build_ppmi_embeddings(fixture["perturbed_corpus"], window_size=2, embedding_dim=4)
    return {
        "vocab_ppmi": vocab_ppmi,
        "ppmi_shape": tuple(ppmi.shape),
        "ppmi_embeddings": ppmi_emb,
        "sgns_vocab": vocab_sgns,
        "sgns_embeddings": sgns_emb,
        "perturbed_ppmi_embeddings": perturbed_ppmi,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    probes = fixture["probes"]
    neighbor_stability = {}
    for token in probes:
        if token not in outputs["vocab_ppmi"]:
            continue
        base_neighbors = _nearest_neighbors(outputs["vocab_ppmi"], outputs["ppmi_embeddings"], token)
        pert_neighbors = _nearest_neighbors(outputs["vocab_ppmi"], outputs["perturbed_ppmi_embeddings"], token)
        overlap = len(set(base_neighbors) & set(pert_neighbors)) / max(len(base_neighbors), 1)
        neighbor_stability[token] = float(overlap)
    return {
        "embedding_shapes": {"ppmi": tuple(outputs["ppmi_embeddings"].shape), "sgns": tuple(outputs["sgns_embeddings"].shape)},
        "isotropy": {
            "ppmi": _isotropy_score(outputs["ppmi_embeddings"]),
            "sgns": _isotropy_score(outputs["sgns_embeddings"]),
        },
        "neighbor_stability": neighbor_stability,
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "small_corpus_geometry_is_fragile",
            "note": "Neighborhoods shift under a one-sentence corpus perturbation.",
            "affected_tokens": fixture["probes"],
        },
        {
            "case": "distributional_similarity_is_not_ground_truth_semantics",
            "example": "speech and language may look close because they share contexts.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_1_classical_foundations",
        "counterintuitive_insight": "Embedding quality is more about neighborhood stability under small corpus shifts than about a single pretty nearest-neighbor screenshot.",
        "covered_claims": [
            "PPMI and SGNS-style embeddings can coexist in one NumPy-only chapter.",
            "Isotropy and neighbor stability are useful intrinsic diagnostics.",
        ],
        "omitted_claims": ["No analogy benchmark yet.", "No GloVe factorization path yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="5",
        implementation_status="FULL",
        core_outputs={
            "ppmi_shape": outputs["ppmi_shape"],
            "ppmi_neighbors": {token: _nearest_neighbors(outputs["vocab_ppmi"], outputs["ppmi_embeddings"], token) for token in fixture["probes"] if token in outputs["vocab_ppmi"]},
            "sgns_neighbors": {token: _nearest_neighbors(outputs["sgns_vocab"], outputs["sgns_embeddings"], token) for token in fixture["probes"] if token in outputs["sgns_vocab"]},
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Construct embeddings from cooccurrence statistics and predictive training.",
            "Compare PPMI and SGNS-style spaces.",
            "Evaluate geometry through isotropy and neighborhood stability.",
        ],
        core_algorithms=["PPMI matrix construction", "truncated SVD embeddings", "SGNS-style negative sampling", "nearest-neighbor probing"],
        minimal_dataset={"corpus_sentences": len(fixture["corpus"]), "perturbed_sentences": len(fixture["perturbed_corpus"]), "probe_tokens": fixture["probes"]},
        reference_experiments=[
            {"name": "ppmi_vs_sgns_geometry", "metric": "isotropy", "expected_signal": "spaces differ in anisotropy"},
            {"name": "perturbation_neighbor_stability", "metric": "neighbor_stability", "expected_signal": "small corpora produce fragile neighborhoods"},
        ],
        book_vs_repo_gap="This chapter captures the main embedding lesson in NumPy, but it still lacks GloVe, analogy suites, and broader corpus-scale intrinsic evaluation.",
    )


SPEC = {
    "key": "5",
    "title": "Embeddings",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
