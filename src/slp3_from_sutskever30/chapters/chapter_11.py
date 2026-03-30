from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import cross_entropy_from_probs, seeded_rng, stable_softmax


def build_fixture() -> dict[str, object]:
    queries = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float64)
    passages = np.asarray([[1.0, 0.0, 0.1], [0.0, 1.0, 0.1], [0.8, 0.2, 0.1], [0.2, 0.8, 0.1]], dtype=np.float64)
    return {"queries": queries, "passages": passages, "retrieval_targets": np.arange(4, dtype=np.int64), "generation_targets": np.asarray([0, 1, 0, 1], dtype=np.int64)}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(11)
    queries = fixture["queries"]
    passages = fixture["passages"]
    retrieval_targets = fixture["retrieval_targets"]
    generation_targets = fixture["generation_targets"]
    wq = rng.normal(scale=0.18, size=(4, 3))
    wp = rng.normal(scale=0.18, size=(4, 3))
    q = np.tanh(queries @ wq.T)
    p = np.tanh(passages @ wp.T)
    dense_scores = q @ p.T
    dense_probs = stable_softmax(dense_scores, axis=1)
    docs = np.stack([passages[[0, 1, 2]], passages[[1, 0, 2]], passages[[0, 2, 1]], passages[[1, 2, 0]]])
    p_docs = np.tanh(np.einsum("bdm,hm->bdh", docs, wp))
    retrieval_mix = stable_softmax(np.einsum("bh,bdh->bd", q, p_docs), axis=1)
    wout = rng.normal(scale=0.18, size=(3, 8))
    combined = np.concatenate([np.repeat(q[:, None, :], 3, axis=1), p_docs], axis=2)
    doc_logits = np.einsum("vh,bdh->bdv", wout, combined)
    doc_token_probs = stable_softmax(doc_logits, axis=2)
    rag_probs = np.sum(retrieval_mix[:, :, None] * doc_token_probs, axis=1)
    return {
        "dense_probs": dense_probs,
        "dense_scores": dense_scores,
        "retrieval_mix": retrieval_mix,
        "rag_probs": rag_probs,
        "retrieval_targets": retrieval_targets,
        "generation_targets": generation_targets,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    dpr_top1 = np.argmax(outputs["dense_probs"], axis=1)
    rag_top1 = np.argmax(outputs["rag_probs"], axis=1)
    return {
        "dpr_loss": cross_entropy_from_probs(outputs["dense_probs"], outputs["retrieval_targets"]),
        "rag_loss": cross_entropy_from_probs(outputs["rag_probs"], outputs["generation_targets"]),
        "retrieval_top1": float(np.mean(dpr_top1 == outputs["retrieval_targets"])),
        "rag_top1": float(np.mean(rag_top1 == outputs["generation_targets"])),
        "retrieval_entropy": float(np.mean(-np.sum(outputs["retrieval_mix"] * np.log(outputs["retrieval_mix"] + 1e-12), axis=1))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "correct_retrieval_does_not_guarantee_correct_generation",
            "note": "RAG quality depends on both retriever ranking and how the generator consumes the retrieved evidence.",
        },
        {
            "case": "retrieval_entropy_controls_rag_brittleness",
            "retrieval_mix_preview": outputs["retrieval_mix"].round(4).tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "Retrieval systems fail less often because of wrong documents than because the model uses the right documents weakly or inconsistently.",
        "covered_claims": [
            "Dense retrieval and generation mixing are separated explicitly.",
            "Retriever confidence is visible through retrieval entropy.",
            "RAG is modeled as a composition of retrieval and token mixing rather than a monolithic score.",
        ],
        "omitted_claims": ["No BM25 baseline yet.", "No document chunking pipeline.", "No multi-hop retrieval loop."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="11",
        implementation_status="ADAPTED",
        core_outputs={
            "dense_scores_shape": tuple(outputs["dense_scores"].shape),
            "retrieval_mix": outputs["retrieval_mix"].round(4).tolist(),
            "rag_probs_shape": tuple(outputs["rag_probs"].shape),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [28, 29], "derivation_lineage": ["pageman/sutskever-30-implementations", "adapted local NumPy retrieval/rag wrapper"]},
        lesson_objectives=[
            "Separate dense retrieval quality from generation quality in a RAG-style pipeline.",
            "Inspect retrieval entropy as a proxy for evidence concentration.",
            "Show why retrieval success and downstream answer success are different events.",
        ],
        core_algorithms=["dense passage retrieval", "softmax retrieval over candidate documents", "retrieval-conditioned token mixture", "top-1 retrieval/generation evaluation"],
        minimal_dataset={"query_count": int(fixture["queries"].shape[0]), "passage_count": int(fixture["passages"].shape[0]), "candidate_docs_per_query": 3},
        reference_experiments=[
            {"name": "retrieval_vs_generation", "metric": ["retrieval_top1", "rag_top1"], "expected_signal": "retrieval and answer quality diverge"},
            {"name": "retrieval_entropy", "metric": "retrieval_entropy", "expected_signal": "less concentrated retrieval is more brittle"},
        ],
        book_vs_repo_gap="This chapter is adapted and method-faithful in miniature, but still lacks lexical baselines, chunking, multi-hop retrieval, and realistic generator conditioning.",
    )


SPEC = {
    "key": "11",
    "title": "Information Retrieval and Retrieval-Augmented Generation",
    "implementation_status": "ADAPTED",
    "source_papers": (28, 29),
    "runner": run_chapter,
}
