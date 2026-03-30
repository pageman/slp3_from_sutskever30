from __future__ import annotations

from collections import Counter

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import edits1, tokenize_words, train_ngram_language_model
from slp3_from_sutskever30.probabilistic_appendices import candidate_score, train_confusion_model


def build_fixture() -> dict[str, object]:
    corpus = [
        "language models process speech",
        "language systems process text",
        "speech recognition uses language models",
        "the model predicts the right word",
    ]
    confusion_pairs = [("language", "langauge"), ("speech", "speach"), ("model", "modle"), ("right", "rigth")]
    observed = "langauge"
    sentence_context = ["the", "langauge", "model"]
    return {"corpus": corpus, "confusion_pairs": confusion_pairs, "observed": observed, "sentence_context": sentence_context}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    lexicon = Counter(token for text in fixture["corpus"] for token in tokenize_words(text))
    confusion = train_confusion_model(fixture["confusion_pairs"])
    lm = train_ngram_language_model(fixture["corpus"], order=2, alpha=0.3)
    candidates = [cand for cand in edits1(fixture["observed"]) if cand in lexicon]
    scores = []
    for candidate in candidates:
        prev = fixture["sentence_context"][0]
        counts = lm.context_counts.get((prev,), {})
        total = sum(counts.values()) + lm.alpha * len(lm.vocab)
        sentence_prob = float((counts.get(candidate, 0) + lm.alpha) / total)
        scores.append((candidate, candidate_score(candidate, fixture["observed"], confusion, lexicon, sentence_prob)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return {"candidates": scores, "confusion": confusion}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    top_candidate = outputs["candidates"][0][0] if outputs["candidates"] else fixture["observed"]
    return {
        "candidate_count": len(outputs["candidates"]),
        "top_candidate": top_candidate,
        "top_score": float(outputs["candidates"][0][1]) if outputs["candidates"] else float("-inf"),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "minimal_edit_distance_is_not_enough",
            "note": "A globally simpler sentence explanation can outrank a locally smaller edit.",
        },
        {
            "case": "real_word_errors_need_sentence_reranking",
            "example_context": fixture["sentence_context"],
            "candidates": outputs["candidates"][:5],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_6_appendices",
        "counterintuitive_insight": "Spelling correction should rank candidates by global explanation cost, not raw edit plausibility.",
        "covered_claims": [
            "Appendix D now combines a learned confusion model with language-model reranking.",
            "Candidate generation and reranking are separated explicitly.",
        ],
        "omitted_claims": ["No sentence-level beam search yet.", "No phonetic channel model yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="D",
        implementation_status="FULL",
        core_outputs={
            "candidates": outputs["candidates"][:5],
            "confusion_pairs": [[f"{src}->{dst}", round(prob, 4)] for (src, dst), prob in list(outputs["confusion"].items())[:5]],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "D",
    "title": "Spelling Correction and the Noisy Channel",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
