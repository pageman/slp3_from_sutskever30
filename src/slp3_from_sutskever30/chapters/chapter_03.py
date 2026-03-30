from __future__ import annotations

from collections import Counter
import math

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import kneser_ney_next_distribution, tokenize_words, train_ngram_language_model


def _sentence_tokens(sentence: str) -> list[str]:
    return ["<s>"] + tokenize_words(sentence) + ["</s>"]


def _context_key(context: list[str], order: int) -> tuple[str, ...]:
    if order == 1:
        return ()
    padded = (["<s>"] * (order - 1) + context)[-(order - 1) :]
    return tuple(padded)


def _heldout_perplexity(model, sentences: list[str]) -> float:
    losses: list[float] = []
    for sentence in sentences:
        tokens = _sentence_tokens(sentence)
        for idx in range(1, len(tokens)):
            context = tokens[max(0, idx - model.order + 1) : idx]
            distribution = _predict_additive(model, context)
            losses.append(-math.log(distribution.get(tokens[idx], 1e-12) + 1e-12))
    return float(math.exp(sum(losses) / max(len(losses), 1)))


def _predict_additive(model, context: list[str]) -> dict[str, float]:
    key = _context_key(context, model.order)
    counts = model.context_counts.get(key, Counter())
    total = sum(counts.values()) + model.alpha * len(model.vocab)
    return {token: float((counts.get(token, 0) + model.alpha) / total) for token in model.vocab}


def _predict_with_backoff(models: dict[str, object], context: list[str]) -> dict[str, object]:
    for name, model in [("trigram", models["trigram"]), ("bigram", models["bigram"]), ("unigram", models["unigram"])]:
        key = _context_key(context, model.order)
        if name == "unigram" or key in model.context_counts:
            distribution = _predict_additive(model, context)
            top = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:5]
            return {"used_model": name, "context": list(key), "top_predictions": top}
    raise RuntimeError("unreachable")


def build_fixture() -> dict[str, object]:
    train = [
        "language models predict words",
        "language models predict tokens",
        "speech models predict words",
        "language technology predicts structure",
        "speech technology predicts pauses",
    ]
    heldout = ["language models predict structure", "speech models predict pauses"]
    probe_context = ["language", "models"]
    unseen_context = ["models", "fail"]
    return {"train": train, "heldout": heldout, "probe_context": probe_context, "unseen_context": unseen_context}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    train = fixture["train"]
    models = {
        "unigram": train_ngram_language_model(train, order=1, alpha=0.1),
        "bigram": train_ngram_language_model(train, order=2, alpha=0.1),
        "trigram": train_ngram_language_model(train, order=3, alpha=0.1),
    }
    probe = _predict_with_backoff(models, fixture["probe_context"])
    unseen = _predict_with_backoff(models, fixture["unseen_context"])
    kn = kneser_ney_next_distribution(train, fixture["probe_context"][-1:], discount=0.75)
    return {
        "models": models,
        "probe_trace": probe,
        "unseen_trace": unseen,
        "kneser_ney_top": sorted(kn.items(), key=lambda item: item[1], reverse=True)[:5],
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    heldout = fixture["heldout"]
    models = outputs["models"]
    return {
        "heldout_perplexity": {
            "unigram": _heldout_perplexity(models["unigram"], heldout),
            "bigram": _heldout_perplexity(models["bigram"], heldout),
            "trigram": _heldout_perplexity(models["trigram"], heldout),
        },
        "backoff_depths": {
            "probe_context": outputs["probe_trace"]["used_model"],
            "unseen_context": outputs["unseen_trace"]["used_model"],
        },
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "unseen_context_forces_backoff",
            "context": fixture["unseen_context"],
            "observed_trace": outputs["unseen_trace"],
        },
        {
            "case": "counts_do_not_capture_long_range_semantics",
            "example": "language models predict pauses",
            "limitation": "The model sees local continuation statistics only and cannot explain semantic mismatch.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_1_classical_foundations",
        "counterintuitive_insight": "Continuation diversity is often more predictive than raw count magnitude; the useful question is how many contexts survive compression.",
        "covered_claims": [
            "Unigram, bigram, and trigram models produce different held-out perplexities.",
            "Backoff traces expose where the model truly has evidence.",
            "Kneser-Ney changes the top distribution even on tiny corpora.",
        ],
        "omitted_claims": ["No streaming trainer yet.", "No full Katz backoff implementation yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="3",
        implementation_status="FULL",
        core_outputs={
            "probe_trace": outputs["probe_trace"],
            "unseen_trace": outputs["unseen_trace"],
            "kneser_ney_top": outputs["kneser_ney_top"],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "3",
    "title": "N-gram Language Models",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
