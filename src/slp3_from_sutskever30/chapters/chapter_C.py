from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import train_ngram_language_model
from slp3_from_sutskever30.probabilistic_appendices import perplexity_from_distribution_fn, recursive_kneser_ney_distribution


def build_fixture() -> dict[str, object]:
    train = [
        "language models predict words",
        "language models predict tokens",
        "speech models predict words",
        "context helps language models",
    ]
    heldout = ["language models predict context", "speech models predict tokens"]
    return {"train": train, "heldout": heldout}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    bigram = train_ngram_language_model(fixture["train"], order=2, alpha=0.5)
    distribution = recursive_kneser_ney_distribution(fixture["train"], ["language"], order=3, discount=0.75)
    return {"bigram": bigram, "distribution": distribution}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    bigram = outputs["bigram"]

    def add_alpha(history: list[str]) -> dict[str, float]:
        padded = (["<s>"] + history)[-1:]
        counts = bigram.context_counts.get(tuple(padded), {})
        total = sum(counts.values()) + bigram.alpha * len(bigram.vocab)
        return {token: float((counts.get(token, 0) + bigram.alpha) / total) for token in bigram.vocab}

    def kn(history: list[str]) -> dict[str, float]:
        return recursive_kneser_ney_distribution(fixture["train"], history[-2:], order=3, discount=0.75)

    return {
        "heldout_perplexity": {
            "add_alpha_bigram": perplexity_from_distribution_fn(fixture["heldout"], add_alpha),
            "recursive_kneser_ney": perplexity_from_distribution_fn(fixture["heldout"], kn),
        },
        "support_size": len(outputs["distribution"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "continuation_bookkeeping_is_the_core",
            "note": "Kneser-Ney works because it tracks continuation diversity, not because it subtracts a constant discount.",
        },
        {
            "case": "formula_memory_is_not_enough",
            "top_predictions": sorted(outputs["distribution"].items(), key=lambda item: item[1], reverse=True)[:5],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_a_classical_foundations",
        "counterintuitive_insight": "Kneser-Ney is fundamentally a continuation-estimation algorithm, not a discount trick.",
        "covered_claims": [
            "Appendix C now uses recursive Kneser-Ney style continuation backoff.",
            "Perplexity is compared against a simpler additive-smoothed baseline.",
        ],
        "omitted_claims": ["No modified discount estimator yet.", "No corpus-scaling curve yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    top = sorted(outputs["distribution"].items(), key=lambda item: item[1], reverse=True)[:5]
    return build_chapter_payload(
        chapter="C",
        implementation_status="FULL",
        core_outputs={
            "top_predictions": top,
            "support_size": len(outputs["distribution"]),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Explain Kneser-Ney through continuation distributions instead of memorized formulas.",
            "Compare continuation-aware smoothing against a simpler additive baseline.",
            "Inspect support and top-prediction behavior on held-out text.",
        ],
        core_algorithms=["recursive Kneser-Ney distribution", "add-alpha bigram baseline", "held-out perplexity comparison"],
        minimal_dataset={"train_sentences": len(fixture["train"]), "heldout_sentences": len(fixture["heldout"]), "order": 3},
        reference_experiments=[
            {"name": "perplexity_baseline_comparison", "metric": "heldout_perplexity", "expected_signal": "continuation-aware smoothing changes held-out behavior"},
            {"name": "support_probe", "metric": "support_size", "expected_signal": "distribution support remains explicit and inspectable"},
        ],
        book_vs_repo_gap="This appendix is method-faithful for recursive Kneser-Ney intuition, but it still omits modified discount estimation and broader corpus studies.",
    )


SPEC = {
    "key": "C",
    "title": "Kneser-Ney Smoothing",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
