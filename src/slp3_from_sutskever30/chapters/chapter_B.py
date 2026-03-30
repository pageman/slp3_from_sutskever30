from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.probabilistic_appendices import calibration_error, predict_naive_bayes, train_naive_bayes_variants


def build_fixture() -> dict[str, object]:
    train_texts = [
        "great fun movie",
        "fun bright comedy",
        "bad dull drama",
        "boring slow movie",
        "great bright hero",
        "dull bad fraud",
    ]
    train_labels = ["pos", "pos", "neg", "neg", "pos", "neg"]
    eval_texts = ["great movie", "boring drama", "fun fraud", "bright comedy"]
    eval_labels = np.asarray([1, 0, 0, 1], dtype=np.int64)
    return {"train_texts": train_texts, "train_labels": train_labels, "eval_texts": eval_texts, "eval_labels": eval_labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    models = train_naive_bayes_variants(fixture["train_texts"], fixture["train_labels"], alpha=1.0)
    multi_probs = np.stack([predict_naive_bayes(models["multinomial"], text, variant="multinomial") for text in fixture["eval_texts"]])
    bern_probs = np.stack([predict_naive_bayes(models["bernoulli"], text, variant="bernoulli") for text in fixture["eval_texts"]])
    return {"models": models, "multinomial_probs": multi_probs, "bernoulli_probs": bern_probs}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    labels = fixture["eval_labels"]
    multi_pred = np.argmax(outputs["multinomial_probs"], axis=1)
    bern_pred = np.argmax(outputs["bernoulli_probs"], axis=1)
    return {
        "multinomial_accuracy": float(np.mean(multi_pred == labels)),
        "bernoulli_accuracy": float(np.mean(bern_pred == labels)),
        "multinomial_ece": calibration_error(outputs["multinomial_probs"], labels),
        "bernoulli_ece": calibration_error(outputs["bernoulli_probs"], labels),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "correlated_features_break_independence",
            "example": "fun bright comedy",
            "note": "Naive Bayes remains useful because the error mode is interpretable even when the assumption is wrong.",
        },
        {
            "case": "variant_choice_changes_behavior",
            "multinomial_probs_first": outputs["multinomial_probs"][0].round(4).tolist(),
            "bernoulli_probs_first": outputs["bernoulli_probs"][0].round(4).tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_a_classical_foundations",
        "counterintuitive_insight": "Naive Bayes is valuable because it is usefully wrong. Its mistakes are structured and debuggable, not opaque.",
        "covered_claims": [
            "Appendix B now compares multinomial and Bernoulli Naive Bayes.",
            "Calibration is measured explicitly instead of assuming probabilistic outputs are trustworthy.",
        ],
        "omitted_claims": ["No smoothing sweep grid yet.", "No odds-ratio feature report yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="B",
        implementation_status="FULL",
        core_outputs={
            "multinomial_probs": outputs["multinomial_probs"].round(4).tolist(),
            "bernoulli_probs": outputs["bernoulli_probs"].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Compare multinomial and Bernoulli Naive Bayes on the same text data.",
            "Measure both accuracy and calibration.",
            "Show why conditional independence can still be operationally useful.",
        ],
        core_algorithms=["multinomial Naive Bayes", "Bernoulli Naive Bayes", "Laplace smoothing", "calibration error"],
        minimal_dataset={"train_examples": len(fixture["train_texts"]), "eval_examples": len(fixture["eval_texts"]), "labels": ["neg", "pos"]},
        reference_experiments=[
            {"name": "nb_variant_comparison", "metric": ["multinomial_accuracy", "bernoulli_accuracy"], "expected_signal": "variant choice changes predictions"},
            {"name": "calibration_check", "metric": ["multinomial_ece", "bernoulli_ece"], "expected_signal": "probabilities are not equally trustworthy"},
        ],
        book_vs_repo_gap="This appendix is faithful to the main Naive Bayes variants, but it still omits richer smoothing sweeps and feature-level odds diagnostics.",
    )


SPEC = {
    "key": "B",
    "title": "Naive Bayes Classification",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
