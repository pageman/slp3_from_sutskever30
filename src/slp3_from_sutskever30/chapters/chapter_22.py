from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.discourse import compose_document_scores, induce_lexicon


def build_fixture() -> dict[str, object]:
    train_texts = [
        "bright hero calm trust",
        "angry fraud delay bad",
        "steady bright support",
        "furious delay fraud",
    ]
    train_labels = [1, 0, 1, 0]
    in_domain = ["bright trust", "angry delay", "not bad support"]
    out_domain = ["hero steady service", "fraud slow refund", "very calm agent"]
    domain_labels = [1, 0, 1]
    return {"train_texts": train_texts, "train_labels": train_labels, "in_domain": in_domain, "out_domain": out_domain, "domain_labels": domain_labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    lexicon = induce_lexicon(fixture["train_texts"], fixture["train_labels"])
    in_domain_scores = [compose_document_scores(text, lexicon) for text in fixture["in_domain"]]
    out_domain_scores = [compose_document_scores(text, lexicon) for text in fixture["out_domain"]]
    return {"lexicon": lexicon, "in_domain_scores": in_domain_scores, "out_domain_scores": out_domain_scores}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    def predict(rows: list[dict[str, np.ndarray]]) -> list[int]:
        return [int(score["document_score"][0] > 0.0) for score in rows]

    in_pred = predict(outputs["in_domain_scores"])
    out_pred = predict(outputs["out_domain_scores"])
    labels = fixture["domain_labels"]
    return {
        "lexicon_size": len(outputs["lexicon"]),
        "in_domain_accuracy": float(np.mean(np.asarray(in_pred) == np.asarray(labels))),
        "out_domain_accuracy": float(np.mean(np.asarray(out_pred) == np.asarray(labels))),
        "portability_gap": float(abs(np.mean(np.asarray(in_pred) == np.asarray(labels)) - np.mean(np.asarray(out_pred) == np.asarray(labels)))),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "domain_shift_changes_token_priors",
            "note": "A lexicon that works in-domain can degrade when new domain words are unseen.",
        },
        {
            "case": "composition_rules_matter",
            "example": "not bad support",
            "score": compose_document_scores("not bad support", outputs["lexicon"])["document_score"].round(4).tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_e_discourse_and_dialogue",
        "counterintuitive_insight": "The value of a lexicon is portability under domain shift, not just in-domain sentiment accuracy.",
        "covered_claims": [
            "Chapter 22 now induces a small lexicon and composes document scores across valence, arousal, dominance, and connotation.",
            "Evaluation reports in-domain and cross-domain behavior separately.",
        ],
        "omitted_claims": ["No semi-supervised lexicon induction yet.", "No phrase-structure composition yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    sample_lexicon = {key: value.round(3).tolist() for key, value in list(outputs["lexicon"].items())[:5]}
    return build_chapter_payload(
        chapter="22",
        implementation_status="FULL",
        core_outputs={
            "lexicon_preview": sample_lexicon,
            "in_domain_document_scores": [row["document_score"].round(4).tolist() for row in outputs["in_domain_scores"]],
            "out_domain_document_scores": [row["document_score"].round(4).tolist() for row in outputs["out_domain_scores"]],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Induce a small affective lexicon from labeled text.",
            "Compose document-level sentiment and connotation scores from token-level lexicon entries.",
            "Measure portability under domain shift rather than only in-domain accuracy.",
        ],
        core_algorithms=["lexicon induction", "valence-arousal-dominance-connotation scoring", "negation-aware composition", "cross-domain evaluation"],
        minimal_dataset={"train_size": len(fixture["train_texts"]), "in_domain_size": len(fixture["in_domain"]), "out_domain_size": len(fixture["out_domain"])},
        reference_experiments=[
            {"name": "in_vs_out_domain_accuracy", "metric": ["in_domain_accuracy", "out_domain_accuracy"], "expected_signal": "domain shift reduces sentiment portability"},
            {"name": "portability_gap", "metric": "portability_gap", "expected_signal": "lexicon robustness should be judged by cross-domain degradation"},
        ],
        book_vs_repo_gap="This chapter is faithful in lexicon induction and composition, but still omits larger lexicon resources, semi-supervised induction, and richer phrase-structure composition.",
    )


SPEC = {
    "key": "22",
    "title": "Lexicons for Sentiment, Affect, and Connotation",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
