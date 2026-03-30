from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import entailment_holds, translate_to_logic


def build_fixture() -> dict[str, object]:
    sentences = [
        ["every", "student", "reads"],
        ["some", "student", "reads"],
        ["john", "reads", "a", "book"],
    ]
    return {"sentences": sentences}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    formulas = [translate_to_logic(tokens) for tokens in fixture["sentences"]]
    return {"formulas": formulas, "entailment_probe": entailment_holds(formulas[0], formulas[1])}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {"formula_count": float(len(outputs["formulas"])), "recognizes_nonentailment": float(not outputs["entailment_probe"])}


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "surface_templates_are_not_full_semantics", "note": "This appendix only handles a tiny inventory of quantified sentence patterns."},
        {"case": "entailment_needs_model_theory", "note": "String-level logical forms still need interpretation machinery for real reasoning."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "Logical representation becomes clearer when the hard part is not syntax but the fact that apparently similar quantified sentences do not entail each other.",
        "covered_claims": ["Sentence patterns map to explicit logical forms.", "A simple entailment probe is enough to surface quantifier reasoning limits."],
        "omitted_claims": ["No lambda reduction engine.", "No scope ambiguity search."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="H",
        implementation_status="FULL",
        core_outputs={"formulas": outputs["formulas"], "entailment_probe": outputs["entailment_probe"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["repo-native logical-form appendix"]},
        lesson_objectives=["Translate simple sentence templates into logical forms.", "Expose the gap between syntactic similarity and entailment.", "Make sentence meaning explicit rather than implicit."],
        core_algorithms=["template-to-logic mapping", "quantifier-sensitive representation", "lightweight entailment check"],
        minimal_dataset={"sentence_count": len(fixture["sentences"]), "template_inventory": 3},
        reference_experiments=[
            {"name": "logic_translation_suite", "metric": "formula_count", "expected_signal": "each template receives an explicit logical form"},
            {"name": "quantifier_probe", "metric": "recognizes_nonentailment", "expected_signal": "universal-to-existential inference is not assumed by string similarity"},
        ],
        book_vs_repo_gap="This appendix captures explicit logical representation in miniature, but omits compositional lambda semantics, ambiguity, and full model-theoretic inference.",
    )


SPEC = {"key": "H", "title": "Logical Representations of Sentence Meaning", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
