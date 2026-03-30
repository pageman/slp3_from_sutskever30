from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import CCGLexicalItem, ccg_derivation


def build_fixture() -> dict[str, object]:
    items = [
        CCGLexicalItem(token="John", category="NP", semantics="john"),
        CCGLexicalItem(token="reads", category="S\\NP", semantics="reads"),
    ]
    complex_items = [
        CCGLexicalItem(token="John", category="NP", semantics="john"),
        CCGLexicalItem(token="likes", category="(S\\NP)/NP", semantics="likes"),
        CCGLexicalItem(token="books", category="NP", semantics="books"),
    ]
    return {"simple": items, "complex": complex_items}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    simple = ccg_derivation(fixture["simple"])
    complex_case = ccg_derivation(fixture["complex"])
    return {"simple": simple, "complex": complex_case}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    simple_final = outputs["simple"][-1]["category"]
    complex_final = outputs["complex"][-1]["category"]
    return {"simple_reaches_sentence": float(simple_final == "S"), "complex_reaches_sentence": float(complex_final == "S"), "derivation_steps": float(len(outputs["complex"]))}


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "category_directionality_is_fragile", "note": "Swapping slash direction changes whether composition is legal."},
        {"case": "toy_ccg_lacks_type_raising", "note": "The appendix captures application but not the richer combinatory inventory of full CCG analyses."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "The power of CCG is not fancy trees but the discipline of directional types that force composition to reveal semantic structure.",
        "covered_claims": ["Lexical categories and application-based derivations are explicit.", "Simple and transitive-verb examples both reduce to sentence categories."],
        "omitted_claims": ["No composition rules beyond application.", "No type raising or full wide-coverage lexicon."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="G",
        implementation_status="FULL",
        core_outputs={"simple_derivation": outputs["simple"], "complex_derivation": outputs["complex"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["repo-native CCG appendix"]},
        lesson_objectives=["Represent lexical items as directional categories.", "Show forward and backward application in a CCG derivation.", "Connect composition steps to semantic assembly."],
        core_algorithms=["CCG lexical typing", "forward application", "backward application", "derivation tracing"],
        minimal_dataset={"simple_item_count": len(fixture["simple"]), "complex_item_count": len(fixture["complex"])},
        reference_experiments=[
            {"name": "sentence_reduction", "metric": ["simple_reaches_sentence", "complex_reaches_sentence"], "expected_signal": "well-typed expressions reduce to S"},
        ],
        book_vs_repo_gap="This appendix captures the compositional core of CCG, but not type raising, composition variants, or wide-coverage parsing.",
    )


SPEC = {"key": "G", "title": "Combinatory Categorial Grammar", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
