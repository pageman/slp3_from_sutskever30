from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import generate_from_cfg, is_in_language


def build_fixture() -> dict[str, object]:
    grammar = {
        "S": [("NP", "VP")],
        "NP": [("Det", "N")],
        "VP": [("V", "NP")],
        "Det": [("the",), ("a",)],
        "N": [("student",), ("book",)],
        "V": [("reads",)],
    }
    accepted = ["the", "student", "reads", "a", "book"]
    rejected = ["student", "the", "reads", "book"]
    return {"grammar": grammar, "accepted": accepted, "rejected": rejected}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    grammar = fixture["grammar"]
    return {
        "generated_sentence": generate_from_cfg("S", grammar),
        "accepted_membership": is_in_language(fixture["accepted"], "S", grammar),
        "rejected_membership": is_in_language(fixture["rejected"], "S", grammar),
        "production_count": sum(len(v) for v in grammar.values()),
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "accepts_well_formed_sentence": float(outputs["accepted_membership"]),
        "rejects_bad_order": float(not outputs["rejected_membership"]),
        "generated_length": float(len(outputs["generated_sentence"])),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "cfg_generation_is_not_semantics", "example": outputs["generated_sentence"], "note": "A sentence can be grammatical and still meaningless."},
        {"case": "coverage_depends_on_lexicon", "note": "Membership fails immediately if terminals are absent even when the structure is licensed."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "A CFG is most useful when treated as a language definition device first and a parser input second.",
        "covered_claims": ["This appendix isolates CFG mechanics from the parsing-specific chapter.", "Generation and membership are both exposed."],
        "omitted_claims": ["No CNF conversion pipeline.", "No left-corner or top-down parser."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="F",
        implementation_status="FULL",
        core_outputs={"generated_sentence": outputs["generated_sentence"], "production_count": outputs["production_count"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["repo-native CFG appendix built beside chapter 18"]},
        lesson_objectives=["Separate grammar definition from statistical parsing.", "Show sentence generation and language membership in a CFG.", "Make formal grammar coverage explicit."],
        core_algorithms=["CFG production expansion", "sentence generation", "CKY-style membership testing"],
        minimal_dataset={"nonterminal_count": len(fixture["grammar"]), "accepted_length": len(fixture["accepted"])},
        reference_experiments=[
            {"name": "language_membership_pair", "metric": ["accepts_well_formed_sentence", "rejects_bad_order"], "expected_signal": "formal grammar distinguishes well-formed from malformed strings"},
        ],
        book_vs_repo_gap="This appendix isolates CFG mechanics well, but omits normal-form conversion, parser families, and richer grammar transforms.",
    )


SPEC = {"key": "F", "title": "Context-Free Grammars", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
