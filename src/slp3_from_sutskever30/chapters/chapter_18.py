from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.parsing import GrammarRule, cky_parse, unlabeled_span_f1


def build_fixture() -> dict[str, object]:
    tokens = ["the", "student", "saw", "the", "telescope"]
    lexical_rules = [
        GrammarRule("DET", ("the",), 0.9),
        GrammarRule("N", ("student",), 0.9),
        GrammarRule("N", ("telescope",), 0.8),
        GrammarRule("V", ("saw",), 0.9),
    ]
    binary_rules = [
        GrammarRule("NP", ("DET", "N"), 1.0),
        GrammarRule("VP", ("V", "NP"), 0.9),
        GrammarRule("PP", ("P", "NP"), 0.4),
        GrammarRule("S", ("NP", "VP"), 1.2),
    ]
    gold_tree = ("S", ("NP", ("DET", "the"), ("N", "student")), ("VP", ("V", "saw"), ("NP", ("DET", "the"), ("N", "telescope"))))
    return {"tokens": tokens, "lexical_rules": lexical_rules, "binary_rules": binary_rules, "gold_tree": gold_tree}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    parse = cky_parse(fixture["tokens"], fixture["lexical_rules"], fixture["binary_rules"], start_symbol="S")
    return parse


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "root_score": float(outputs["root_score"]),
        "span_metrics": unlabeled_span_f1(outputs["tree"], fixture["gold_tree"]),
        "ambiguity_count": int(outputs["ambiguity_count"]),
        "chart_entries": len(outputs["chart"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "ambiguity_matter_more_than_single_tree",
            "ambiguity_count": int(outputs["ambiguity_count"]),
            "note": "The useful signal is where multiple derivations compete, not only which top tree wins.",
        },
        {
            "case": "grammar_coverage_limits_parse_quality",
            "note": "A small grammar can produce a tree while still failing to represent real ambiguity classes and unary structure.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_4_structured_prediction_b",
        "counterintuitive_insight": "Parsing systems benefit more from accounting for ambiguity than from reporting a single best tree. Where the chart stays uncertain is often the most important output.",
        "covered_claims": [
            "Chapter 18 now includes explicit grammar rules, CKY parsing, backpointers, and tree reconstruction.",
            "Evaluation includes unlabeled span F1 and ambiguity diagnostics.",
        ],
        "omitted_claims": ["No unary-chain handling yet.", "No lexicalized PCFG induction yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="18",
        implementation_status="FULL",
        core_outputs={
            "tree": outputs["tree"],
            "chart_entries": len(outputs["chart"]),
            "root_score": float(outputs["root_score"]),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "18",
    "title": "Context-Free Grammars and Constituency Parsing",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
