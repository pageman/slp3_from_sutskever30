from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.parsing import GrammarRule, cky_parse, unlabeled_span_f1
from slp3_from_sutskever30.web_appendices import induced_pcfg_counts, pcfg_rules_from_counts


def build_fixture() -> dict[str, object]:
    treebank = [
        ("S", ("NP", ("Det", "the"), ("N", "student")), ("VP", ("V", "likes"), ("NP", ("N", "books")))),
        ("S", ("NP", ("Det", "the"), ("N", "researcher")), ("VP", ("V", "writes"), ("NP", ("N", "papers")))),
    ]
    tokens = ["the", "student", "likes", "books"]
    gold_tree = treebank[0]
    return {"treebank": treebank, "tokens": tokens, "gold_tree": gold_tree}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    counts = induced_pcfg_counts(fixture["treebank"])
    probs = pcfg_rules_from_counts(counts)
    lexical_rules: list[GrammarRule] = []
    binary_rules: list[GrammarRule] = []
    for lhs, rhs_probs in probs.items():
        for rhs, prob in rhs_probs.items():
            score = float(prob)
            rule = GrammarRule(lhs=lhs, rhs=rhs, score=score)
            if len(rhs) == 1 and rhs[0].islower():
                lexical_rules.append(rule)
            elif len(rhs) == 2:
                binary_rules.append(rule)
    parsed = cky_parse(fixture["tokens"], lexical_rules, binary_rules)
    return {"pcfg": probs, "parsed": parsed}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    spans = unlabeled_span_f1(outputs["parsed"]["tree"], fixture["gold_tree"])
    return {"root_score": float(outputs["parsed"]["root_score"]), "ambiguity_count": int(outputs["parsed"]["ambiguity_count"]), "unlabeled_f1": spans["f1"]}


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "tiny_treebanks_give_brittle_rule_probabilities", "rule_count": sum(len(v) for v in outputs["pcfg"].values())},
        {"case": "statistics_do_not_replace_grammar_coverage", "note": "A sparse PCFG still fails on unseen lexical items and missing productions."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "Statistical parsing is not a replacement for grammar structure; it is a ranking layer over a grammar that still has to exist.",
        "covered_claims": ["This appendix separates statistical rule estimation from the base CFG chapter.", "A tiny treebank can induce a runnable PCFG and score parses."],
        "omitted_claims": ["No lexicalized parser.", "No packed forest or discriminative reranker."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="E",
        implementation_status="FULL",
        core_outputs={"pcfg_rules": {lhs: {str(rhs): round(prob, 4) for rhs, prob in rhs_probs.items()} for lhs, rhs_probs in outputs["pcfg"].items()}, "tree": outputs["parsed"]["tree"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["chapter 18 parser utilities", "repo-native PCFG estimation appendix"]},
        lesson_objectives=["Estimate a PCFG from a tiny treebank.", "Run statistical constituency parsing separately from pure CFG mechanics.", "Evaluate parses with span-level tree metrics."],
        core_algorithms=["PCFG rule counting", "rule normalization", "CKY parsing with rule scores", "span F1 evaluation"],
        minimal_dataset={"treebank_size": len(fixture["treebank"]), "parse_length": len(fixture["tokens"])},
        reference_experiments=[
            {"name": "pcfg_parse_reconstruction", "metric": "unlabeled_f1", "expected_signal": "gold-like trees can be recovered from induced rule weights"},
            {"name": "ambiguity_probe", "metric": "ambiguity_count", "expected_signal": "ambiguity remains visible even in a tiny statistical parser"},
        ],
        book_vs_repo_gap="This appendix captures the core idea of statistical constituency parsing, but omits lexicalization, discriminative reranking, and larger treebanks.",
    )


SPEC = {"key": "E", "title": "Statistical Constituency Parsing", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
