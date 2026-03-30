from __future__ import annotations

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import gloss_overlap_score, wordnet_similarity


def build_fixture() -> dict[str, object]:
    graph = {
        "dog.n.01": ["canine.n.02", "animal.n.01"],
        "canine.n.02": ["animal.n.01"],
        "bank.n.river": ["geological_formation.n.01"],
        "bank.n.finance": ["institution.n.01"],
    }
    senses = {
        "bank": {
            "river": ["sloping", "land", "beside", "river"],
            "finance": ["financial", "institution", "money", "deposit"],
        }
    }
    context = ["deposit", "money", "at", "the", "bank"]
    return {"graph": graph, "senses": senses, "context": context}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    similarities = {
        "dog_to_animal": wordnet_similarity(fixture["graph"], "dog.n.01", "animal.n.01"),
        "finance_to_river_bank": wordnet_similarity(fixture["graph"], "bank.n.finance", "bank.n.river"),
    }
    overlaps = {
        sense: gloss_overlap_score(fixture["context"], gloss)
        for sense, gloss in fixture["senses"]["bank"].items()
    }
    best_sense = max(overlaps, key=overlaps.get)
    return {"similarities": similarities, "gloss_overlaps": overlaps, "predicted_sense": best_sense}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "semantic_hierarchy_signal": float(outputs["similarities"]["dog_to_animal"]),
        "cross_branch_similarity": float(outputs["similarities"]["finance_to_river_bank"]),
        "wsd_correct": float(outputs["predicted_sense"] == "finance"),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "gloss_overlap_is_brittle", "note": "Changing context wording can flip WSD even when meaning stays constant."},
        {"case": "path_similarity_is_taxonomic_only", "note": "WordNet graph distance captures ontology structure, not contextual usage."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "Word-sense resources are most useful when they fail: taxonomy and context are different signals, and forcing them together reveals the real problem.",
        "covered_claims": ["This appendix separates lexical hierarchy from context-sensitive disambiguation.", "A toy WordNet graph and gloss-overlap WSD are both explicit."],
        "omitted_claims": ["No supervised WSD.", "No full synset inventory or relation types beyond tiny samples."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="I",
        implementation_status="FULL",
        core_outputs={"similarities": outputs["similarities"], "gloss_overlaps": outputs["gloss_overlaps"], "predicted_sense": outputs["predicted_sense"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["repo-native WordNet and WSD appendix"]},
        lesson_objectives=["Represent lexical hierarchy as a graph.", "Contrast taxonomy-based similarity with context-based WSD.", "Make word sense ambiguity operational."],
        core_algorithms=["path-based lexical similarity", "gloss-overlap scoring", "toy word-sense disambiguation"],
        minimal_dataset={"sense_inventory_size": len(fixture["senses"]["bank"]), "context_length": len(fixture["context"])},
        reference_experiments=[
            {"name": "hierarchy_vs_context", "metric": ["semantic_hierarchy_signal", "cross_branch_similarity"], "expected_signal": "taxonomic closeness differs from contextual ambiguity"},
            {"name": "toy_wsd_probe", "metric": "wsd_correct", "expected_signal": "context can pick the financial-bank sense"},
        ],
        book_vs_repo_gap="This appendix captures taxonomy and simple WSD in miniature, but omits large lexical resources, supervised WSD, and richer semantic relations.",
    )


SPEC = {"key": "I", "title": "Word Senses and WordNet", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
