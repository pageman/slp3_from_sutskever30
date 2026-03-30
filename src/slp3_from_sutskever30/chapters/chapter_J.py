from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.web_appendices import nearest_neighbors, ppmi_matrix


def build_fixture() -> dict[str, object]:
    vocab = ["dog", "cat", "bark", "meow", "pet"]
    cooc = np.asarray(
        [
            [0, 4, 3, 0, 5],
            [4, 0, 0, 3, 5],
            [3, 0, 0, 0, 1],
            [0, 3, 0, 0, 1],
            [5, 5, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    return {"vocab": vocab, "cooc": cooc}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    matrix = ppmi_matrix(fixture["cooc"])
    dog_neighbors = nearest_neighbors(matrix, fixture["vocab"], fixture["vocab"].index("dog"))
    cat_neighbors = nearest_neighbors(matrix, fixture["vocab"], fixture["vocab"].index("cat"))
    return {"ppmi": matrix, "dog_neighbors": dog_neighbors, "cat_neighbors": cat_neighbors}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    return {
        "nonzero_entries": float(np.sum(outputs["ppmi"] > 0)),
        "dog_top_neighbor_is_pet_or_cat": float(outputs["dog_neighbors"][0][0] in {"pet", "cat"}),
        "cat_top_neighbor_is_pet_or_dog": float(outputs["cat_neighbors"][0][0] in {"pet", "dog"}),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {"case": "ppmi_is_sparse_and_frequency_sensitive", "note": "Rare cooccurrences can dominate representation quality if counts are tiny."},
        {"case": "distributional_similarity_is_not_identity", "note": "Words that occur in similar contexts are not necessarily synonyms."},
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_f_web_appendices",
        "counterintuitive_insight": "PPMI matters most when isolated from downstream embeddings because it exposes which semantic structure comes from counts alone.",
        "covered_claims": ["This appendix factors PPMI out of chapter 5 into a standalone count-based representation chapter.", "Nearest-neighbor structure is visible directly in the PPMI matrix."],
        "omitted_claims": ["No dimensionality reduction step.", "No very large cooccurrence corpus."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="J",
        implementation_status="FULL",
        core_outputs={"ppmi_shape": tuple(outputs["ppmi"].shape), "dog_neighbors": outputs["dog_neighbors"], "cat_neighbors": outputs["cat_neighbors"]},
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["chapter 5 embedding materials", "repo-native PPMI appendix"]},
        lesson_objectives=["Compute PPMI from raw cooccurrence counts.", "Inspect count-based distributional similarity directly.", "Separate count geometry from downstream neural embedding objectives."],
        core_algorithms=["cooccurrence counting", "PPMI transformation", "cosine neighbor search"],
        minimal_dataset={"vocab_size": len(fixture["vocab"]), "cooccurrence_shape": tuple(fixture["cooc"].shape)},
        reference_experiments=[
            {"name": "ppmi_neighbor_probe", "metric": ["dog_top_neighbor_is_pet_or_cat", "cat_top_neighbor_is_pet_or_dog"], "expected_signal": "distributional neighborhoods reflect shared context"},
        ],
        book_vs_repo_gap="This appendix isolates PPMI cleanly, but omits larger corpora, SVD reductions, and broader intrinsic evaluation.",
    )


SPEC = {"key": "J", "title": "PPMI", "implementation_status": "FULL", "source_papers": (), "runner": run_chapter}
