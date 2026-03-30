from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng
from slp3_from_sutskever30.span_graph import greedy_clusters, late_revision, pair_f1, retrieve_candidates


def build_fixture() -> dict[str, object]:
    mention_doc_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    gold_clusters = [[0, 2], [1], [3, 5], [4]]
    gold_links = np.asarray([0, 1, 0, 2, 3, 2], dtype=np.int64)
    return {"mention_doc_ids": mention_doc_ids, "gold_clusters": gold_clusters, "gold_links": gold_links}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(23)
    mention_repr = rng.normal(scale=0.2, size=(6, 12))
    kb_repr = rng.normal(scale=0.2, size=(4, 12))
    pair_scores = mention_repr @ mention_repr.T
    np.fill_diagonal(pair_scores, -1e9)
    clusters = greedy_clusters(pair_scores, threshold=0.1)
    candidate_ids, candidate_scores = retrieve_candidates(mention_repr, kb_repr, top_k=2)
    cluster_ids = np.zeros((mention_repr.shape[0],), dtype=np.int64)
    for cluster_idx, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[mention] = cluster_idx
    revised_links = late_revision(cluster_ids, candidate_ids, candidate_scores)
    return {
        "pair_scores": pair_scores,
        "clusters": clusters,
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "revised_links": revised_links,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    coref = pair_f1(outputs["clusters"], fixture["gold_clusters"])
    initial_links = outputs["candidate_ids"][:, 0]
    revised_links = outputs["revised_links"]
    gold_links = fixture["gold_links"]
    return {
        "coref_pair_metrics": coref,
        "initial_link_accuracy": float(np.mean(initial_links == gold_links)),
        "revised_link_accuracy": float(np.mean(revised_links == gold_links)),
        "cluster_count": len(outputs["clusters"]),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "late_linking_revision_changes_entity_hypotheses",
            "initial_links": outputs["candidate_ids"][:, 0].tolist(),
            "revised_links": outputs["revised_links"].tolist(),
        },
        {
            "case": "early_clustering_can_lock_in_wrong_links",
            "clusters": outputs["clusters"],
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_4_span_graphs",
        "counterintuitive_insight": "The system should delay commitment longer than humans prefer. Linking evidence should be allowed to rewrite coreference hypotheses late.",
        "covered_claims": [
            "Chapter 23 now includes pair scoring, clustering, candidate retrieval, and late link revision.",
            "Coreference and linking are evaluated jointly instead of as separate islands.",
        ],
        "omitted_claims": ["No external memory persistence yet.", "No cross-document retrieval index yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="23",
        implementation_status="FULL",
        core_outputs={
            "pair_scores_shape": tuple(outputs["pair_scores"].shape),
            "clusters": outputs["clusters"],
            "candidate_ids": outputs["candidate_ids"].tolist(),
            "revised_links": outputs["revised_links"].tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "23",
    "title": "Coreference Resolution and Entity Linking",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
