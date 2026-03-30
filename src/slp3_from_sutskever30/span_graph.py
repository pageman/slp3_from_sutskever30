from __future__ import annotations

import itertools

import numpy as np


def propose_spans(token_repr: np.ndarray, *, max_width: int = 3) -> tuple[np.ndarray, np.ndarray]:
    batch, length, hidden = token_repr.shape
    span_reprs = []
    span_bounds = []
    for batch_idx in range(batch):
        spans = []
        bounds = []
        for start in range(length):
            for end in range(start, min(length, start + max_width)):
                spans.append(np.mean(token_repr[batch_idx, start : end + 1], axis=0))
                bounds.append((start, end))
        span_reprs.append(spans)
        span_bounds.append(bounds)
    return np.asarray(span_reprs), np.asarray(span_bounds, dtype=np.int64)


def pair_features(span_repr: np.ndarray) -> np.ndarray:
    left = np.repeat(span_repr[:, :, None, :], span_repr.shape[1], axis=2)
    right = np.repeat(span_repr[:, None, :, :], span_repr.shape[1], axis=1)
    return np.concatenate([left, right], axis=3)


def schema_constrained_relations(entity_types: np.ndarray, relation_logits: np.ndarray, type_vocab: list[str], relation_vocab: list[str]) -> np.ndarray:
    constrained = relation_logits.copy()
    for batch_idx in range(entity_types.shape[0]):
        for left_idx in range(entity_types.shape[1]):
            for right_idx in range(entity_types.shape[1]):
                left_type = type_vocab[int(entity_types[batch_idx, left_idx])]
                right_type = type_vocab[int(entity_types[batch_idx, right_idx])]
                for rel_idx, relation in enumerate(relation_vocab):
                    if relation == "works_for" and not ({left_type, right_type} <= {"PER", "ORG"}):
                        constrained[batch_idx, left_idx, right_idx, rel_idx] = -1e9
                    if relation == "located_in" and not ({left_type, right_type} <= {"ORG", "LOC"}):
                        constrained[batch_idx, left_idx, right_idx, rel_idx] = -1e9
    return constrained


def greedy_clusters(pair_scores: np.ndarray, *, threshold: float = 0.0) -> list[list[int]]:
    parent = list(range(pair_scores.shape[0]))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, j in itertools.combinations(range(pair_scores.shape[0]), 2):
        if pair_scores[i, j] >= threshold:
            union(i, j)

    groups: dict[int, list[int]] = {}
    for node in range(pair_scores.shape[0]):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def retrieve_candidates(mention_repr: np.ndarray, kb_repr: np.ndarray, *, top_k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    scores = mention_repr @ kb_repr.T
    order = np.argsort(-scores, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(scores, order, axis=1)
    return order, top_scores


def late_revision(cluster_ids: np.ndarray, candidate_ids: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    revised = candidate_ids[:, 0].copy()
    for cluster in np.unique(cluster_ids):
        members = np.where(cluster_ids == cluster)[0]
        cluster_candidates = candidate_ids[members].ravel()
        cluster_scores = candidate_scores[members].ravel()
        best = int(cluster_candidates[int(np.argmax(cluster_scores))])
        revised[members] = best
    return revised


def pair_f1(predicted_clusters: list[list[int]], gold_clusters: list[list[int]]) -> dict[str, float]:
    def pairs(clusters: list[list[int]]) -> set[tuple[int, int]]:
        output: set[tuple[int, int]] = set()
        for cluster in clusters:
            for i, j in itertools.combinations(sorted(cluster), 2):
                output.add((i, j))
        return output

    pred_pairs = pairs(predicted_clusters)
    gold_pairs = pairs(gold_clusters)
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
