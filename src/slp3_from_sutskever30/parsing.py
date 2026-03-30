from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrammarRule:
    lhs: str
    rhs: tuple[str, ...]
    score: float


def cky_parse(tokens: list[str], lexical_rules: list[GrammarRule], binary_rules: list[GrammarRule], start_symbol: str = "S") -> dict[str, object]:
    n = len(tokens)
    chart: dict[tuple[int, int, str], float] = {}
    back: dict[tuple[int, int, str], tuple[object, ...]] = {}
    alternatives: dict[tuple[int, int, str], int] = {}

    for i, token in enumerate(tokens):
        for rule in lexical_rules:
            if len(rule.rhs) == 1 and rule.rhs[0] == token:
                key = (i, i, rule.lhs)
                chart[key] = rule.score
                back[key] = (token,)
                alternatives[key] = alternatives.get(key, 0) + 1

    for span in range(2, n + 1):
        for left in range(0, n - span + 1):
            right = left + span - 1
            for split in range(left, right):
                for rule in binary_rules:
                    left_key = (left, split, rule.rhs[0])
                    right_key = (split + 1, right, rule.rhs[1])
                    if left_key not in chart or right_key not in chart:
                        continue
                    score = rule.score + chart[left_key] + chart[right_key]
                    key = (left, right, rule.lhs)
                    alternatives[key] = alternatives.get(key, 0) + 1
                    if score > chart.get(key, -np.inf):
                        chart[key] = score
                        back[key] = (split, rule.rhs[0], rule.rhs[1])

    root_key = (0, n - 1, start_symbol)
    tree = reconstruct_tree(back, root_key)
    ambiguity = sum(count - 1 for count in alternatives.values() if count > 1)
    return {"chart": chart, "backpointers": back, "tree": tree, "root_score": chart.get(root_key, -np.inf), "ambiguity_count": ambiguity}


def reconstruct_tree(back: dict[tuple[int, int, str], tuple[object, ...]], key: tuple[int, int, str]) -> tuple[object, ...]:
    payload = back.get(key)
    if payload is None:
        return (key[2],)
    if len(payload) == 1:
        return (key[2], payload[0])
    split, left_label, right_label = payload
    left_tree = reconstruct_tree(back, (key[0], int(split), left_label))
    right_tree = reconstruct_tree(back, (int(split) + 1, key[1], right_label))
    return (key[2], left_tree, right_tree)


def bracket_spans(tree: tuple[object, ...], start: int = 0) -> tuple[list[tuple[int, int, str]], int]:
    label = str(tree[0])
    if len(tree) == 2 and isinstance(tree[1], str):
        return [(start, start, label)], start + 1
    spans: list[tuple[int, int, str]] = []
    cursor = start
    for child in tree[1:]:
        child_spans, cursor = bracket_spans(child, cursor)
        spans.extend(child_spans)
    spans.append((start, cursor - 1, label))
    return spans, cursor


def unlabeled_span_f1(predicted: tuple[object, ...], gold: tuple[object, ...]) -> dict[str, float]:
    pred_spans = {(left, right) for left, right, label in bracket_spans(predicted)[0] if right > left}
    gold_spans = {(left, right) for left, right, label in bracket_spans(gold)[0] if right > left}
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def projective_arc_decode(scores: np.ndarray) -> np.ndarray:
    batch, length, _ = scores.shape
    heads = np.zeros((batch, length), dtype=np.int64)
    for b in range(batch):
        heads[b, 0] = 0
        for dep in range(1, length):
            order = np.argsort(-scores[b, :, dep])
            chosen = 0
            for head in order:
                if int(head) == dep:
                    continue
                chosen = int(head)
                break
            heads[b, dep] = chosen
    return heads


def mst_decode(scores: np.ndarray) -> np.ndarray:
    batch, length, _ = scores.shape
    heads = np.zeros((batch, length), dtype=np.int64)
    for b in range(batch):
        heads[b, 0] = 0
        for dep in range(1, length):
            row = scores[b, :, dep].copy()
            row[dep] = -np.inf
            heads[b, dep] = int(np.argmax(row))
    return heads


def attachment_scores(pred_heads: np.ndarray, gold_heads: np.ndarray, pred_labels: np.ndarray, gold_labels: np.ndarray) -> dict[str, float]:
    mask = np.ones_like(gold_heads, dtype=bool)
    mask[:, 0] = False
    uas = np.mean((pred_heads[mask] == gold_heads[mask]).astype(np.float64))
    las = np.mean(((pred_heads == gold_heads) & (pred_labels == gold_labels))[mask].astype(np.float64))
    return {"uas": float(uas), "las": float(las)}


def crossing_count(heads: np.ndarray) -> int:
    arcs = []
    for dep, head in enumerate(heads):
        if dep == 0 or head == dep:
            continue
        left, right = sorted((dep, int(head)))
        arcs.append((left, right))
    crossings = 0
    for i, (a_left, a_right) in enumerate(arcs):
        for b_left, b_right in arcs[i + 1 :]:
            if a_left < b_left < a_right < b_right or b_left < a_left < b_right < a_right:
                crossings += 1
    return crossings
