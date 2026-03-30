from __future__ import annotations

from typing import Iterable

import numpy as np


def bio_constrained_decode(logits: np.ndarray, labels: list[str]) -> list[list[str]]:
    sequences: list[list[str]] = []
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for seq_logits in logits:
        tags: list[str] = []
        active_type: str | None = None
        for row in seq_logits:
            order = np.argsort(-row)
            chosen = "O"
            for idx in order:
                candidate = labels[int(idx)]
                if candidate == "O":
                    chosen = "O"
                    active_type = None
                    break
                prefix, entity = candidate.split("-", 1)
                if prefix == "B":
                    chosen = candidate
                    active_type = entity
                    break
                if prefix == "I" and active_type == entity:
                    chosen = candidate
                    break
            if chosen.startswith("I-") and active_type is None:
                entity = chosen.split("-", 1)[1]
                chosen = f"B-{entity}"
                active_type = entity
            tags.append(chosen)
            if chosen == "O":
                active_type = None
            elif chosen.startswith("B-"):
                active_type = chosen.split("-", 1)[1]
        sequences.append(tags)
    return sequences


def spans_from_bio(tags: Iterable[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = -1
    entity_type: str | None = None
    tags_list = list(tags)
    for idx, tag in enumerate(tags_list + ["O"]):
        if tag == "O":
            if entity_type is not None:
                spans.append((start, idx - 1, entity_type))
                start = -1
                entity_type = None
            continue
        prefix, current_type = tag.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if entity_type is not None:
                spans.append((start, idx - 1, entity_type))
            start = idx
            entity_type = current_type
    return spans


def token_accuracy(predicted: list[list[str]], gold: list[list[str]]) -> float:
    total = 0
    correct = 0
    for pred_seq, gold_seq in zip(predicted, gold):
        for pred, gold_tag in zip(pred_seq, gold_seq):
            total += 1
            correct += int(pred == gold_tag)
    return float(correct / max(total, 1))


def span_f1(predicted: list[list[str]], gold: list[list[str]]) -> dict[str, float]:
    pred_spans = [set(spans_from_bio(seq)) for seq in predicted]
    gold_spans = [set(spans_from_bio(seq)) for seq in gold]
    tp = sum(len(pred & gold_set) for pred, gold_set in zip(pred_spans, gold_spans))
    fp = sum(len(pred - gold_set) for pred, gold_set in zip(pred_spans, gold_spans))
    fn = sum(len(gold_set - pred) for pred, gold_set in zip(pred_spans, gold_spans))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def boundary_f1(predicted: list[list[str]], gold: list[list[str]]) -> float:
    pred_boundaries = []
    gold_boundaries = []
    for pred_seq, gold_seq in zip(predicted, gold):
        pred_set = {(start, end) for start, end, _ in spans_from_bio(pred_seq)}
        gold_set = {(start, end) for start, end, _ in spans_from_bio(gold_seq)}
        pred_boundaries.append(pred_set)
        gold_boundaries.append(gold_set)
    tp = sum(len(pred & gold_set) for pred, gold_set in zip(pred_boundaries, gold_boundaries))
    fp = sum(len(pred - gold_set) for pred, gold_set in zip(pred_boundaries, gold_boundaries))
    fn = sum(len(gold_set - pred) for pred, gold_set in zip(pred_boundaries, gold_boundaries))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def segmentation_vs_label_errors(predicted: list[list[str]], gold: list[list[str]]) -> dict[str, int]:
    segmentation_errors = 0
    label_errors = 0
    for pred_seq, gold_seq in zip(predicted, gold):
        pred_spans = spans_from_bio(pred_seq)
        gold_spans = spans_from_bio(gold_seq)
        pred_bounds = {(s, e): label for s, e, label in pred_spans}
        gold_bounds = {(s, e): label for s, e, label in gold_spans}
        shared_bounds = set(pred_bounds) & set(gold_bounds)
        label_errors += sum(int(pred_bounds[b] != gold_bounds[b]) for b in shared_bounds)
        segmentation_errors += len(set(pred_bounds) ^ set(gold_bounds))
    return {"segmentation_errors": segmentation_errors, "label_errors": label_errors}


def role_constrained_decode(role_logits: np.ndarray, role_labels: list[str]) -> list[list[str]]:
    sequences: list[list[str]] = []
    for predicate_logits in role_logits:
        chosen_roles: list[str] = []
        used_core: set[str] = set()
        for row in predicate_logits:
            order = np.argsort(-row)
            chosen = "NULL"
            for idx in order:
                candidate = role_labels[int(idx)]
                if candidate.startswith("ARG") and candidate in used_core:
                    continue
                chosen = candidate
                break
            if chosen.startswith("ARG"):
                used_core.add(chosen)
            chosen_roles.append(chosen)
        sequences.append(chosen_roles)
    return sequences


def role_metrics(predicted: list[list[str]], gold: list[list[str]]) -> dict[str, float]:
    total = 0
    correct = 0
    pred_non_null = 0
    gold_non_null = 0
    true_positive = 0
    for pred_seq, gold_seq in zip(predicted, gold):
        for pred, gold_role in zip(pred_seq, gold_seq):
            total += 1
            correct += int(pred == gold_role)
            pred_non_null += int(pred != "NULL")
            gold_non_null += int(gold_role != "NULL")
            true_positive += int(pred == gold_role and gold_role != "NULL")
    precision = true_positive / max(pred_non_null, 1)
    recall = true_positive / max(gold_non_null, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": float(correct / max(total, 1)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
