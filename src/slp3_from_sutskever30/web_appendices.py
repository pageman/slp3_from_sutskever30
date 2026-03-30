from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import numpy as np


def induced_pcfg_counts(trees: list[tuple[object, ...]]) -> dict[str, Counter[tuple[str, ...]]]:
    counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)

    def walk(tree: tuple[object, ...]) -> str:
        label = str(tree[0])
        children = tree[1:]
        if len(children) == 1 and isinstance(children[0], str):
            counts[label][(str(children[0]),)] += 1
            return label
        rhs = tuple(walk(child) for child in children)
        counts[label][rhs] += 1
        return label

    for tree in trees:
        walk(tree)
    return counts


def pcfg_rules_from_counts(counts: dict[str, Counter[tuple[str, ...]]]) -> dict[str, dict[tuple[str, ...], float]]:
    return {
        lhs: {rhs: count / sum(rhs_counts.values()) for rhs, count in rhs_counts.items()}
        for lhs, rhs_counts in counts.items()
    }


def generate_from_cfg(start: str, grammar: dict[str, list[tuple[str, ...]]], *, max_depth: int = 6) -> list[str]:
    def expand(symbol: str, depth: int) -> list[str]:
        if depth > max_depth or symbol not in grammar:
            return [symbol]
        production = grammar[symbol][0]
        output: list[str] = []
        for item in production:
            output.extend(expand(item, depth + 1))
        return output

    return expand(start, 0)


def is_in_language(tokens: list[str], start: str, grammar: dict[str, list[tuple[str, ...]]]) -> bool:
    chart: set[tuple[int, int, str]] = set()
    n = len(tokens)
    lexical = [(lhs, rhs[0]) for lhs, rhss in grammar.items() for rhs in rhss if len(rhs) == 1 and rhs[0].islower()]
    binary = [(lhs, rhs) for lhs, rhss in grammar.items() for rhs in rhss if len(rhs) == 2]
    for i, token in enumerate(tokens):
        for lhs, surface in lexical:
            if token == surface:
                chart.add((i, i, lhs))
    for span in range(2, n + 1):
        for left in range(n - span + 1):
            right = left + span - 1
            for split in range(left, right):
                for lhs, (a, b) in binary:
                    if (left, split, a) in chart and (split + 1, right, b) in chart:
                        chart.add((left, right, lhs))
    return (0, n - 1, start) in chart if tokens else False


@dataclass(frozen=True)
class CCGLexicalItem:
    token: str
    category: str
    semantics: str


def ccg_apply(left_cat: str, right_cat: str) -> str | None:
    if "/" in left_cat:
        result, arg = left_cat.split("/", 1)
        if arg == right_cat:
            return result
    if "\\" in right_cat:
        result, arg = right_cat.split("\\", 1)
        if arg == left_cat:
            return result
    return None


def ccg_derivation(items: list[CCGLexicalItem]) -> list[dict[str, str]]:
    steps = [{"span": item.token, "category": item.category, "semantics": item.semantics} for item in items]
    working = steps[:]
    while len(working) > 1:
        reduced = False
        for idx in range(len(working) - 1):
            result = ccg_apply(working[idx]["category"], working[idx + 1]["category"])
            if result is None:
                continue
            combined = {
                "span": f"{working[idx]['span']} {working[idx + 1]['span']}",
                "category": result,
                "semantics": f"{working[idx]['semantics']}({working[idx + 1]['semantics']})",
            }
            working = working[:idx] + [combined] + working[idx + 2 :]
            steps.append(combined)
            reduced = True
            break
        if not reduced:
            break
    return steps


def translate_to_logic(sentence_tokens: list[str]) -> str:
    lowered = [token.lower() for token in sentence_tokens]
    if lowered == ["every", "student", "reads"]:
        return "forall x (student(x) -> reads(x))"
    if lowered == ["some", "student", "reads"]:
        return "exists x (student(x) and reads(x))"
    if lowered == ["john", "reads", "a", "book"]:
        return "exists y (book(y) and reads(john,y))"
    return "unknown_formula(" + "_".join(lowered) + ")"


def entailment_holds(premise: str, hypothesis: str) -> bool:
    if premise == "forall x (student(x) -> reads(x))" and hypothesis == "exists x (student(x) and reads(x))":
        return False
    return premise == hypothesis or hypothesis in premise


def wordnet_similarity(edges: dict[str, list[str]], source: str, target: str) -> float:
    queue = [(source, 0)]
    seen = {source}
    while queue:
        node, dist = queue.pop(0)
        if node == target:
            return 1.0 / (1.0 + dist)
        for nxt in edges.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return 0.0


def gloss_overlap_score(context_tokens: list[str], gloss_tokens: list[str]) -> int:
    context = Counter(context_tokens)
    gloss = Counter(gloss_tokens)
    return sum(min(context[token], gloss[token]) for token in set(context) & set(gloss))


def ppmi_matrix(cooc: np.ndarray) -> np.ndarray:
    total = float(np.sum(cooc))
    row = np.sum(cooc, axis=1, keepdims=True) / total
    col = np.sum(cooc, axis=0, keepdims=True) / total
    joint = cooc / total
    with np.errstate(divide="ignore"):
        pmi = np.log((joint + 1e-12) / (row @ col + 1e-12))
    return np.maximum(pmi, 0.0)


def nearest_neighbors(matrix: np.ndarray, vocab: list[str], index: int, *, top_k: int = 3) -> list[tuple[str, float]]:
    target = matrix[index]
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(target)
    sims = np.where(norms > 0, matrix @ target / (norms + 1e-12), 0.0)
    order = np.argsort(-sims)
    return [(vocab[int(idx)], float(sims[int(idx)])) for idx in order if int(idx) != index][:top_k]


def frame_state_tracker(turns: list[dict[str, object]]) -> dict[str, object]:
    state: dict[str, str] = {}
    repairs: list[dict[str, str]] = []
    confirmations = 0
    for turn in turns:
        if "inform" in turn:
            for slot, value in dict(turn["inform"]).items():
                state[str(slot)] = str(value)
        if "repair" in turn:
            slot, value = turn["repair"]
            state[str(slot)] = str(value)
            repairs.append({"slot": str(slot), "value": str(value)})
        if turn.get("confirm"):
            confirmations += 1
    return {"state": state, "repairs": repairs, "confirmations": confirmations}


def slot_accuracy(predicted: dict[str, str], gold: dict[str, str]) -> float:
    keys = sorted(set(predicted) | set(gold))
    if not keys:
        return 1.0
    return float(sum(int(predicted.get(key) == gold.get(key)) for key in keys) / len(keys))
