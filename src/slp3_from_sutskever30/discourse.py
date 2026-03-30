from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def induce_lexicon(texts: list[str], labels: list[int]) -> dict[str, np.ndarray]:
    token_scores: dict[str, list[float]] = defaultdict(list)
    for text, label in zip(texts, labels):
        for token in text.lower().split():
            valence = 1.0 if label > 0 else -1.0
            arousal = 0.6 if token in {"angry", "excited", "furious"} else 0.1
            dominance = 0.4 if token in {"calm", "steady", "confident"} else -0.2
            connotation = 0.7 if token in {"hero", "trust", "bright"} else -0.5 if token in {"fraud", "delay", "angry"} else 0.0
            token_scores[token].append(np.asarray([valence, arousal, dominance, connotation], dtype=np.float64))
    return {token: np.mean(values, axis=0) for token, values in token_scores.items()}


def compose_document_scores(text: str, lexicon: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    tokens = text.lower().split()
    token_vectors = []
    negation = False
    intensifier = 1.0
    for token in tokens:
        if token in {"not", "never"}:
            negation = True
            continue
        if token in {"very", "extremely"}:
            intensifier = 1.5
            continue
        vec = lexicon.get(token, np.zeros((4,), dtype=np.float64)).copy()
        if negation:
            vec[0] *= -1.0
            vec[3] *= -1.0
            negation = False
        vec *= intensifier
        intensifier = 1.0
        token_vectors.append(vec)
    stacked = np.stack(token_vectors) if token_vectors else np.zeros((1, 4), dtype=np.float64)
    return {"token_scores": stacked, "document_score": np.mean(stacked, axis=0)}


def entity_grid(sentences: list[list[str]], tracked_entities: list[str]) -> np.ndarray:
    grid = np.zeros((len(sentences), len(tracked_entities)), dtype=np.int64)
    for sent_idx, sent in enumerate(sentences):
        sentence_tokens = set(token.lower() for token in sent)
        for ent_idx, entity in enumerate(tracked_entities):
            grid[sent_idx, ent_idx] = int(entity.lower() in sentence_tokens)
    return grid


def perturb_order(sentences: list[list[str]]) -> list[list[str]]:
    if len(sentences) < 3:
        return list(reversed(sentences))
    perturbed = sentences.copy()
    perturbed[1], perturbed[2] = perturbed[2], perturbed[1]
    return perturbed


def coherence_score(sentence_repr: np.ndarray, entity_presence: np.ndarray) -> float:
    local_flow = np.mean(np.einsum("id,jd->ij", sentence_repr[:-1], sentence_repr[1:]))
    entity_transitions = np.mean(entity_presence[1:] == entity_presence[:-1])
    return float(local_flow + entity_transitions)


def dialogue_state(turns: list[dict[str, object]]) -> dict[str, object]:
    commitments: dict[str, str] = {}
    repair_count = 0
    grounding = 0
    act_counts: Counter[str] = Counter()
    speaker_turns: Counter[str] = Counter()
    for turn in turns:
        speaker = str(turn["speaker"])
        speaker_turns[speaker] += 1
        act = str(turn["act"])
        act_counts[act] += 1
        if "commit" in turn:
            key, value = turn["commit"]
            commitments[str(key)] = str(value)
        if turn.get("repair"):
            repair_count += 1
        if turn.get("grounded"):
            grounding += 1
    return {
        "commitments": commitments,
        "repair_count": repair_count,
        "grounding_rate": grounding / max(len(turns), 1),
        "act_counts": dict(act_counts),
        "speaker_turns": dict(speaker_turns),
    }


def commitment_consistency(turns: list[dict[str, object]]) -> float:
    memory: dict[str, str] = {}
    consistent = 0
    total = 0
    for turn in turns:
        if "check" in turn:
            total += 1
            key, expected = turn["check"]
            consistent += int(memory.get(str(key)) == str(expected))
        if "commit" in turn:
            key, value = turn["commit"]
            memory[str(key)] = str(value)
    return float(consistent / max(total, 1))
