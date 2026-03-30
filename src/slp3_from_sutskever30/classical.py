from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence

import numpy as np

from slp3_from_sutskever30.common import seeded_rng, stable_softmax


TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\w\s]")


def tokenize_words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def edit_distance(source: str, target: str) -> int:
    rows = len(source) + 1
    cols = len(target) + 1
    dp = np.zeros((rows, cols), dtype=np.int64)
    dp[:, 0] = np.arange(rows)
    dp[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[-1, -1])


@dataclass(frozen=True)
class NGramLanguageModel:
    order: int
    alpha: float
    vocab: tuple[str, ...]
    context_counts: dict[tuple[str, ...], Counter[str]]


def train_ngram_language_model(sentences: Sequence[str], *, order: int = 2, alpha: float = 0.5) -> NGramLanguageModel:
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    vocab: set[str] = {"</s>"}
    prefix = ["<s>"] * max(order - 1, 0)
    for sentence in sentences:
        tokens = prefix + tokenize_words(sentence) + ["</s>"]
        vocab.update(tokens)
        for idx in range(order - 1, len(tokens)):
            context = tuple(tokens[idx - order + 1 : idx])
            counts[context][tokens[idx]] += 1
    return NGramLanguageModel(order=order, alpha=alpha, vocab=tuple(sorted(vocab)), context_counts=dict(counts))


def predict_next_distribution(model: NGramLanguageModel, context: Sequence[str]) -> dict[str, float]:
    if model.order == 1:
        context_key: tuple[str, ...] = ()
    else:
        padded = (["<s>"] * (model.order - 1) + [token.lower() for token in context])[-(model.order - 1) :]
        context_key = tuple(padded)
    next_counts = model.context_counts.get(context_key, Counter())
    total = sum(next_counts.values()) + model.alpha * len(model.vocab)
    return {token: float((next_counts.get(token, 0) + model.alpha) / total) for token in model.vocab}


@dataclass(frozen=True)
class NaiveBayesTextModel:
    labels: tuple[str, ...]
    vocab: tuple[str, ...]
    log_priors: np.ndarray
    log_likelihoods: np.ndarray

    def predict(self, text: str) -> str:
        vocab_index = {token: idx for idx, token in enumerate(self.vocab)}
        features = np.zeros((len(self.vocab),), dtype=np.float64)
        for token in tokenize_words(text):
            idx = vocab_index.get(token)
            if idx is not None:
                features[idx] += 1.0
        scores = self.log_priors + features @ self.log_likelihoods.T
        return self.labels[int(np.argmax(scores))]


def train_naive_bayes_text(examples: Sequence[str], labels: Sequence[str], *, alpha: float = 1.0) -> NaiveBayesTextModel:
    label_list = sorted(set(labels))
    vocab = sorted({token for text in examples for token in tokenize_words(text)})
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    label_index = {label: idx for idx, label in enumerate(label_list)}
    token_counts = np.full((len(label_list), len(vocab)), alpha, dtype=np.float64)
    doc_counts = np.zeros((len(label_list),), dtype=np.float64)
    for text, label in zip(examples, labels):
        doc_counts[label_index[label]] += 1.0
        for token in tokenize_words(text):
            token_counts[label_index[label], vocab_index[token]] += 1.0
    return NaiveBayesTextModel(
        labels=tuple(label_list),
        vocab=tuple(vocab),
        log_priors=np.log(doc_counts / doc_counts.sum()),
        log_likelihoods=np.log(token_counts / token_counts.sum(axis=1, keepdims=True)),
    )


def text_batch_to_bow(texts: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    batch = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in tokenize_words(text):
            idx = vocab_index.get(token)
            if idx is not None:
                batch[row, idx] += 1.0
    return batch


@dataclass(frozen=True)
class LogisticRegressionTextModel:
    vocab: tuple[str, ...]
    labels: tuple[str, ...]
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, text: str) -> str:
        x = text_batch_to_bow([text], self.vocab)
        logits = x @ self.weights.T + self.bias
        return self.labels[int(np.argmax(logits[0]))]


def train_logistic_regression_text(
    examples: Sequence[str], labels: Sequence[str], *, steps: int = 200, learning_rate: float = 0.2
) -> tuple[LogisticRegressionTextModel, list[float]]:
    vocab = tuple(sorted({token for text in examples for token in tokenize_words(text)}))
    label_list = tuple(sorted(set(labels)))
    x = text_batch_to_bow(examples, vocab)
    y = np.asarray([label_list.index(label) for label in labels], dtype=np.int64)
    weights = np.zeros((len(label_list), len(vocab)), dtype=np.float64)
    bias = np.zeros((len(label_list),), dtype=np.float64)
    targets = np.eye(len(label_list))[y]
    losses: list[float] = []
    for _ in range(steps):
        logits = x @ weights.T + bias
        probs = stable_softmax(logits, axis=1)
        losses.append(float(-np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12))))
        grad = (probs - targets) / x.shape[0]
        weights -= learning_rate * (grad.T @ x)
        bias -= learning_rate * grad.sum(axis=0)
    return LogisticRegressionTextModel(vocab=vocab, labels=label_list, weights=weights, bias=bias), losses


def build_ppmi_embeddings(
    sentences: Sequence[str], *, window_size: int = 2, embedding_dim: int = 4
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    vocab = tuple(sorted({token for sentence in sentences for token in tokenize_words(sentence)}))
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    counts = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
    for sentence in sentences:
        tokens = tokenize_words(sentence)
        for center_idx, center in enumerate(tokens):
            left = max(0, center_idx - window_size)
            right = min(len(tokens), center_idx + window_size + 1)
            for ctx_idx in range(left, right):
                if ctx_idx != center_idx:
                    counts[vocab_index[center], vocab_index[tokens[ctx_idx]]] += 1.0
    total = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    ppmi = np.maximum(np.log((counts * total + 1e-12) / (row @ col + 1e-12)), 0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = u[:, :embedding_dim] * np.sqrt(s[:embedding_dim])[None, :]
    return vocab, ppmi, emb


@dataclass(frozen=True)
class MLPTextModel:
    vocab: tuple[str, ...]
    labels: tuple[str, ...]
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    def predict(self, text: str) -> str:
        x = text_batch_to_bow([text], self.vocab)
        hidden = np.tanh(x @ self.w1.T + self.b1)
        logits = hidden @ self.w2.T + self.b2
        return self.labels[int(np.argmax(logits[0]))]


def train_mlp_text_classifier(
    examples: Sequence[str], labels: Sequence[str], *, hidden_dim: int = 8, steps: int = 250, learning_rate: float = 0.1, seed: int = 6
) -> tuple[MLPTextModel, list[float]]:
    vocab = tuple(sorted({token for text in examples for token in tokenize_words(text)}))
    label_list = tuple(sorted(set(labels)))
    x = text_batch_to_bow(examples, vocab)
    y = np.asarray([label_list.index(label) for label in labels], dtype=np.int64)
    rng = seeded_rng(seed)
    w1 = rng.normal(scale=0.2, size=(hidden_dim, len(vocab)))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(scale=0.2, size=(len(label_list), hidden_dim))
    b2 = np.zeros((len(label_list),), dtype=np.float64)
    targets = np.eye(len(label_list))[y]
    losses: list[float] = []
    for _ in range(steps):
        hidden_linear = x @ w1.T + b1
        hidden = np.tanh(hidden_linear)
        logits = hidden @ w2.T + b2
        probs = stable_softmax(logits, axis=1)
        losses.append(float(-np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12))))
        grad_logits = (probs - targets) / x.shape[0]
        grad_hidden = (grad_logits @ w2) * (1.0 - np.tanh(hidden_linear) ** 2)
        w2 -= learning_rate * (grad_logits.T @ hidden)
        b2 -= learning_rate * grad_logits.sum(axis=0)
        w1 -= learning_rate * (grad_hidden.T @ x)
        b1 -= learning_rate * grad_hidden.sum(axis=0)
    return MLPTextModel(vocab=vocab, labels=label_list, w1=w1, b1=b1, w2=w2, b2=b2), losses


@dataclass(frozen=True)
class HiddenMarkovModel:
    states: tuple[str, ...]
    vocab: tuple[str, ...]
    start_log_probs: np.ndarray
    transition_log_probs: np.ndarray
    emission_log_probs: np.ndarray


def train_hmm_tagger(tagged_sentences: Sequence[Sequence[tuple[str, str]]], *, alpha: float = 0.5) -> HiddenMarkovModel:
    states = tuple(sorted({tag for sentence in tagged_sentences for _, tag in sentence}))
    vocab = tuple(sorted({word.lower() for sentence in tagged_sentences for word, _ in sentence}))
    state_index = {tag: idx for idx, tag in enumerate(states)}
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    start = np.full((len(states),), alpha, dtype=np.float64)
    trans = np.full((len(states), len(states)), alpha, dtype=np.float64)
    emit = np.full((len(states), len(vocab)), alpha, dtype=np.float64)
    for sentence in tagged_sentences:
        for idx, (word, tag) in enumerate(sentence):
            emit[state_index[tag], vocab_index[word.lower()]] += 1.0
            if idx == 0:
                start[state_index[tag]] += 1.0
            else:
                trans[state_index[sentence[idx - 1][1]], state_index[tag]] += 1.0
    return HiddenMarkovModel(
        states=states,
        vocab=vocab,
        start_log_probs=np.log(start / start.sum()),
        transition_log_probs=np.log(trans / trans.sum(axis=1, keepdims=True)),
        emission_log_probs=np.log(emit / emit.sum(axis=1, keepdims=True)),
    )


def viterbi_decode(model: HiddenMarkovModel, tokens: Sequence[str]) -> list[str]:
    vocab_index = {word: idx for idx, word in enumerate(model.vocab)}
    num_states = len(model.states)
    fallback = np.full((num_states,), -math.log(len(model.vocab)), dtype=np.float64)
    dp = np.full((len(tokens), num_states), -np.inf, dtype=np.float64)
    back = np.zeros((len(tokens), num_states), dtype=np.int64)
    idx0 = vocab_index.get(tokens[0].lower())
    dp[0] = model.start_log_probs + (model.emission_log_probs[:, idx0] if idx0 is not None else fallback)
    for t in range(1, len(tokens)):
        idx = vocab_index.get(tokens[t].lower())
        emission = model.emission_log_probs[:, idx] if idx is not None else fallback
        for state in range(num_states):
            scores = dp[t - 1] + model.transition_log_probs[:, state]
            back[t, state] = int(np.argmax(scores))
            dp[t, state] = float(np.max(scores) + emission[state])
    state = int(np.argmax(dp[-1]))
    path = [state]
    for t in range(len(tokens) - 1, 0, -1):
        state = int(back[t, state])
        path.append(state)
    path.reverse()
    return [model.states[idx] for idx in path]


@dataclass(frozen=True)
class NoisyChannelSpeller:
    lexicon: Counter[str]


def build_demo_speller(corpus: Iterable[str]) -> NoisyChannelSpeller:
    return NoisyChannelSpeller(Counter(token for text in corpus for token in tokenize_words(text)))


def edits1(token: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(token[:i], token[i:]) for i in range(len(token) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
    replaces = {left + letter + right[1:] for left, right in splits if right for letter in letters}
    inserts = {left + letter + right for left, right in splits for letter in letters}
    return deletes | transposes | replaces | inserts


def correct_token(token: str, speller: NoisyChannelSpeller) -> str:
    token = token.lower()
    if token in speller.lexicon:
        return token
    candidates = [candidate for candidate in edits1(token) if candidate in speller.lexicon]
    return max(candidates, key=lambda c: (speller.lexicon[c], -edit_distance(token, c), c)) if candidates else token


def kneser_ney_next_distribution(sentences: Sequence[str], context: Sequence[str], *, discount: float = 0.75) -> dict[str, float]:
    tokens = [tokenize_words(sentence) + ["</s>"] for sentence in sentences]
    vocab = sorted({token for sent in tokens for token in sent})
    bigram_counts: dict[tuple[str, str], int] = defaultdict(int)
    context_counts: Counter[str] = Counter()
    continuation_sets: dict[str, set[str]] = defaultdict(set)
    predecessor_sets: dict[str, set[str]] = defaultdict(set)
    for sent in tokens:
        prev = "<s>"
        for token in sent:
            bigram_counts[(prev, token)] += 1
            context_counts[prev] += 1
            continuation_sets[prev].add(token)
            predecessor_sets[token].add(prev)
            prev = token
    prev = context[-1].lower() if context else "<s>"
    total_continuations = sum(len(values) for values in predecessor_sets.values())
    distribution: dict[str, float] = {}
    for token in vocab:
        c_bigram = bigram_counts[(prev, token)]
        c_prev = context_counts[prev]
        continuation_prob = len(predecessor_sets[token]) / max(total_continuations, 1)
        lambda_prev = discount * len(continuation_sets[prev]) / max(c_prev, 1)
        distribution[token] = max(c_bigram - discount, 0.0) / max(c_prev, 1) + lambda_prev * continuation_prob
    total = sum(distribution.values())
    return {token: value / total for token, value in distribution.items()}
