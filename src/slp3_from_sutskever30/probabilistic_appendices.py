from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from slp3_from_sutskever30.classical import tokenize_words


def forward_backward(start: np.ndarray, transition: np.ndarray, emission: np.ndarray, observations: np.ndarray) -> dict[str, np.ndarray]:
    num_states = start.shape[0]
    time_steps = observations.shape[0]
    alpha = np.zeros((time_steps, num_states), dtype=np.float64)
    beta = np.zeros((time_steps, num_states), dtype=np.float64)
    scales = np.zeros((time_steps,), dtype=np.float64)

    alpha[0] = start * emission[:, observations[0]]
    scales[0] = np.sum(alpha[0]) + 1e-12
    alpha[0] /= scales[0]
    for t in range(1, time_steps):
        alpha[t] = (alpha[t - 1] @ transition) * emission[:, observations[t]]
        scales[t] = np.sum(alpha[t]) + 1e-12
        alpha[t] /= scales[t]

    beta[-1] = 1.0
    for t in range(time_steps - 2, -1, -1):
        beta[t] = transition @ (emission[:, observations[t + 1]] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= np.sum(gamma, axis=1, keepdims=True)
    log_likelihood = float(np.sum(np.log(scales + 1e-12)))
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "log_likelihood": np.asarray(log_likelihood)}


def baum_welch_step(start: np.ndarray, transition: np.ndarray, emission: np.ndarray, observations: np.ndarray) -> dict[str, np.ndarray]:
    fb = forward_backward(start, transition, emission, observations)
    gamma = fb["gamma"]
    xi = np.zeros((observations.shape[0] - 1, transition.shape[0], transition.shape[1]), dtype=np.float64)
    for t in range(observations.shape[0] - 1):
        unnorm = (
            fb["alpha"][t][:, None]
            * transition
            * emission[:, observations[t + 1]][None, :]
            * fb["beta"][t + 1][None, :]
        )
        xi[t] = unnorm / (np.sum(unnorm) + 1e-12)
    new_start = gamma[0]
    new_transition = np.sum(xi, axis=0)
    new_transition /= np.sum(new_transition, axis=1, keepdims=True)
    new_emission = np.zeros_like(emission)
    for state in range(emission.shape[0]):
        for obs in range(emission.shape[1]):
            mask = observations == obs
            new_emission[state, obs] = np.sum(gamma[mask, state])
    new_emission /= np.sum(new_emission, axis=1, keepdims=True)
    return {"start": new_start, "transition": new_transition, "emission": new_emission, "gamma": gamma}


def train_naive_bayes_variants(texts: list[str], labels: list[str], *, alpha: float = 1.0) -> dict[str, dict[str, object]]:
    vocab = sorted({token for text in texts for token in tokenize_words(text)})
    label_list = sorted(set(labels))
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    label_index = {label: idx for idx, label in enumerate(label_list)}
    multinomial = np.full((len(label_list), len(vocab)), alpha, dtype=np.float64)
    bernoulli = np.full((len(label_list), len(vocab)), alpha, dtype=np.float64)
    class_counts = np.zeros((len(label_list),), dtype=np.float64)
    for text, label in zip(texts, labels):
        class_idx = label_index[label]
        class_counts[class_idx] += 1.0
        seen = set()
        for token in tokenize_words(text):
            idx = vocab_index[token]
            multinomial[class_idx, idx] += 1.0
            seen.add(idx)
        for idx in seen:
            bernoulli[class_idx, idx] += 1.0
    multinomial /= np.sum(multinomial, axis=1, keepdims=True)
    bernoulli /= (class_counts[:, None] + 2 * alpha)
    priors = class_counts / np.sum(class_counts)
    return {
        "multinomial": {"vocab": vocab, "labels": label_list, "priors": priors, "likelihoods": multinomial},
        "bernoulli": {"vocab": vocab, "labels": label_list, "priors": priors, "likelihoods": bernoulli},
    }


def predict_naive_bayes(model: dict[str, object], text: str, *, variant: str) -> np.ndarray:
    vocab = model["vocab"]
    labels = model["labels"]
    priors = model["priors"]
    likelihoods = model["likelihoods"]
    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    counts = Counter(tokenize_words(text))
    scores = np.log(priors + 1e-12)
    if variant == "multinomial":
        for token, count in counts.items():
            idx = vocab_index.get(token)
            if idx is not None:
                scores += count * np.log(likelihoods[:, idx] + 1e-12)
    else:
        present = {vocab_index[token] for token in counts if token in vocab_index}
        for idx in range(len(vocab)):
            prob = likelihoods[:, idx]
            scores += np.log(prob + 1e-12) if idx in present else np.log(1.0 - prob + 1e-12)
    scores = np.exp(scores - np.max(scores))
    return scores / np.sum(scores)


def calibration_error(probs: np.ndarray, labels: np.ndarray, *, bins: int = 5) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidences >= left) & (confidences < right if right < 1.0 else confidences <= right)
        if not np.any(mask):
            continue
        accuracy = np.mean(predictions[mask] == labels[mask])
        confidence = np.mean(confidences[mask])
        error += float(np.mean(mask) * abs(accuracy - confidence))
    return error


def recursive_kneser_ney_distribution(sentences: list[str], context: list[str], *, order: int = 3, discount: float = 0.75) -> dict[str, float]:
    tokenized = [(["<s>"] * (order - 1)) + tokenize_words(sentence) + ["</s>"] for sentence in sentences]
    vocab = sorted({token for sent in tokenized for token in sent})

    ngram_counts: dict[int, Counter[tuple[str, ...]]] = {n: Counter() for n in range(1, order + 1)}
    predecessors: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for sent in tokenized:
        for n in range(1, order + 1):
            for idx in range(len(sent) - n + 1):
                gram = tuple(sent[idx : idx + n])
                ngram_counts[n][gram] += 1
                if n == 2:
                    predecessors[gram[-1]].add((gram[0],))

    def recurse(history: tuple[str, ...], depth: int) -> dict[str, float]:
        if depth == 1:
            total_cont = sum(len(values) for values in predecessors.values()) + 1e-12
            return {token: float(len(predecessors[token]) / total_cont) for token in vocab}
        counts = Counter({gram[-1]: count for gram, count in ngram_counts[depth].items() if gram[:-1] == history})
        context_count = sum(counts.values())
        lower = recurse(history[1:] if history else (), depth - 1)
        if context_count == 0:
            return lower
        unique_cont = len(counts)
        backoff = discount * unique_cont / context_count
        result = {}
        for token in vocab:
            discounted = max(counts[token] - discount, 0.0) / context_count
            result[token] = float(discounted + backoff * lower[token])
        norm = sum(result.values()) + 1e-12
        return {token: value / norm for token, value in result.items()}

    history = tuple((["<s>"] * (order - 1) + [token.lower() for token in context])[-(order - 1) :])
    return recurse(history, order)


def perplexity_from_distribution_fn(sentences: list[str], predict_fn) -> float:
    losses = []
    for sentence in sentences:
        tokens = tokenize_words(sentence) + ["</s>"]
        history: list[str] = []
        for token in tokens:
            dist = predict_fn(history)
            losses.append(-math.log(dist.get(token, 1e-12) + 1e-12))
            history.append(token)
    return float(math.exp(sum(losses) / max(len(losses), 1)))


def train_confusion_model(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for source, target in pairs:
        for src_char, tgt_char in zip(source, target):
            if src_char != tgt_char:
                counts[(src_char, tgt_char)] += 1
                totals[src_char] += 1
    return {key: value / max(totals[key[0]], 1) for key, value in counts.items()}


def candidate_score(candidate: str, observed: str, confusion: dict[tuple[str, str], float], lexicon: Counter[str], sentence_prob: float) -> float:
    channel = 0.0
    for cand_char, obs_char in zip(candidate, observed):
        if cand_char != obs_char:
            channel += math.log(confusion.get((cand_char, obs_char), 1e-3))
    language = math.log(lexicon[candidate] + 1.0) + math.log(sentence_prob + 1e-12)
    return language + channel
