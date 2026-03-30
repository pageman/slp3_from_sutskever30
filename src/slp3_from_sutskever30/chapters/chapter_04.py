from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import text_batch_to_bow, tokenize_words
from slp3_from_sutskever30.common import stable_softmax


def _hashed_features(texts: list[str], *, dim: int) -> np.ndarray:
    batch = np.zeros((len(texts), dim), dtype=np.float64)
    for row, text in enumerate(texts):
        for token in tokenize_words(text):
            batch[row, hash(token) % dim] += 1.0
    return batch


def _train_softmax(x: np.ndarray, y: np.ndarray, *, steps: int = 250, learning_rate: float = 0.2, l2: float = 1e-3) -> tuple[np.ndarray, np.ndarray, list[float]]:
    num_classes = int(np.max(y)) + 1
    weights = np.zeros((num_classes, x.shape[1]), dtype=np.float64)
    bias = np.zeros((num_classes,), dtype=np.float64)
    targets = np.eye(num_classes)[y]
    losses: list[float] = []
    for _ in range(steps):
        logits = x @ weights.T + bias
        probs = stable_softmax(logits, axis=1)
        loss = -np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12)) + 0.5 * l2 * float(np.sum(weights**2))
        losses.append(float(loss))
        grad = (probs - targets) / x.shape[0]
        weights -= learning_rate * (grad.T @ x + l2 * weights)
        bias -= learning_rate * grad.sum(axis=0)
    return weights, bias, losses


def _ece(probs: np.ndarray, y: np.ndarray, *, bins: int = 5) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidences >= left) & (confidences < right if right < 1.0 else confidences <= right)
        if not np.any(mask):
            continue
        accuracy = np.mean(predictions[mask] == y[mask])
        confidence = np.mean(confidences[mask])
        error += float(np.mean(mask) * abs(accuracy - confidence))
    return error


def build_fixture() -> dict[str, object]:
    train_texts = [
        "great acting and warm dialogue",
        "excellent pacing and great cast",
        "boring script and dull acting",
        "bad pacing and boring scenes",
        "excellent camera work and warm ending",
        "dull plot and bad jokes",
    ]
    train_labels = ["pos", "pos", "neg", "neg", "pos", "neg"]
    val_texts = ["great warm movie", "dull boring movie", "excellent acting", "bad script"]
    val_labels = ["pos", "neg", "pos", "neg"]
    return {"train_texts": train_texts, "train_labels": train_labels, "val_texts": val_texts, "val_labels": val_labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    vocab = tuple(sorted({token for text in fixture["train_texts"] for token in tokenize_words(text)}))
    label_list = ("neg", "pos")
    y_train = np.asarray([label_list.index(label) for label in fixture["train_labels"]], dtype=np.int64)
    y_val = np.asarray([label_list.index(label) for label in fixture["val_labels"]], dtype=np.int64)
    x_train = text_batch_to_bow(fixture["train_texts"], vocab)
    x_val = text_batch_to_bow(fixture["val_texts"], vocab)
    bow_weights, bow_bias, bow_losses = _train_softmax(x_train, y_train, l2=5e-3)
    bow_val_probs = stable_softmax(x_val @ bow_weights.T + bow_bias, axis=1)
    hashed_dim = 12
    hashed_train = _hashed_features(fixture["train_texts"], dim=hashed_dim)
    hashed_val = _hashed_features(fixture["val_texts"], dim=hashed_dim)
    hashed_weights, hashed_bias, hashed_losses = _train_softmax(hashed_train, y_train, l2=5e-3)
    hashed_val_probs = stable_softmax(hashed_val @ hashed_weights.T + hashed_bias, axis=1)
    top_pos = np.argsort(-(bow_weights[1] - bow_weights[0]))[:5]
    top_neg = np.argsort(-(bow_weights[0] - bow_weights[1]))[:5]
    return {
        "vocab": vocab,
        "label_list": label_list,
        "bow_losses": bow_losses,
        "hashed_losses": hashed_losses,
        "bow_val_probs": bow_val_probs,
        "hashed_val_probs": hashed_val_probs,
        "y_val": y_val,
        "top_features": {
            "pos": [vocab[idx] for idx in top_pos],
            "neg": [vocab[idx] for idx in top_neg],
        },
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    bow_probs = outputs["bow_val_probs"]
    hashed_probs = outputs["hashed_val_probs"]
    y_val = outputs["y_val"]
    return {
        "validation_accuracy": {
            "bag_of_words": float(np.mean(np.argmax(bow_probs, axis=1) == y_val)),
            "hashed_features": float(np.mean(np.argmax(hashed_probs, axis=1) == y_val)),
        },
        "expected_calibration_error": {
            "bag_of_words": _ece(bow_probs, y_val),
            "hashed_features": _ece(hashed_probs, y_val),
        },
        "loss_reduction": {
            "bag_of_words": float(outputs["bow_losses"][0] - outputs["bow_losses"][-1]),
            "hashed_features": float(outputs["hashed_losses"][0] - outputs["hashed_losses"][-1]),
        },
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    bow_probs = outputs["bow_val_probs"]
    texts = fixture["val_texts"]
    y_val = outputs["y_val"]
    failures = []
    for text, probs, target in zip(texts, bow_probs, y_val):
        pred = int(np.argmax(probs))
        if pred != int(target) or abs(float(np.max(probs)) - 0.5) < 0.1:
            failures.append({"text": text, "target": outputs["label_list"][int(target)], "predicted": outputs["label_list"][pred], "confidence": float(np.max(probs))})
    return failures or [{"case": "no_validation_failures", "note": "Current split is linearly easy; calibration remains the real diagnostic."}]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_1_classical_foundations",
        "counterintuitive_insight": "Calibration matters more than feature count. A linear model that knows when it is uncertain is often more useful than a richer feature map with uncalibrated confidence.",
        "covered_claims": [
            "Bag-of-words and hashed features can be compared on the same validation set.",
            "Regularized logistic regression supports attribution and calibration diagnostics.",
        ],
        "omitted_claims": ["No sparse matrix path yet.", "No L1 regularization sweep yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="4",
        implementation_status="FULL",
        core_outputs={
            "top_features": outputs["top_features"],
            "final_probabilities": {"bag_of_words": outputs["bow_val_probs"].round(4).tolist(), "hashed_features": outputs["hashed_val_probs"].round(4).tolist()},
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Train and evaluate logistic regression text classifiers on comparable feature sets.",
            "Compare bag-of-words and hashed features.",
            "Inspect calibration alongside accuracy.",
        ],
        core_algorithms=["bag-of-words features", "hashed text features", "softmax logistic regression", "expected calibration error"],
        minimal_dataset={"train_examples": len(fixture["train_texts"]), "validation_examples": len(fixture["val_texts"]), "labels": list(outputs["label_list"])},
        reference_experiments=[
            {"name": "feature_map_comparison", "metric": "validation_accuracy", "expected_signal": "simple linear models remain strong baselines"},
            {"name": "calibration_check", "metric": "expected_calibration_error", "expected_signal": "confidence quality is separable from accuracy"},
        ],
        book_vs_repo_gap="This chapter is faithful to the logistic-regression core, but it still omits sparse-matrix tooling, L1 paths, and larger feature-template studies.",
    )


SPEC = {
    "key": "4",
    "title": "Logistic Regression and Text Classification",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
