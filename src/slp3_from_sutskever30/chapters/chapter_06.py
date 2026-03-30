from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import text_batch_to_bow, tokenize_words
from slp3_from_sutskever30.common import seeded_rng, stable_softmax


def _train_linear(x: np.ndarray, y: np.ndarray, *, steps: int = 220, learning_rate: float = 0.2) -> tuple[np.ndarray, np.ndarray, list[float]]:
    num_classes = int(np.max(y)) + 1
    weights = np.zeros((num_classes, x.shape[1]), dtype=np.float64)
    bias = np.zeros((num_classes,), dtype=np.float64)
    targets = np.eye(num_classes)[y]
    losses: list[float] = []
    for _ in range(steps):
        logits = x @ weights.T + bias
        probs = stable_softmax(logits, axis=1)
        losses.append(float(-np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12))))
        grad = (probs - targets) / x.shape[0]
        weights -= learning_rate * (grad.T @ x)
        bias -= learning_rate * grad.sum(axis=0)
    return weights, bias, losses


def _train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int = 10,
    steps: int = 260,
    learning_rate: float = 0.08,
    momentum: float = 0.9,
    dropout: float = 0.15,
    init_scale: float = 0.25,
    seed: int = 6,
) -> tuple[dict[str, np.ndarray], list[float], np.ndarray]:
    rng = seeded_rng(seed)
    num_classes = int(np.max(y)) + 1
    params = {
        "w1": rng.normal(scale=init_scale, size=(hidden_dim, x.shape[1])),
        "b1": np.zeros((hidden_dim,), dtype=np.float64),
        "w2": rng.normal(scale=init_scale, size=(num_classes, hidden_dim)),
        "b2": np.zeros((num_classes,), dtype=np.float64),
    }
    velocity = {name: np.zeros_like(value) for name, value in params.items()}
    targets = np.eye(num_classes)[y]
    losses: list[float] = []
    hidden_snapshot = np.zeros((x.shape[0], hidden_dim), dtype=np.float64)
    for step in range(steps):
        hidden_linear = x @ params["w1"].T + params["b1"]
        hidden = np.tanh(hidden_linear)
        if dropout > 0.0:
            keep_mask = (rng.random(hidden.shape) >= dropout).astype(np.float64) / (1.0 - dropout)
            hidden_dropped = hidden * keep_mask
        else:
            keep_mask = np.ones_like(hidden)
            hidden_dropped = hidden
        logits = hidden_dropped @ params["w2"].T + params["b2"]
        probs = stable_softmax(logits, axis=1)
        losses.append(float(-np.mean(np.log(probs[np.arange(y.shape[0]), y] + 1e-12))))
        grad_logits = (probs - targets) / x.shape[0]
        grad_w2 = grad_logits.T @ hidden_dropped
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ params["w2"]) * (1.0 - hidden**2) * keep_mask
        grad_w1 = grad_hidden.T @ x
        grad_b1 = grad_hidden.sum(axis=0)
        grads = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
        for name, grad in grads.items():
            velocity[name] = momentum * velocity[name] - learning_rate * grad
            params[name] += velocity[name]
        if step == steps - 1:
            hidden_snapshot = hidden
    return params, losses, hidden_snapshot


def _predict_mlp(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    hidden = np.tanh(x @ params["w1"].T + params["b1"])
    return stable_softmax(hidden @ params["w2"].T + params["b2"], axis=1)


def build_fixture() -> dict[str, object]:
    train_texts = [
        "great acting bright ending",
        "warm funny script",
        "boring acting dull jokes",
        "bad script slow ending",
        "excellent bright dialogue",
        "awful dull pacing",
        "funny warm chemistry",
        "slow boring drama",
    ]
    train_labels = ["pos", "pos", "neg", "neg", "pos", "neg", "pos", "neg"]
    val_texts = ["bright funny movie", "dull slow movie", "warm dialogue", "bad pacing"]
    val_labels = ["pos", "neg", "pos", "neg"]
    return {"train_texts": train_texts, "train_labels": train_labels, "val_texts": val_texts, "val_labels": val_labels}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    vocab = tuple(sorted({token for text in fixture["train_texts"] for token in tokenize_words(text)}))
    label_list = ("neg", "pos")
    x_train = text_batch_to_bow(fixture["train_texts"], vocab)
    x_val = text_batch_to_bow(fixture["val_texts"], vocab)
    y_train = np.asarray([label_list.index(label) for label in fixture["train_labels"]], dtype=np.int64)
    y_val = np.asarray([label_list.index(label) for label in fixture["val_labels"]], dtype=np.int64)
    linear_weights, linear_bias, linear_losses = _train_linear(x_train, y_train)
    linear_probs = stable_softmax(x_val @ linear_weights.T + linear_bias, axis=1)
    mlp_params, mlp_losses, hidden = _train_mlp(x_train, y_train)
    mlp_probs = _predict_mlp(mlp_params, x_val)
    bad_params, bad_losses, bad_hidden = _train_mlp(x_train, y_train, init_scale=1.5, learning_rate=0.02, dropout=0.0, seed=16)
    bad_probs = _predict_mlp(bad_params, x_val)
    return {
        "label_list": label_list,
        "linear_losses": linear_losses,
        "mlp_losses": mlp_losses,
        "bad_losses": bad_losses,
        "linear_probs": linear_probs,
        "mlp_probs": mlp_probs,
        "bad_probs": bad_probs,
        "y_val": y_val,
        "hidden": hidden,
        "bad_hidden": bad_hidden,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    y_val = outputs["y_val"]
    hidden = outputs["hidden"]
    bad_hidden = outputs["bad_hidden"]
    return {
        "validation_accuracy": {
            "linear": float(np.mean(np.argmax(outputs["linear_probs"], axis=1) == y_val)),
            "mlp": float(np.mean(np.argmax(outputs["mlp_probs"], axis=1) == y_val)),
            "bad_init_mlp": float(np.mean(np.argmax(outputs["bad_probs"], axis=1) == y_val)),
        },
        "loss_reduction": {
            "linear": float(outputs["linear_losses"][0] - outputs["linear_losses"][-1]),
            "mlp": float(outputs["mlp_losses"][0] - outputs["mlp_losses"][-1]),
            "bad_init_mlp": float(outputs["bad_losses"][0] - outputs["bad_losses"][-1]),
        },
        "representation_health": {
            "hidden_variance": float(np.mean(np.var(hidden, axis=0))),
            "bad_init_hidden_variance": float(np.mean(np.var(bad_hidden, axis=0))),
            "saturation_rate": float(np.mean(np.abs(hidden) > 0.95)),
            "bad_init_saturation_rate": float(np.mean(np.abs(bad_hidden) > 0.95)),
        },
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "oversized_initialization_saturates_hidden_units",
            "bad_init_final_loss": float(outputs["bad_losses"][-1]),
            "bad_init_confidences": np.max(outputs["bad_probs"], axis=1).round(4).tolist(),
        },
        {
            "case": "nonlinear_model_can_destroy_easy_linear_structure",
            "note": "The MLP is only useful if its hidden representation stays diverse instead of collapsing into tanh saturation.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_1_classical_foundations",
        "counterintuitive_insight": "The important diagnostic is what the network forgets. A larger nonlinear model can be worse than a linear baseline if hidden activations collapse or saturate.",
        "covered_claims": [
            "A NumPy MLP can be compared directly against a linear baseline.",
            "Momentum, dropout, and initialization scale materially affect hidden-state health.",
        ],
        "omitted_claims": ["No batch normalization path yet.", "No optimizer sweep beyond momentum SGD yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="6",
        implementation_status="FULL",
        core_outputs={
            "final_probabilities": {
                "linear": outputs["linear_probs"].round(4).tolist(),
                "mlp": outputs["mlp_probs"].round(4).tolist(),
                "bad_init_mlp": outputs["bad_probs"].round(4).tolist(),
            }
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Compare linear and nonlinear neural classifiers on the same task.",
            "Track optimization progress and hidden-state health.",
            "Show how initialization and saturation affect model behavior.",
        ],
        core_algorithms=["linear softmax classifier", "single-hidden-layer tanh MLP", "momentum SGD", "dropout regularization"],
        minimal_dataset={"train_examples": len(fixture["train_texts"]), "validation_examples": len(fixture["val_texts"]), "label_space": list(outputs["label_list"])},
        reference_experiments=[
            {"name": "linear_vs_mlp", "metric": "validation_accuracy", "expected_signal": "nonlinearity helps only when representation stays healthy"},
            {"name": "representation_health_probe", "metric": "representation_health", "expected_signal": "bad initialization increases saturation"},
        ],
        book_vs_repo_gap="This chapter is faithful for a compact feed-forward network lesson, but it still omits batch normalization, deeper networks, and broader optimizer comparisons.",
    )


SPEC = {
    "key": "6",
    "title": "Neural Networks",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
