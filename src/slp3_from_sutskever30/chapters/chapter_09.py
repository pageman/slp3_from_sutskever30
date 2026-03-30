from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.common import seeded_rng, stable_softmax


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _candidate_distribution(prompt_features: np.ndarray, candidate_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    joint = np.concatenate([np.repeat(prompt_features[:, None, :], candidate_features.shape[1], axis=1), candidate_features], axis=2)
    logits = np.einsum("d,bkd->bk", weights, joint)
    return stable_softmax(logits, axis=1)


def _pairwise_margin(prompt_features: np.ndarray, chosen_features: np.ndarray, rejected_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    chosen_joint = np.concatenate([prompt_features, chosen_features], axis=1)
    rejected_joint = np.concatenate([prompt_features, rejected_features], axis=1)
    return np.sum(chosen_joint * weights, axis=1) - np.sum(rejected_joint * weights, axis=1)


def build_fixture() -> dict[str, object]:
    prompts = [
        "summarize safety policy",
        "answer math carefully",
        "follow formatting instruction",
        "refuse dangerous request",
    ]
    prompt_features = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [
            [[1.0, 0.6, 0.2], [0.6, 0.2, 1.0], [0.2, 1.0, 0.3]],
            [[0.4, 1.0, 0.2], [0.2, 0.3, 1.0], [1.0, 0.2, 0.3]],
            [[0.7, 0.4, 1.0], [1.0, 0.2, 0.4], [0.3, 1.0, 0.2]],
            [[0.2, 1.0, 0.9], [1.0, 0.2, 0.2], [0.5, 0.6, 0.4]],
        ],
        dtype=np.float64,
    )
    sft_targets = np.asarray([0, 0, 1, 0], dtype=np.int64)
    chosen_idx = np.asarray([0, 0, 1, 0], dtype=np.int64)
    rejected_idx = np.asarray([1, 2, 2, 1], dtype=np.int64)
    verifier_targets = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    return {
        "prompts": prompts,
        "prompt_features": prompt_features,
        "candidates": candidates,
        "sft_targets": sft_targets,
        "chosen_idx": chosen_idx,
        "rejected_idx": rejected_idx,
        "verifier_targets": verifier_targets,
    }


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    rng = seeded_rng(9)
    prompt_features = fixture["prompt_features"]
    candidate_features = fixture["candidates"]
    sft_targets = fixture["sft_targets"]
    chosen_features = candidate_features[np.arange(candidate_features.shape[0]), fixture["chosen_idx"]]
    rejected_features = candidate_features[np.arange(candidate_features.shape[0]), fixture["rejected_idx"]]

    joint_dim = prompt_features.shape[1] + candidate_features.shape[2]
    policy_weights = rng.normal(scale=0.15, size=(joint_dim,))
    verifier_weights = rng.normal(scale=0.15, size=(joint_dim,))
    sft_losses: list[float] = []
    dpo_losses: list[float] = []

    for _ in range(140):
        probs = _candidate_distribution(prompt_features, candidate_features, policy_weights)
        sft_loss = -np.mean(np.log(probs[np.arange(probs.shape[0]), sft_targets] + 1e-12))
        sft_losses.append(float(sft_loss))
        joint = np.concatenate([np.repeat(prompt_features[:, None, :], candidate_features.shape[1], axis=1), candidate_features], axis=2)
        chosen_joint = joint[np.arange(joint.shape[0]), sft_targets]
        expected_joint = np.sum(probs[:, :, None] * joint, axis=1)
        policy_weights -= 0.18 * np.mean((expected_joint - chosen_joint), axis=0)

    for _ in range(180):
        margins = _pairwise_margin(prompt_features, chosen_features, rejected_features, policy_weights)
        sigma = _sigmoid(margins)
        dpo_loss = -np.mean(np.log(sigma + 1e-12))
        dpo_losses.append(float(dpo_loss))
        grad = -(1.0 - sigma)[:, None] * (
            np.concatenate([prompt_features, chosen_features], axis=1) - np.concatenate([prompt_features, rejected_features], axis=1)
        )
        policy_weights -= 0.12 * np.mean(grad, axis=0)

    verifier_joint = np.concatenate([prompt_features, chosen_features], axis=1)
    verifier_scores = verifier_joint @ verifier_weights
    verifier_probs = _sigmoid(verifier_scores)
    verifier_grad = ((verifier_probs - fixture["verifier_targets"])[:, None] * verifier_joint).mean(axis=0)
    verifier_weights -= 0.25 * verifier_grad

    final_policy_probs = _candidate_distribution(prompt_features, candidate_features, policy_weights)
    all_joint = np.concatenate([np.repeat(prompt_features[:, None, :], candidate_features.shape[1], axis=1), candidate_features], axis=2)
    verifier_all = _sigmoid(np.einsum("d,bkd->bk", verifier_weights, all_joint))
    reranked_scores = 0.65 * final_policy_probs + 0.35 * verifier_all
    reranked_indices = np.argsort(-reranked_scores, axis=1)
    budget_traces = {
        "budget_1": reranked_indices[:, :1].tolist(),
        "budget_2": reranked_indices[:, :2].tolist(),
        "budget_3": reranked_indices[:, :3].tolist(),
    }
    return {
        "sft_losses": sft_losses,
        "dpo_losses": dpo_losses,
        "final_policy_probs": final_policy_probs,
        "verifier_scores": verifier_all,
        "reranked_indices": reranked_indices,
        "budget_traces": budget_traces,
        "policy_weights": policy_weights,
        "verifier_weights": verifier_weights,
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    targets = fixture["sft_targets"]
    top_policy = np.argmax(outputs["final_policy_probs"], axis=1)
    top_reranked = outputs["reranked_indices"][:, 0]
    chosen_in_top2 = [int(target in row[:2]) for target, row in zip(targets, outputs["reranked_indices"])]
    verifier_alignment = outputs["verifier_scores"][np.arange(len(targets)), targets]
    return {
        "policy_accuracy": float(np.mean(top_policy == targets)),
        "reranked_accuracy": float(np.mean(top_reranked == targets)),
        "top2_recall": float(np.mean(chosen_in_top2)),
        "loss_reduction": {
            "sft": float(outputs["sft_losses"][0] - outputs["sft_losses"][-1]),
            "dpo": float(outputs["dpo_losses"][0] - outputs["dpo_losses"][-1]),
        },
        "verifier_alignment": float(np.mean(verifier_alignment)),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    targets = fixture["sft_targets"]
    failures: list[dict[str, object]] = []
    for prompt, target, top_policy, reranked_row in zip(
        fixture["prompts"], targets, np.argmax(outputs["final_policy_probs"], axis=1), outputs["reranked_indices"]
    ):
        if int(top_policy) != int(target):
            failures.append(
                {
                    "case": "policy_misses_preferred_candidate",
                    "prompt": prompt,
                    "target_candidate": int(target),
                    "policy_top_candidate": int(top_policy),
                    "reranked_top_candidate": int(reranked_row[0]),
                }
            )
    failures.append(
        {
            "case": "reward_hacking_risk",
            "note": "Verifier-guided reranking can prefer fluent but policy-divergent candidates if the verifier overweights surface features.",
        }
    )
    return failures


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_b_lm_and_seq_models",
        "counterintuitive_insight": "Post-training is mostly a search-control chapter. The decisive system behavior often lives in candidate generation, reranking, and budget traces rather than in the policy loss alone.",
        "covered_claims": [
            "SFT and pairwise preference optimization can be modeled separately.",
            "Verifier-guided reranking changes candidate order under compute budgets.",
            "Top-1 policy quality and reranked quality are distinct metrics.",
        ],
        "omitted_claims": ["No real language decoder yet.", "No multi-turn judge model yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="9",
        implementation_status="FULL",
        core_outputs={
            "final_policy_probs": outputs["final_policy_probs"].round(4).tolist(),
            "verifier_scores": outputs["verifier_scores"].round(4).tolist(),
            "budget_traces": outputs["budget_traces"],
            "reranked_indices": outputs["reranked_indices"].tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
        lesson_objectives=[
            "Separate supervised preference fitting from pairwise preference optimization.",
            "Make reranking and compute-budget traces visible as system behavior.",
            "Show why verifier-guided search can change outcomes without changing the underlying policy loss.",
        ],
        core_algorithms=["supervised fine-tuning objective", "pairwise preference optimization", "verifier scoring", "budgeted reranking"],
        minimal_dataset={
            "prompt_count": len(fixture["prompts"]),
            "prompt_feature_dim": int(fixture["prompt_features"].shape[1]),
            "candidates_per_prompt": int(fixture["candidates"].shape[1]),
            "candidate_feature_dim": int(fixture["candidates"].shape[2]),
        },
        reference_experiments=[
            {"name": "policy_vs_reranked_accuracy", "metric": ["policy_accuracy", "reranked_accuracy"], "expected_signal": "reranking can improve top-1 without changing base policy argmax"},
            {"name": "compute_budget_trace", "metric": "top2_recall", "expected_signal": "more budget should recover chosen candidates more often"},
        ],
        book_vs_repo_gap="This chapter is faithful only in miniature: it exposes alignment and search-control mechanics, but not a real language model decoder, human feedback dataset, or judge model at scale.",
    )


SPEC = {
    "key": "9",
    "title": "Post-training: Instruction Tuning, Alignment, and Test-Time Compute",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
