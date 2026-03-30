from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.probabilistic_appendices import baum_welch_step, forward_backward


def build_fixture() -> dict[str, object]:
    start = np.asarray([0.6, 0.4], dtype=np.float64)
    transition = np.asarray([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    emission = np.asarray([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
    observations = np.asarray([0, 1, 2, 1], dtype=np.int64)
    return {"start": start, "transition": transition, "emission": emission, "observations": observations}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    fb = forward_backward(fixture["start"], fixture["transition"], fixture["emission"], fixture["observations"])
    em = baum_welch_step(fixture["start"], fixture["transition"], fixture["emission"], fixture["observations"])
    return {"forward_backward": fb, "em_step": em}


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    gamma = outputs["forward_backward"]["gamma"]
    return {
        "log_likelihood": float(outputs["forward_backward"]["log_likelihood"]),
        "posterior_row_sums": np.sum(gamma, axis=1).round(6).tolist(),
        "em_transition_row_sums": np.sum(outputs["em_step"]["transition"], axis=1).round(6).tolist(),
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "state_aliasing_is_real",
            "note": "Equivalent latent-state permutations can preserve likelihood, so named hidden states should not be over-interpreted.",
        },
        {
            "case": "posterior_dynamics_not_just_viterbi",
            "gamma_first_two_steps": outputs["forward_backward"]["gamma"][:2].round(4).tolist(),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_6_appendices",
        "counterintuitive_insight": "Hidden states are not inherently identifiable. The valuable object is the posterior structure and likelihood, not a human-friendly state name.",
        "covered_claims": [
            "Appendix A now includes forward-backward and one Baum-Welch update step.",
            "Posterior marginals are exposed directly instead of only best-path decoding.",
        ],
        "omitted_claims": ["No multi-sequence EM loop yet.", "No supervised tagger bridge yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="A",
        implementation_status="FULL",
        core_outputs={
            "alpha_shape": tuple(outputs["forward_backward"]["alpha"].shape),
            "beta_shape": tuple(outputs["forward_backward"]["beta"].shape),
            "gamma": outputs["forward_backward"]["gamma"].round(4).tolist(),
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "A",
    "title": "Hidden Markov Models",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
