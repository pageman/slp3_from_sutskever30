# SLP3 From Sutskever30

[![CircleCI](https://img.shields.io/circleci/build/gh/pageman/slp3_from_sutskever30/main?logo=circleci&label=CircleCI)](https://app.circleci.com/pipelines/github/pageman/slp3_from_sutskever30)
[![Smoke Tested](https://img.shields.io/badge/Smoke-Tested%20Locally-brightgreen.svg)](./research/SMOKE_TEST_STATUS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](./pyproject.toml)
[![NumPy Only](https://img.shields.io/badge/Backend-NumPy%20Only-orange.svg)](./src/slp3_from_sutskever30/numpy_chapters.py)

This GitHub repository is a NumPy-only toy reimplementation set for the Stanford *Speech and Language Processing* third-edition draft:

- https://web.stanford.edu/~jurafsky/slp3/

It is assembled as a standalone package and derived from local work based on:

- `pageman/sutskever-30-implementations`
- `sutskever-30-beyond-numpy`

This deliverable is intentionally small-scale and pedagogical, but batch-by-batch upgrades are turning selected chapters into fuller chapter modules:

- every chapter runner is self-contained and NumPy-only
- every SLP3 chapter and appendix has a physical per-chapter module entrypoint
- chapters that align with the Sutskever 30 papers were rewritten into standalone chapter demos
- chapters without a direct paper match were implemented as compact NumPy toy analogs

## Layout

- `src/slp3_from_sutskever30/` - package code
- `src/slp3_from_sutskever30/chapters/` - one module per SLP3 chapter and appendix
- `scripts/run_slp3.py` - CLI to list and run chapters
- `scripts/smoke_test.py` - repository-local smoke runner over all chapter entries
- `scripts/generate_verification_status.py` - observability artifact generator
- `scripts/generate_circleci_artifacts.py` - CircleCI run metadata artifact generator
- `scripts/generate_deliverable_manifest.py` - machine-readable deliverable manifest generator
- `scripts/generate_batch_a_artifacts.py` - Batch A fixture and eval-pack generator
- `scripts/generate_batch_b_artifacts.py` - Batch B fixture and eval-pack generator
- `scripts/generate_batch_c_artifacts.py` - Batch C fixture and eval-pack generator
- `scripts/generate_batch_d_artifacts.py` - Batch D fixture and eval-pack generator
- `scripts/generate_batch_e_artifacts.py` - Batch E fixture and eval-pack generator
- `tests/test_smoke.py` - contributor-facing local regression tests
- `research/` - source mapping and packaging notes

## Quick Start

```bash
cd /Users/hifi/Downloads/slp3_from_sutskever30
python3 scripts/run_slp3.py --list
python3 scripts/run_slp3.py --chapter 8
python3 scripts/smoke_test.py
python3 scripts/generate_verification_status.py --run-checks
python3 scripts/generate_circleci_artifacts.py
python3 scripts/generate_deliverable_manifest.py
python3 scripts/generate_batch_a_artifacts.py
python3 scripts/generate_batch_b_artifacts.py
python3 scripts/generate_batch_c_artifacts.py
python3 scripts/generate_batch_d_artifacts.py
python3 scripts/generate_batch_e_artifacts.py
```

## Coverage

- Main chapters: `2-25`
- Appendices: `A-D`
- Status classes: `FULL`, `DIRECT`, `ADAPTED`, `SCAFFOLDED`
- Current split: `23 FULL`, `2 DIRECT`, `3 ADAPTED`, `0 SCAFFOLDED`

All runners are NumPy-only and executable. Batch 1 upgraded chapters `2-6`, batch 2 upgraded chapters `9-10`, batch 3 upgraded chapters `14-16`, the first structured-prediction sub-batch upgraded chapters `17` and `21`, the parsing sub-batch upgraded chapters `18` and `19`, the span-graph sub-batch upgraded chapters `20` and `23`, the discourse/dialogue batch upgraded chapters `22`, `24`, and `25`, and the appendix batch upgraded `A-D`, to fuller chapter contracts with evaluation and failure-case reporting. There are now no scaffolded SLP3 entries left in the registry.

See [research/CHAPTER_SURVEY.md](./research/CHAPTER_SURVEY.md) for the per-chapter survey and orphan audit.
See [research/FULL_IMPLEMENTATION_MARCH2026.md](./research/FULL_IMPLEMENTATION_MARCH2026.md) for the March 2026 full-implementation blueprint for all `SCAFFOLDED` chapters.
See [research/IMPLEMENTATION_BACKLOG_MARCH2026.md](./research/IMPLEMENTATION_BACKLOG_MARCH2026.md) for the concrete upgrade backlog and the GitHub Actions failure diagnosis.
See [research/SMOKE_TEST_STATUS.md](./research/SMOKE_TEST_STATUS.md) for the billing-independent smoke-test status.
See [research/STATUS_VOCABULARY.md](./research/STATUS_VOCABULARY.md) for the observability vocabulary used by the repo telemetry.

Committed observability artifacts:

- `observability/smoke_test.json`
- `observability/smoke_test.sqlite`
- `observability/verification.json`
- `observability/verification.yaml`
- `observability/verification.sqlite`
- `observability/circleci_run.json`
- `observability/circleci_run.sqlite`

The CircleCI-specific artifacts include the derived CircleCI workflow URL. A compact CI metadata snapshot is also mirrored into `observability/verification.json` and `observability/verification.yaml`.

Committed observability artifacts are local snapshot baselines. CircleCI job artifacts are the per-run source of truth for live CI metadata like `build_url` and `workflow_url`.

Local regeneration now writes to `observability/local/`, while CircleCI writes to `observability/ci_latest/`.

Batch packaging now covers `research/batches/batch_a_classical_foundations/`, `research/batches/batch_b_lm_and_seq_models/`, `research/batches/batch_c_speech/`, `research/batches/batch_d_structure_and_ie/`, and `research/batches/batch_e_discourse_and_dialogue/`. Each generated batch pack includes a batch manifest, per-chapter fixtures, and per-chapter eval packs.

## Citation

If you use this repository, cite the project metadata in [CITATION.cff](./CITATION.cff).
