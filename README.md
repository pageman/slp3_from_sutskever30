# SLP3 From Sutskever30

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
- `tests/test_smoke.py` - contributor-facing local regression tests
- `research/` - source mapping and packaging notes

## Quick Start

```bash
cd /Users/hifi/Downloads/slp3_from_sutskever30
python3 scripts/run_slp3.py --list
python3 scripts/run_slp3.py --chapter 8
python3 scripts/smoke_test.py
python3 scripts/generate_verification_status.py --run-checks
```

## Coverage

- Main chapters: `2-25`
- Appendices: `A-D`
- Status classes: `FULL`, `DIRECT`, `ADAPTED`, `SCAFFOLDED`
- Current split: `10 FULL`, `2 DIRECT`, `3 ADAPTED`, `13 SCAFFOLDED`

All runners are NumPy-only and executable. Batch 1 upgraded chapters `2-6`, batch 2 upgraded chapters `9-10`, and batch 3 upgraded chapters `14-16`, to fuller chapter contracts with evaluation and failure-case reporting. The remaining `SCAFFOLDED` chapters are not yet full implementations.

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

## Citation

If you use this repository, cite the project metadata in [CITATION.cff](./CITATION.cff).
