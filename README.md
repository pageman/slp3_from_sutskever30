# SLP3 From Sutskever30

[![Smoke Tested](https://img.shields.io/badge/Smoke-Tested%20Locally-brightgreen.svg)](./research/SMOKE_TEST_STATUS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](./pyproject.toml)
[![NumPy Only](https://img.shields.io/badge/Backend-NumPy%20Only-orange.svg)](./src/slp3_from_sutskever30/numpy_chapters.py)

This GitHub repository is a full NumPy re-implementation of the Stanford *Speech and Language Processing* third-edition draft:

- https://web.stanford.edu/~jurafsky/slp3/

It is assembled as a standalone package and derived from local work based on:

- `pageman/sutskever-30-implementations`
- `sutskever-30-beyond-numpy`

This deliverable is intentionally small-scale and pedagogical:

- every chapter runner is self-contained and NumPy-only
- chapters that align with the Sutskever 30 papers were rewritten into standalone chapter demos
- chapters without a direct paper match were implemented as compact NumPy toy analogs

## Layout

- `src/slp3_from_sutskever30/` - package code
- `scripts/run_slp3.py` - CLI to list and run chapters
- `scripts/smoke_test.py` - repository-local smoke runner over all chapter entries
- `tests/test_smoke.py` - contributor-facing local regression tests
- `research/` - source mapping and packaging notes

## Quick Start

```bash
cd /Users/hifi/Downloads/slp3_from_sutskever30
python3 scripts/run_slp3.py --list
python3 scripts/run_slp3.py --chapter 8
python3 scripts/smoke_test.py
```

## Coverage

- Main chapters: `2-25`
- Appendices: `A-D`
- Status classes: `DIRECT`, `ADAPTED`, `SCAFFOLDED`

All runners are NumPy-only and executable. They are toy reimplementations, not full-scale reproductions of the textbook or original papers.

See [research/CHAPTER_SURVEY.md](./research/CHAPTER_SURVEY.md) for the per-chapter survey and orphan audit.
See [research/FULL_IMPLEMENTATION_MARCH2026.md](./research/FULL_IMPLEMENTATION_MARCH2026.md) for the March 2026 full-implementation blueprint for all `SCAFFOLDED` chapters.
See [research/IMPLEMENTATION_BACKLOG_MARCH2026.md](./research/IMPLEMENTATION_BACKLOG_MARCH2026.md) for the concrete upgrade backlog and the GitHub Actions failure diagnosis.
See [research/SMOKE_TEST_STATUS.md](./research/SMOKE_TEST_STATUS.md) for the billing-independent smoke-test status.

## Citation

If you use this repository, cite the project metadata in [CITATION.cff](./CITATION.cff).
