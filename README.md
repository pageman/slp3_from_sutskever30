# SLP3 From Sutskever30

[![Pytest](https://github.com/pageman/slp3_from_sutskever30/actions/workflows/pytest.yml/badge.svg)](https://github.com/pageman/slp3_from_sutskever30/actions/workflows/pytest.yml)
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
- `tests/test_smoke.py` - smoke tests over all chapter runners
- `research/` - source mapping and packaging notes

## Quick Start

```bash
cd /Users/hifi/Downloads/slp3_from_sutskever30
python3 scripts/run_slp3.py --list
python3 scripts/run_slp3.py --chapter 8
python3 -m pytest
```

## Coverage

- Main chapters: `2-25`
- Appendices: `A-D`

All runners are NumPy-only and executable. They are toy reimplementations, not full-scale reproductions of the textbook or original papers.

## Citation

If you use this repository, cite the project metadata in [CITATION.cff](./CITATION.cff).
