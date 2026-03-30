# SLP3 From Sutskever30

NumPy-only toy reimplementations of the Stanford *Speech and Language Processing* third-edition draft chapters, assembled into a standalone package and derived from local work based on:

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
