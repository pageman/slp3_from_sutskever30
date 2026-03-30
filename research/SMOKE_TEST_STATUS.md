# Smoke Test Status

This repository now treats the smoke test as the first health signal, independent of GitHub Actions billing state.

Canonical command:

```bash
cd /Users/hifi/Downloads/slp3_from_sutskever30
python3 scripts/smoke_test.py
```

Expected behavior:

- every chapter in `2-25` and `A-D` executes
- no orphaned chapters are reported
- no unexpected chapter entries are reported

Latest local result:

- status: `PASS`
- chapter count: `28`
- orphaned chapters: `[]`
- unexpected chapters: `[]`

Notes:

- this is a repository-local execution signal
- it does not depend on GitHub Actions startup or billing state
- `pytest` can still be used locally by contributors, but smoke status is the primary front-page signal
