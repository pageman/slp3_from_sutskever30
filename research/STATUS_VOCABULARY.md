# Status Vocabulary

This repository uses a small observability vocabulary inspired by the telemetry pattern in `sutskever-30-beyond-numpy`, but refactored for SLP3 chapter modules.

## Implementation Status

- `DIRECT`: chapter maps one-to-one onto an existing NumPy lineage from the source Sutskever repos
- `ADAPTED`: chapter is mostly sourced from those repos but required chapter-level recomposition or adaptation
- `SCAFFOLDED`: chapter is implemented here as a NumPy-only lite module because no clean direct upstream chapter exists

## Repo Checks

- `smoke_test`: repository-local chapter execution sweep
- `pytest_local`: contributor-facing local regression tests
- `survey`: chapter inventory and orphan audit

## Orphan Semantics

- `orphaned_chapters`: expected SLP3 keys missing from the registry
- `unexpected_chapters`: keys present in the registry that are outside the expected SLP3 set

## Artifact Intent

The observability artifacts are not claims of full scientific reproduction.

They exist to answer:

- what chapter entries exist
- what status each chapter currently has
- whether the local smoke and survey checks pass
- whether the repo has drifted away from the expected SLP3 chapter set
