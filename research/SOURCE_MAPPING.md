# Source Mapping

This package was assembled locally from the following source repos already present on disk:

- `/Users/hifi/sutskever-30-implementations`
- `/Users/hifi/sutskever-30-beyond-numpy`
- `/Users/hifi/Sutskever-Agent`

## Status Mapping

- `DIRECT`: chapter can be mapped one-to-one to an existing NumPy implementation lineage in the source repos
- `ADAPTED`: chapter is primarily sourced from the repos but required local chapter-specific adaptation
- `FULL`: chapter is implemented natively in this repo as a richer contract-bound NumPy chapter module

## Direct lineage

- Chapter `7` is marked `DIRECT` from paper `27`
- Chapter `8` is marked `DIRECT` from paper `13`

## Adapted lineage

- Chapter `11` is marked `ADAPTED` from papers `28` and `29`
- Chapter `12` is marked `ADAPTED` from paper `14`
- Chapter `13` is marked `ADAPTED` from papers `2` and `3`
- Chapter `15` is marked `ADAPTED` from paper `21`

## Repo-native full chapters

The remaining chapters were implemented natively in this package as richer NumPy chapter modules and are marked `FULL`:

- `1-6`, `9`, `10`, `14`, `16-25`, `A-K`

These are not direct textbook copies. They are executable educational analogs designed to expose each chapter’s core computational object with explicit metrics, failure modes, and `book_vs_repo_gap` notes.
