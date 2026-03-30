# Source Mapping

This package was assembled locally from the following source repos already present on disk:

- `/Users/hifi/sutskever-30-implementations`
- `/Users/hifi/sutskever-30-beyond-numpy`
- `/Users/hifi/Sutskever-Agent`

## Direct lineage

- Chapter `7` reuses the multi-token-prediction idea from paper `27`
- Chapter `8` reuses the transformer toy structure from paper `13`
- Chapter `11` reuses dense retrieval and RAG toy structure from papers `28` and `29`
- Chapter `12` reuses additive attention ideas from paper `14`
- Chapter `13` reuses simple RNN/LSTM toy structure from papers `2` and `3`
- Chapter `15` reuses the closed-form CTC toy structure from paper `21`

## New NumPy-first toy chapters

The remaining chapters were implemented directly in this package as compact NumPy analogs:

- `9`, `10`, `14`, `16-25`, `C`

These are not direct textbook copies. They are minimal runnable educational approximations that expose the core computational object of each chapter.
