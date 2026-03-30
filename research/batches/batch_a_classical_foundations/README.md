# Batch A: Classical Foundations

This batch is the reference-quality family for making the repository more textbook-faithful while staying NumPy-only.

Included chapters:

- `2` Words and Tokens
- `3` N-gram Language Models
- `4` Logistic Regression and Text Classification
- `5` Embeddings
- `6` Neural Networks
- `A` Hidden Markov Models
- `B` Naive Bayes Classification
- `C` Kneser-Ney Smoothing
- `D` Spelling Correction and the Noisy Channel

Batch goals:

- define the strongest version of the shared chapter contract
- supply stable fixtures and evaluation packs
- create the reference packaging layout for later chapter families

Folders:

- `fixtures/` small deterministic inputs for Batch A chapters
- `eval_packs/` serialized chapter-family evaluation bundles
- `notes/` batch-specific implementation notes

Generated artifacts:

- `BATCH_A_MANIFEST.json` batch-level index for fixtures and eval packs
- `fixtures/chapter_XX_fixture.json` chapter-level deterministic inputs
- `eval_packs/chapter_XX_eval_pack.json` chapter-level lesson and evaluation bundle
