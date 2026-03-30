# Batch B: LM and Sequence Models

This batch packages the modern language-model and sequence-model family.

Included chapters:

- `7` Large Language Models
- `8` Transformers
- `9` Post-training: Instruction Tuning, Alignment, and Test-Time Compute
- `10` Masked Language Models
- `11` Information Retrieval and Retrieval-Augmented Generation
- `12` Machine Translation
- `13` RNNs and LSTMs

Batch goals:

- convert thin wrappers into native chapter contracts
- make method-faithful miniature implementations explicit
- package stable fixtures and eval packs for the modern model family

Folders:

- `fixtures/` deterministic inputs for Batch B chapters
- `eval_packs/` serialized lesson and evaluation bundles
- `notes/` batch-specific implementation notes

Generated artifacts:

- `BATCH_B_MANIFEST.json` batch-level index for fixtures and eval packs
- `fixtures/chapter_XX_fixture.json` chapter-level deterministic inputs
- `eval_packs/chapter_XX_eval_pack.json` chapter-level lesson and evaluation bundle
