# Batch C: Speech

This batch packages the speech front-end, ASR, and TTS family.

Included chapters:

- `14` Phonetics and Speech Feature Extraction
- `15` Automatic Speech Recognition
- `16` Text-to-Speech

Batch goals:

- make the speech chapters populate the rich contract natively
- package deterministic speech fixtures and evaluation bundles
- keep the DSP, alignment, and timing layers separated

Folders:

- `fixtures/` deterministic inputs for Batch C chapters
- `eval_packs/` serialized lesson and evaluation bundles
- `notes/` batch-specific implementation notes

Generated artifacts:

- `BATCH_C_MANIFEST.json` batch-level index for fixtures and eval packs
- `fixtures/chapter_XX_fixture.json` chapter-level deterministic inputs
- `eval_packs/chapter_XX_eval_pack.json` chapter-level lesson and evaluation bundle
