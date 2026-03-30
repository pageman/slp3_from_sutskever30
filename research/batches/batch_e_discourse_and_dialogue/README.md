# Batch E: Discourse and Dialogue

This batch packages sentiment lexicons, coreference/entity linking, discourse coherence, and conversation structure.

Included chapters:

- `22` Lexicons for Sentiment, Affect, and Connotation
- `23` Coreference Resolution and Entity Linking
- `24` Discourse Coherence
- `25` Conversation and its Structure

Batch goals:

- make discourse and dialogue chapters populate the rich contract natively
- package deterministic fixtures and evaluation bundles
- keep portability, coherence margins, and commitment consistency explicit

Folders:

- `fixtures/` deterministic inputs for Batch E chapters
- `eval_packs/` serialized lesson and evaluation bundles
- `notes/` batch-specific implementation notes

Generated artifacts:

- `BATCH_E_MANIFEST.json` batch-level index for fixtures and eval packs
- `fixtures/chapter_XX_fixture.json` chapter-level deterministic inputs
- `eval_packs/chapter_XX_eval_pack.json` chapter-level lesson and evaluation bundle
