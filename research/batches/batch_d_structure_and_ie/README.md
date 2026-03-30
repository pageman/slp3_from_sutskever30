# Batch D: Structure and IE

This batch packages sequence labeling, parsing, information extraction, and semantic role labeling.

Included chapters:

- `17` Sequence Labeling for Parts of Speech and Named Entities
- `18` Context-Free Grammars and Constituency Parsing
- `19` Dependency Parsing
- `20` Information Extraction: Relations, Events, and Time
- `21` Semantic Role Labeling and Argument Structure

Batch goals:

- make structured prediction and IE chapters populate the rich contract natively
- package deterministic fixtures and evaluation bundles
- keep local scoring and structural consistency checks explicit

Folders:

- `fixtures/` deterministic inputs for Batch D chapters
- `eval_packs/` serialized lesson and evaluation bundles
- `notes/` batch-specific implementation notes

Generated artifacts:

- `BATCH_D_MANIFEST.json` batch-level index for fixtures and eval packs
- `fixtures/chapter_XX_fixture.json` chapter-level deterministic inputs
- `eval_packs/chapter_XX_eval_pack.json` chapter-level lesson and evaluation bundle
