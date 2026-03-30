# Narrative And Method Arc 2026-03-30

This document gives a single end-to-end frame for the project:

- the **narrative arc** explains what story the research tells from beginning to end
- the **methodological arc** explains how the work was actually carried out
- **story boxes** summarize what each phase means in human terms
- **method boxes** summarize the technical move, artifact, and verification logic for that phase

The goal is to make the project readable as one coherent research program rather than a pile of chapter implementations, tests, and CI steps.

---

## Overarching Narrative Arc

The project began as a question of translation:

- can the conceptual structure of Stanford SLP3 be re-expressed through NumPy-only executable chapters
- can the implementation lineage be anchored in `sutskever-30-implementations` and adjacent `beyond-numpy` work
- can the result remain honest about where it is faithful, where it is miniature, and where it is only an educational analog

The answer that emerged is not:

- "we reproduced the textbook exactly"

The answer is:

- "we built a progressively more disciplined NumPy research scaffold that can host textbook-faithful upgrades chapter by chapter"

The final story is therefore not a story about scale. It is a story about:

- compression of textbook ideas into executable NumPy form
- explicit labeling of fidelity and omission
- replacement of ad hoc chapter demos with batch-structured, contract-bound, CI-verified research assets

---

## Overarching Methodological Arc

The method evolved through five large moves:

1. **Coverage first**
   Make sure every targeted SLP3 chapter key exists and runs.

2. **Status honesty**
   Distinguish `DIRECT`, `ADAPTED`, and `SCAFFOLDED` work instead of pretending everything is equally faithful.

3. **Contract discipline**
   Give every chapter a shared payload structure so they can be compared, tested, packaged, and audited consistently.

4. **Batch packaging**
   Group chapters into coherent families `A-F`, then generate fixtures, eval packs, and batch manifests for each family.

5. **Verification duality**
   Separate local reproducibility from CI-truth:
   - local artifacts under `observability/local/`
   - CircleCI run truth under `observability/ci_latest/`

This converts the repo from a loose implementation archive into a research system with:

- stable interfaces
- reproducible packaging
- explicit epistemic limits
- live operational verification

---

## Phase 1: Coverage Before Fidelity

### Story Box

The first act is territorial, not philosophical.

The project had to answer a basic legitimacy question:

- are all the targeted SLP3 chapters and appendices actually present

At this point, breadth mattered more than elegance. The project needed complete chapter and appendix coverage before any claim about research coherence or textbook faithfulness could be taken seriously.

### Method Box

Technical move:

- create chapter entries for the expected SLP3 scope:
  - chapters `1-25`
  - appendices `A-K`

Artifacts produced:

- registry coverage
- per-chapter entry modules
- smoke and survey checks

Verification logic:

- fail if a chapter key is missing
- fail if unexpected chapter keys appear
- establish a first orphan audit

Research meaning:

- this phase turns the repo into a complete map, even before it becomes a faithful territory

---

## Phase 2: Honesty About Provenance

### Story Box

Once chapter coverage existed, the next question was trust:

- what came directly from prior NumPy work
- what was adapted
- what was only a scaffold

This phase matters because the project’s credibility depends on refusing false equivalence between:

- direct lineage
- adaptation
- placeholder-lite implementations

### Method Box

Technical move:

- classify chapters as:
  - `DIRECT`
  - `ADAPTED`
  - `SCAFFOLDED`

Artifacts produced:

- survey reports
- provenance tracking
- chapter-level status metadata

Verification logic:

- smoke and survey scripts report status counts
- README and telemetry mirror the split

Research meaning:

- this phase introduces epistemic discipline
- the repo stops saying "implemented" as if that were one thing

---

## Phase 3: Scaffolds To Full Modules

### Story Box

The third act is the long middle:

- move chapters out of lightweight placeholders
- replace payload-only scaffolds with executable, evaluated chapter modules

This is where the project became a real body of research code rather than a chapter index.

The key narrative shift here is:

- from "can it run"
- to "does it teach the chapter’s computational lesson"

### Method Box

Technical move:

- upgrade chapters family by family
- implement real NumPy chapter logic
- add evaluation and failure-case reporting

Artifacts produced:

- `FULL` chapter modules
- richer tests
- chapter-specific metrics and diagnostics

Verification logic:

- `pytest`
- smoke tests
- survey updates
- chapter outputs inspected for structural consistency

Research meaning:

- this phase transforms runnable chapter shells into pedagogical computational objects

---

## Phase 4: Contract As Research Infrastructure

### Story Box

After enough chapters became substantial, the problem changed.

The core challenge was no longer:

- how to add another chapter

It became:

- how to stop the repo from becoming structurally incoherent

This is the point where the project matured from implementation accumulation into methodology design.

### Method Box

Technical move:

- create a shared chapter contract with fields such as:
  - `lesson_objectives`
  - `core_algorithms`
  - `minimal_dataset`
  - `reference_experiments`
  - `book_vs_repo_gap`

Artifacts produced:

- chapter contract module
- normalization path for legacy payloads
- manifest generator

Verification logic:

- tests assert chapters normalize into the same schema
- manifest coverage must match the registry

Research meaning:

- the contract is the project’s internal theory of what a chapter implementation should be
- it is the bridge between code, pedagogy, and auditability

---

## Phase 5: Batch Families As Research Units

### Story Box

Once the contract existed, the repo could stop thinking chapter-by-chapter and start thinking family-by-family.

That produced six research families:

- **Batch A**: classical foundations
- **Batch B**: language models and sequence models
- **Batch C**: speech
- **Batch D**: structure and information extraction
- **Batch E**: discourse and dialogue
- **Batch F**: web appendices

The narrative significance of batching is that it creates intelligible research units larger than a single chapter but smaller than the whole repo.

### Method Box

Technical move:

- package each family with:
  - fixtures
  - eval packs
  - batch manifest
  - batch README

Artifacts produced:

- `research/batches/batch_a_classical_foundations/`
- `research/batches/batch_b_lm_and_seq_models/`
- `research/batches/batch_c_speech/`
- `research/batches/batch_d_structure_and_ie/`
- `research/batches/batch_e_discourse_and_dialogue/`
- `research/batches/batch_f_web_appendices/`

Verification logic:

- each batch has generated artifacts
- tests verify batch payload coverage
- manifest updates reflect family ownership

Research meaning:

- the batch is now the main unit of packaging, explanation, and future textbook-faithfulness work

---

## Phase 6: Dual Verification Worlds

### Story Box

As the repo became more disciplined, a new tension appeared:

- local reproducibility
- versus
- CI-truth for a specific run

Without separating them, local regeneration would overwrite richer CI-backed metadata and blur the difference between:

- "what exists on the machine"
- and
- "what happened in a verified CI run"

### Method Box

Technical move:

- split observability into:
  - `observability/local/`
  - `observability/ci_latest/`

Artifacts produced:

- local smoke, verification, and CircleCI metadata snapshots
- live CircleCI artifact uploads under `ci_latest`

Verification logic:

- local scripts write to `local`
- CircleCI writes to `ci_latest`
- `circleci_run.json` and `verification.json` carry live workflow metadata in CI

Research meaning:

- this phase gives the project a clean epistemic separation between reproducible local state and operational CI truth

---

## Phase 7: From Structural Completion To Textbook-Faithfulness

### Story Box

By the end of the batching work, the project crossed an important threshold:

- the structural problem was solved

What remains is no longer:

- missing folders
- missing manifests
- missing contract fields
- missing CI shape

What remains is the deeper research problem:

- how to make the chapters more textbook-faithful while staying honest about NumPy’s scale limits

### Method Box

Technical move:

- stop structural refactoring
- use the batch packs as the fixed substrate
- perform chapter-by-chapter fidelity upgrades only

Artifacts already supporting this:

- gap audit
- feasibility checklist
- batch manifests
- eval packs
- stable contract

Verification logic:

- improvements should preserve:
  - batch packaging
  - manifest coverage
  - contract shape
  - CI artifact flow

Research meaning:

- the repo is no longer building its scaffolding
- it is ready to do the real intellectual work of fidelity improvement

---

## Batch-Level Narrative Boxes

### Story Box: Batch A

Batch A establishes credibility.

It shows that classical NLP and the probabilistic appendices can be represented strongly and honestly in NumPy, making it the reference family for textbook-faithful upgrades.

### Method Box: Batch A

- focus: chapter `1`, chapters `2-6`, `A-D`
- role: reference-quality family
- contribution: strongest bridge between textbook method and NumPy feasibility, now anchored by an executable introductory chapter

### Story Box: Batch B

Batch B establishes restraint.

It demonstrates how to model modern chapters without pretending a NumPy repo can reproduce modern system scale.

### Method Box: Batch B

- focus: chapters `7-13`
- role: method-faithful miniatures
- contribution: explicit treatment of what is preserved and what scale prevents

### Story Box: Batch C

Batch C establishes decomposition.

It shows that speech chapters become understandable when broken into front-end, alignment, timing, and generation components rather than treated as monolithic systems.

### Method Box: Batch C

- focus: chapters `14-16`
- role: speech pipeline family
- contribution: separates DSP, ASR dynamic programming, and TTS timing

### Story Box: Batch D

Batch D establishes structure.

It centers the idea that structured prediction problems are not just classification tasks with more labels; they are problems where local scores and global consistency must interact.

### Method Box: Batch D

- focus: chapters `17-21`
- role: structured prediction and IE family
- contribution: constraints, charts, graphs, and structural diagnostics become first-class objects

### Story Box: Batch E

Batch E establishes persistence.

It shows that the highest-level language tasks depend on what persists across tokens, sentences, mentions, and turns:

- sentiment priors across domains
- identity across mentions
- coherence across sentences
- commitments across dialogue turns

### Method Box: Batch E

- focus: chapters `22-25`
- role: discourse and dialogue family
- contribution: portability, coherence margins, and commitment consistency are made operational

### Story Box: Batch F

Batch F establishes closure.

It removes the last structural mismatch with the live Stanford web table of contents by bringing the web-only appendices into the same contract, packaging, and verification regime as the rest of the repo.

### Method Box: Batch F

- focus: appendices `E-K`
- role: web-appendix family
- contribution: extends the repo into statistical parsing, formal grammar, CCG, logic, WordNet/WSD, standalone PPMI, and frame-based dialogue tracking

---

## End-To-End Research Thesis

The project’s end-to-end thesis can be stated plainly:

- SLP3 can be re-expressed as a NumPy-only executable research program
- but only if fidelity is defined at the level of method, pedagogy, diagnostics, and explicit omissions
- not at the level of modern model scale

The project therefore contributes:

1. a runnable NumPy chapter map for the targeted SLP3 scope
2. a contract for what a chapter implementation should contain
3. a family-based packaging system for evaluation and future upgrades
4. a dual local/CI verification model
5. an honest framework for textbook-faithfulness work going forward

---

## Final Story Box

The repo began as a translation project.

It became a classification project:

- direct
- adapted
- scaffolded

It then became an implementation project:

- fuller chapter modules

It then became an infrastructure project:

- contracts
- manifests
- batch packs
- CI separation

It is now ready to become what it was really trying to be all along:

- a disciplined, transparent, chapter-by-chapter program for making SLP3 computationally legible in NumPy

## Final Method Box

The next phase should not add more structure.

It should only do this:

- choose a chapter
- improve textbook faithfulness
- preserve the batch contract
- update the eval pack
- rerun local checks
- confirm in CircleCI

That is the stable research loop the project has now earned.
