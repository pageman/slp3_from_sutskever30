# Implementation Backlog, March 2026

This backlog focuses on two things:

1. how to convert the 22 `SCAFFOLDED` chapters from lite mode into full implementations
2. how to improve the existing `ADAPTED` and `DIRECT` chapters so they stop being thin transcriptions and become durable chapter modules

## Cross-Cutting Refactors

These should happen before major chapter expansion:

### 1. Introduce a chapter contract

Every chapter should expose:

- `build_fixture()`
- `run_numpy()`
- `evaluate()`
- `failure_cases()`
- `chapter_notes()`

Why:

- right now most chapters only expose one runner payload
- full implementations need a clean separation between fixture generation, core algorithm, evaluation, and known failure modes

### 2. Separate chapter metadata from chapter execution

Move all provenance, status, and source mapping metadata into one manifest file or dedicated metadata module.

Why:

- the registry currently mixes identity, provenance, and executable routing
- full implementations will need richer metadata: datasets, metrics, backend coverage, status, and missing claims

### 3. Standardize payloads

Every chapter should return:

- `chapter`
- `implementation_status`
- `core_outputs`
- `metrics`
- `failure_modes`
- `sources`

Why:

- current payloads are inconsistent, which makes repo-wide tooling weaker than it should be

### 4. Add chapter-specific tests, not only smoke tests

Smoke tests should remain, but every chapter should also have:

- invariant tests
- metric sanity tests
- failure-case tests
- serialization tests if artifacts are saved

### 5. Add backend expansion where it adds leverage

Do not add all backends everywhere immediately.

Add them selectively:

- `SymPy` where the derivation is the teaching bottleneck
- `tinygrad` where minimal autodiff clarifies the chapter
- `PyTorch` where practical training matters
- `JAX` where parity and function/state separation are valuable
- `Cubical Agda` only for structural invariants worth formalizing

## SCAFFOLDED Chapters: What To Build

### 2. Words and Tokens

To become full:

- implement BPE and unigram LM tokenization
- add Unicode normalization benchmarks
- add compression and OOV evaluation
- include token-boundary ambiguity cases and error locality analysis

Priority code additions:

- tokenizer trainer
- corpus iterator
- compression evaluator

### 3. N-gram Language Models

To become full:

- add Katz and interpolated/modified Kneser-Ney
- add held-out perplexity
- add corpus scaling curves
- add backoff trace explanations

Priority code additions:

- smoothing module
- perplexity evaluator
- ablation harness

### 4. Logistic Regression

To become full:

- add sparse features and hashed features
- add L1/L2 regularization
- add calibration and threshold analysis
- add feature attribution reports

Priority code additions:

- sparse matrix path
- calibration utilities
- regularization sweeps

### 5. Embeddings

To become full:

- add GloVe-style factorization
- add SGNS-style training
- add geometry and stability diagnostics
- add frequency and window sweeps

Priority code additions:

- negative sampling trainer
- analogy benchmark
- isotropy and neighborhood stability metrics

### 6. Neural Networks

To become full:

- add optimizer comparisons
- add dropout and normalization
- add hidden-state probes and collapse diagnostics
- add systematic baseline comparisons against chapter 4

Priority code additions:

- optimizer module
- activations/regularization module
- learning-curve instrumentation

### 9. Post-training

To become full:

- implement supervised fine-tuning, pairwise preference optimization, and reranking
- add verifier-versus-policy separation
- add budget-aware decoding evaluation
- add candidate-trace logging

Priority code additions:

- preference dataset format
- reranker/verifier module
- decode budget evaluator

### 10. Masked Language Models

To become full:

- implement dynamic and span masking
- add a multi-layer bidirectional encoder
- add probing tasks for syntactic and lexical information
- add corruption-policy ablations

Priority code additions:

- corruption sampler
- encoder stack
- probe suite

### 14. Phonetics and Speech Features

To become full:

- implement real DSP stages
- add delta and CMVN features
- compare raw, spectral, and mel front ends
- test downstream alignment usefulness

Priority code additions:

- DSP module
- feature normalization
- downstream probe tasks

### 16. Text-to-Speech

To become full:

- implement text normalization and G2P
- model duration explicitly
- separate attention/alignment from acoustic prediction
- add teacher-forcing mismatch evaluation

Priority code additions:

- front-end normalization
- duration model
- acoustic decoder

### 17. Sequence Labeling

To become full:

- add BIO decoding and CRF or constrained decoding
- add boundary-sensitive metrics
- add character-token fusion beyond averaging
- add error breakdown by segmentation versus labeling

Priority code additions:

- decoder module
- corpus-level evaluator
- structured error analysis

### 18. Constituency Parsing

To become full:

- add real CFG rule machinery
- add CKY with backpointer forests
- add tree output and tree evaluation
- add ambiguity and sparsity diagnostics

Priority code additions:

- grammar representation
- chart parser
- tree reconstruction

### 19. Dependency Parsing

To become full:

- add projective and nonprojective decoding
- add relation labels
- add structural and length-based error analysis
- compare arc-factored versus transition-based baselines

Priority code additions:

- decoder algorithms
- relation classifier
- structural metrics

### 20. Information Extraction

To become full:

- add span proposal
- add entity, relation, event, and time heads
- add joint decoding constraints
- add cascading evaluation rather than isolated micro-heads

Priority code additions:

- span graph module
- schema-constrained decoder
- joint evaluator

### 21. Semantic Role Labeling

To become full:

- add predicate-conditioned encoding
- add span proposal and constrained role decoding
- separate core roles, adjuncts, and null arguments
- add predicate-centric evaluation

Priority code additions:

- predicate-conditioned encoder
- span scorer
- role decoder

### 22. Lexicons for Sentiment, Affect, and Connotation

To become full:

- add lexicon induction
- add compositional polarity rules
- add domain-shift tests
- separate valence, arousal, dominance, and connotation dimensions

Priority code additions:

- lexicon induction pipeline
- composition module
- cross-domain evaluator

### 23. Coreference Resolution and Entity Linking

To become full:

- add mention proposal
- add pair scoring and clustering
- add retrieval-backed entity linking
- support late revision of entity hypotheses

Priority code additions:

- mention detector
- clustering module
- candidate retrieval and linker

### 24. Discourse Coherence

To become full:

- add entity-grid and sentence-ordering objectives
- add coherence perturbation datasets
- add hierarchical discourse encoders
- evaluate near-correct and minimally corrupted documents

Priority code additions:

- perturbation generator
- hierarchical encoder
- discourse evaluator

### 25. Conversation Structure

To become full:

- add dialogue acts, turn-taking, grounding, and repair signals
- add multi-speaker state tracking
- add long-range commitment memory
- evaluate consistency under long contexts

Priority code additions:

- dialogue state module
- grounding/repair signals
- commitment consistency tests

### A. Hidden Markov Models

To become full:

- add forward-backward
- add Baum-Welch/EM
- add posterior marginals and state-aliasing diagnostics
- support both supervised and unsupervised learning

Priority code additions:

- dynamic programming module
- EM trainer
- posterior visualizer

### B. Naive Bayes

To become full:

- add Bernoulli and multinomial variants
- add smoothing sweeps
- add calibration
- add correlation-failure diagnostics

Priority code additions:

- model variants
- calibration evaluator
- feature-correlation diagnostics

### C. Kneser-Ney

To become full:

- add modified Kneser-Ney
- add multi-order recursion
- add perplexity comparisons versus simpler smoothing
- expose continuation bookkeeping explicitly

Priority code additions:

- recursive smoother
- discount estimator
- perplexity benchmark

### D. Spelling Correction

To become full:

- add learned confusion matrices
- support real-word errors
- add language-model reranking
- add latency-aware candidate pruning

Priority code additions:

- confusion model
- reranker
- sentence-level correction evaluator

## ADAPTED Chapters: How To Improve

### 11. Information Retrieval and RAG

Improve by:

- splitting DPR and RAG into reusable submodules instead of one chapter function
- adding retrieval diagnostics, hard negatives, and passage perturbation tests
- adding context-placement stress tests inspired by lost-in-the-middle behavior

### 12. Machine Translation

Improve by:

- separating encoder, aligner, and decoder code
- adding alignment visualization and length-generalization tests
- comparing additive attention and simple transformer baselines under the same evaluation harness

### 13. RNNs and LSTMs

Improve by:

- adding a shared recurrent core library
- exposing gradient diagnostics and memory-retention tasks
- comparing vanilla RNN, LSTM, and regularized recurrent variants under one benchmark

### 15. Automatic Speech Recognition

Improve by:

- separating acoustic model and CTC dynamic program
- adding feature-front-end hooks so chapter 14 and 15 connect cleanly
- adding alignment entropy and beam-search diagnostics

## DIRECT Chapters: How To Improve

### 7. Large Language Models

Improve by:

- extracting a reusable decoder-only core from the current chapter runner
- adding train-time versus decode-time objective comparisons
- connecting the chapter explicitly to post-training and long-context evaluation

### 8. Transformers

Improve by:

- refactoring attention, MLP, residual, and positional logic into composable blocks
- adding masked, causal, and bidirectional attention variants from one common block
- adding shape, invariance, and gradient tests beyond smoke coverage

## Why pytest is failing on GitHub

It is not currently failing because of the local Python code.

Local result:

- `python3 -m pytest /Users/hifi/Downloads/slp3_from_sutskever30`
- passes

GitHub Actions result:

- the job never starts
- the account is locked due to a billing issue

Observed GitHub annotation:

- `The job was not started because your account is locked due to a billing issue.`

So the fix is operational, not code-level:

1. resolve the GitHub billing lock on the `pageman` account
2. re-run the `pytest` workflow
3. only if the workflow then fails after startup should we debug the workflow itself
