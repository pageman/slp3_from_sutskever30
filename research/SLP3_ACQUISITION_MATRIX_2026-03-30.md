# SLP3 Acquisition Matrix

This document maps each SLP3 chapter family to the best practical anchors for:

- data sources
- benchmark suites
- systems stacks
- layer-1, layer-2, and layer-3 fit
- licensing posture
- why each anchor is the right fit

It is a planning document for moving from NumPy-only lesson fidelity toward stronger textbook fidelity and, where appropriate, modern-system parity.

## Three-Layer Nuance

This matrix should be read through a three-layer lens:

### Layer 1: Lesson Fidelity

- goal: transparent, inspectable textbook-faithful pedagogy
- typical stack: `NumPy`, `SymPy`
- success criterion: the code teaches the chapter’s core method, evaluation logic, and failure modes clearly

### Layer 2: Research Fidelity

- goal: trainable, extensible implementations that preserve the lesson while supporting stronger experiments
- typical stack: `TinyGrad`, `PyTorch`, `JAX`
- success criterion: the chapter can support compact but serious ablations, diagnostics, and benchmark-facing experiments

### Layer 3: Modern-System Parity

- goal: realistic system behavior for the broadest modern SLP3 chapters
- typical stack: `PyTorch`, `JAX`, tokenizer/runtime infrastructure, retrieval or speech tooling, checkpoint and evaluation infrastructure
- success criterion: the chapter approaches real benchmark and systems practice rather than only miniature pedagogical analogs

## Placement

This file lives in `research/` rather than a batch folder because it spans the whole project:

- chapter scope: `1-25`
- appendix scope: `A-K`
- batch scope: `A-F`

## Matrix

| Chapter(s) | Topic | Layer 1 | Layer 2 | Layer 3 | Best data source | Best benchmark | Best systems stack | Open vs licensed | Why this is the right anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1-2` | Intro, words, tokens | Strong fit | Usually unnecessary | Not needed | Small open corpora, Wikipedia slices, Project Gutenberg excerpts | Intrinsic tokenization and corpus-stat diagnostics | NumPy, SymPy | Open | These chapters are primarily pedagogical and do not need large benchmark ecosystems first. |
| `3` | N-gram language models | Strong fit | Strong fit | Not needed | Brown-style text corpora, WikiText | Perplexity on held-out text | NumPy, SymPy, PyTorch optional | Open | The chapter is best anchored by explicit smoothing and perplexity comparisons. |
| `4` | Logistic regression and text classification | Strong fit | Strong fit | Usually unnecessary | AG News, SST-style subsets, subjectivity corpora | Accuracy, calibration, F1 | NumPy, PyTorch | Open | This chapter is about linear classification behavior and calibration more than large-scale systems. |
| `5` | Embeddings | Strong fit | Strong fit | Partial fit | Text8, WikiText, small Wikipedia dumps | MTEB-style embedding tasks | NumPy, PyTorch, JAX | Open | Embedding chapters need both intrinsic geometry checks and downstream transfer evaluation. |
| `6` | Neural networks | Strong fit | Strong fit | Partial fit | Small classification corpora and synthetic stress tasks | Accuracy, calibration, collapse diagnostics | NumPy, TinyGrad, PyTorch | Open | This chapter is best anchored by representation-health metrics and controlled ablations. |
| `7` | Large language models | Partial fit | Strong fit | Essential | The Pile-style open text mixtures, curated instruction corpora | HELM-style evals, lm-eval-style suites | PyTorch, JAX, tokenizer stack, serving infra | Open | This is the clearest case where lesson fidelity can be approximated in NumPy but system parity requires full ML infrastructure. |
| `8` | Transformers | Strong fit | Strong fit | Strong fit | WikiText, C4 slices, small translation corpora | Loss, attention diagnostics, downstream transfer | NumPy, TinyGrad, PyTorch, JAX | Open | Transformers can be taught in NumPy, but parity requires trainable modern stacks and tokenization tooling. |
| `9` | Post-training and alignment | Partial fit | Strong fit | Essential | Preference datasets, instruction datasets, verifier-style corpora | MT-Bench-style evals, reward/preference diagnostics | PyTorch, JAX | Open | Alignment chapters require optimizer, sampler, and evaluation infrastructure beyond NumPy. |
| `10` | Masked language models | Strong fit | Strong fit | Partial fit | BookCorpus-style open alternatives, WikiText, Wikipedia | MLM loss, probing, GLUE-style transfer | NumPy, PyTorch, JAX | Open | This chapter is one of the modern chapters most amenable to faithful miniature implementations. |
| `11` | Retrieval and RAG | Partial fit | Strong fit | Essential | BEIR corpora, FEVER, SQuAD | BEIR, MTEB, FEVER retrieval | PyTorch, Transformers, vector DB tooling | Open | RAG needs both retrieval benchmarks and end-to-end retrieval-generation plumbing. |
| `12` | Machine translation | Partial fit | Strong fit | Essential | WMT parallel corpora | WMT shared-task metrics | PyTorch, JAX, seq2seq/tokenizer stack | Open | MT needs real decoding, tokenization, and benchmark harnesses to approach chapter parity. |
| `13` | RNNs and LSTMs | Strong fit | Strong fit | Partial fit | Penn Treebank, WikiText, small seq2seq corpora | Perplexity, sequence accuracy | NumPy, TinyGrad, PyTorch | Open | This chapter is highly matchable with compact trainable systems. |
| `14` | Speech feature extraction | Strong fit | Strong fit | Usually unnecessary | LibriSpeech, Common Voice audio slices | Feature diagnostics, downstream ASR front-end metrics | NumPy, PyTorch optional | Open | Speech front-end work is highly textbook-faithful even without large modern systems. |
| `15` | Automatic speech recognition | Partial fit | Strong fit | Essential | LibriSpeech, Common Voice | SUPERB, WER/CER suites | PyTorch, JAX, Kaldi, NeMo | Open | Real ASR parity needs decoding, manifests, beam search, and large speech tooling. |
| `16` | Text-to-speech | Partial fit | Strong fit | Essential | LJSpeech, Common Voice derivatives, small aligned corpora | MOS proxies, mel/duration diagnostics | PyTorch, JAX, NeMo | Open | TTS can be taught in miniature, but parity requires alignment and vocoder ecosystems. |
| `17` | Sequence labeling | Strong fit | Strong fit | Partial fit | CoNLL-2003, OntoNotes | CoNLL metrics | NumPy, PyTorch | Mixed: CoNLL open, OntoNotes licensed | These tasks are structurally faithful with constrained decoding and standard span/token metrics. |
| `18` | Constituency parsing | Strong fit | Strong fit | Partial fit | Penn Treebank, OntoNotes syntax slices | Span F1 | NumPy, PyTorch | Mixed: PTB access conventions vary, OntoNotes licensed | Parsing chapters are highly matchable with exact decoding and chart diagnostics. |
| `19` | Dependency parsing | Strong fit | Strong fit | Partial fit | Universal Dependencies | UAS/LAS | NumPy, PyTorch, JAX optional | Open | UD is the cleanest open anchor for textbook-faithful dependency work. |
| `20` | Information extraction | Partial fit | Strong fit | Strong fit | ACE-style corpora, FEVER evidence, DocRED-style relation data | Relation/event F1, retrieval-grounding checks | PyTorch, Transformers | Mixed | IE needs span proposal, schema constraints, and usually larger annotation ecosystems. |
| `21` | Semantic role labeling | Strong fit | Strong fit | Partial fit | PropBank, OntoNotes | SRL span F1 | NumPy, PyTorch | Mixed: PropBank openish ecosystem, OntoNotes licensed | Predicate-argument work is structurally faithful with good span metrics and decoding constraints. |
| `22` | Sentiment, affect, connotation | Strong fit | Strong fit | Partial fit | SST, GoEmotions, lexicon resources | Accuracy, macro-F1, cross-domain transfer | NumPy, PyTorch | Open | This chapter benefits from combining lexicon induction with supervised affect tasks. |
| `23` | Coreference and entity linking | Partial fit | Strong fit | Strong fit | OntoNotes, BLINK-style entity resources | Coref metrics, linking accuracy | PyTorch, Transformers | Mixed | Coref/linking needs candidate generation, clustering, and entity resources beyond toy demos. |
| `24` | Discourse coherence | Strong fit | Strong fit | Partial fit | PDTB, sentence-order corpora, RST-style resources | Coherence discrimination and relation metrics | NumPy, PyTorch | Mixed: PDTB licensed | This chapter is relatively tractable with perturbation tests and discourse relation datasets. |
| `25` | Conversation and structure | Partial fit | Strong fit | Essential | MultiWOZ, task-oriented dialogue corpora, open conversation corpora | DST, act prediction, response grounding | PyTorch, Transformers | Open | Conversation chapters need dialogue state, repair, grounding, and evaluation loops. |
| `A` | Hidden Markov Models | Strong fit | Strong fit | Not needed | Toy sequence corpora, POS-style tagged data | Sequence accuracy, likelihood | NumPy, SymPy | Open | Exact inference and EM-style updates are fully feasible and textbook-faithful in NumPy. |
| `B` | Naive Bayes | Strong fit | Strong fit | Not needed | Text classification corpora | Accuracy, calibration | NumPy, SymPy | Open | This appendix is a strong candidate for exact derivation plus empirical comparison. |
| `C` | Kneser-Ney smoothing | Strong fit | Strong fit | Not needed | Language modeling corpora | Perplexity | NumPy, SymPy | Open | This appendix is fundamentally about exact smoothing behavior and held-out evaluation. |
| `D` | Spelling correction | Strong fit | Strong fit | Not needed | Edit-distance dictionaries, toy noisy-channel corpora | Correction accuracy, reranking accuracy | NumPy, SymPy | Open | This appendix is highly faithful with exact edit and noisy-channel modeling. |
| `E-K` | Web appendices | Strong fit | Partial fit | Usually unnecessary | Appendix-specific small corpora, formal examples, chapter-local fixtures | Appendix-local correctness and diagnostic checks | NumPy, SymPy, PyTorch only where needed | Mostly open | These appendices are best anchored by exactness, formal diagnostics, and compact reproducible cases rather than scale. |

## Strongest Anchors By Resource

### Open Data / Benchmarks

- `Universal Dependencies` for `19`
- `CoNLL-2003` for `17`
- `BEIR` for `11`
- `MTEB` for `5`, `11`
- `WMT` for `12`
- `LibriSpeech` for `14-16`
- `Common Voice` for `14-16`
- `SUPERB` for `15-16`
- `MultiWOZ` for `25`

### Licensed / Restricted But High-Value

- `OntoNotes 5.0` for `17-19`, `21`, `23`
- `PDTB 3.0` for `24`

## Practical Guidance

If the goal is **Layer 1 textbook lesson fidelity**:

- prioritize `1-6`, `13`, `17-19`, `21-24`, `A-K`
- prefer exact or compact open datasets first
- keep systems stacks light unless the benchmark demands otherwise

If the goal is **Layer 2 research fidelity**:

- prioritize `5-13`, `15-16`, `20`, `23`, `25`
- add trainable implementations, stronger diagnostics, and benchmark-adjacent eval loops
- use TinyGrad, PyTorch, or JAX where NumPy stops being enough for meaningful experimentation

If the goal is **Layer 3 modern-system parity**:

- prioritize `7-12`, `15-16`, `20`, `23`, `25`
- anchor to benchmark ecosystems first
- use PyTorch or JAX plus tokenizer, decoding, and checkpoint infrastructure

## Recommendation

Treat this repo as a three-layer program:

1. `NumPy/SymPy` for lesson fidelity
2. `TinyGrad/PyTorch/JAX` for research-fidelity implementations
3. benchmark-anchored systems work for modern chapter parity

This matrix is the acquisition plan for layers `2` and `3`.
