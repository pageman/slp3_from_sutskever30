# Textbook-Faithful NumPy Checklist 2026-03-30

This checklist asks a narrower question than the gap audit:

- Can each SLP3 chapter become textbook-faithful while still staying NumPy-only?

The answer is not uniform. Some chapters can be made very faithful in NumPy, some only in miniature, and some are not realistically faithful without non-NumPy infrastructure such as large-scale training stacks, dataset tooling, or GPU-oriented runtimes.

## Feasibility Labels

- `fully feasible in NumPy`: the chapter’s central methods, evaluation style, and failure cases can be represented faithfully in small-to-medium NumPy code without hiding the main lesson
- `faithful only in miniature`: the main algorithmic lesson can be taught honestly in NumPy, but scale, data, or systems constraints prevent a truly representative modern implementation
- `not realistically faithful without non-NumPy infrastructure`: a NumPy-only implementation can still be educational, but it will not feel faithful to the modern chapter unless external infrastructure is introduced

## Cross-Chapter Rule

To count as textbook-faithful in this repo, each chapter should eventually have:

- one canonical algorithm from the chapter
- one canonical evaluation
- one canonical ablation
- one canonical failure mode
- an explicit `book_vs_repo_gap` note documenting what remains omitted

## Checklist

| Key | Chapter | Current repo status | Feasibility | What to add for textbook-faithful status |
|---|---|---|---|---|
| 1 | Introduction | `FULL` | `fully feasible in NumPy` | Add a small speech-inclusive bridge, fuller historical framing, and a clearer map from chapter 1 micro-demos into later chapters |
| 2 | Words and Tokens | `FULL` | `fully feasible in NumPy` | Add unigram LM tokenization, Unicode edge suites, and larger compression/OOV comparisons |
| 3 | N-gram Language Models | `FULL` | `fully feasible in NumPy` | Add Katz/interpolated Kneser-Ney variants, held-out tuning, and more faithful backoff traces |
| 4 | Logistic Regression and Text Classification | `FULL` | `fully feasible in NumPy` | Add sparse feature paths, richer regularization sweeps, threshold analysis, and chapter-style dataset baselines |
| 5 | Embeddings | `FULL` | `fully feasible in NumPy` | Add GloVe-style factorization, stronger analogy/intrinsic evaluation, and corpus-frequency ablations |
| 6 | Neural Networks | `FULL` | `fully feasible in NumPy` | Add optimizer comparisons, normalization, deeper hidden-state probes, and a clearer bridge from chapter 4 to chapter 6 |
| 7 | Large Language Models | `DIRECT` | `not realistically faithful without non-NumPy infrastructure` | Keep the NumPy miniature, but document that pretraining scale, checkpoint management, and realistic inference stacks exceed a NumPy-only repo |
| 8 | Transformers | `DIRECT` | `faithful only in miniature` | Add layer norm, residual structure, positional encoding variants, and multi-layer tracing; keep it honest about scale limits |
| 9 | Post-training: Instruction Tuning, Alignment, and Test-Time Compute | `FULL` | `faithful only in miniature` | Add cleaner objective definitions, decode-trace analysis, verifier/policy separation, and policy-vs-reranker ablations |
| 10 | Masked Language Models | `FULL` | `faithful only in miniature` | Add span masking, corruption-policy comparison, deeper probes, and clearer encoder-stack ablations |
| 11 | Information Retrieval and Retrieval-Augmented Generation | `ADAPTED` | `faithful only in miniature` | Add BM25, retrieval diagnostics, chunking studies, retrieval failure taxonomies, and retrieval-vs-generation ablations |
| 12 | Machine Translation | `ADAPTED` | `faithful only in miniature` | Add seq2seq decoding variants, beam search analysis, length penalties, and alignment visualizations |
| 13 | RNNs and LSTMs | `ADAPTED` | `fully feasible in NumPy` | Add teacher forcing studies, gradient diagnostics, stacked recurrence, and stronger sequence tasks |
| 14 | Phonetics and Speech Feature Extraction | `FULL` | `fully feasible in NumPy` | Add spectrogram comparisons, invertibility checks, and more controlled front-end evaluation |
| 15 | Automatic Speech Recognition | `FULL` | `faithful only in miniature` | Add CER/WER, forced alignment tests, language-model rescoring, and more realistic decoding analysis |
| 16 | Text-to-Speech | `FULL` | `faithful only in miniature` | Add stronger duration/alignment evaluation, mel quality checks, and clearer acoustic-vs-alignment failure separation |
| 17 | Sequence Labeling for Parts of Speech and Named Entities | `FULL` | `fully feasible in NumPy` | Add CRF/Viterbi variants, richer corpus metrics, and label-confusion analysis |
| 18 | Context-Free Grammars and Constituency Parsing | `FULL` | `fully feasible in NumPy` | Add packed forests, grammar sparsity studies, and labeled tree evaluation improvements |
| 19 | Dependency Parsing | `FULL` | `fully feasible in NumPy` | Add transition-based baselines, nonprojective decoding comparisons, and detailed structural error slices |
| 20 | Information Extraction: Relations, Events, and Time | `FULL` | `faithful only in miniature` | Split entity/relation/event/time evaluation more clearly and add schema-specific error analysis |
| 21 | Semantic Role Labeling and Argument Structure | `FULL` | `fully feasible in NumPy` | Add better predicate selection studies, frame-style constraints, and richer role-set evaluation |
| 22 | Lexicons for Sentiment, Affect, and Connotation | `FULL` | `fully feasible in NumPy` | Add stronger lexicon baselines, domain-shift experiments, and more direct comparisons among sentiment dimensions |
| 23 | Coreference Resolution and Entity Linking | `FULL` | `faithful only in miniature` | Add standard coref metrics, candidate-generation ablations, and explicit cluster/link disagreement analysis |
| 24 | Discourse Coherence | `FULL` | `fully feasible in NumPy` | Add broader perturbation suites, sentence-order baselines, and stronger coherence feature studies |
| 25 | Conversation and its Structure | `FULL` | `faithful only in miniature` | Add dialogue-state evolution, adjacency-pair diagnostics, and clearer repair/grounding evaluation suites |
| A | Hidden Markov Models | `FULL` | `fully feasible in NumPy` | Add decoding variants, supervised-vs-unsupervised contrasts, and richer posterior analysis |
| B | Naive Bayes Classification | `FULL` | `fully feasible in NumPy` | Add feature smoothing ablations, threshold calibration, and more dataset comparisons |
| C | Kneser-Ney Smoothing | `FULL` | `fully feasible in NumPy` | Add modified Kneser-Ney variants and more formal held-out comparisons |
| D | Spelling Correction and the Noisy Channel | `FULL` | `fully feasible in NumPy` | Add richer candidate generation, channel-model ablations, and more realistic spelling-noise suites |
| E | Statistical Constituency Parsing | missing | `fully feasible in NumPy` | Add a dedicated statistical parsing appendix instead of folding everything into chapter 18 |
| F | Context-Free Grammars | missing | `fully feasible in NumPy` | Add a grammar-focused appendix that separates formal CFG mechanics from parsing practice |
| G | Combinatory Categorial Grammar | missing | `fully feasible in NumPy` | Add a compact combinatory rule engine and derivation visualizer |
| H | Logical Representations of Sentence Meaning | missing | `fully feasible in NumPy` | Add lambda-calculus or first-order semantics miniatures plus scope ambiguity examples |
| I | Word Senses and WordNet | missing | `fully feasible in NumPy` | Add sense inventories, similarity structure, and WSD-style toy tasks |
| J | PPMI | missing | `fully feasible in NumPy` | Extract the PPMI material from chapter 5 into a dedicated appendix with standalone evaluation |
| K | Frame-based Dialogue Systems | missing | `fully feasible in NumPy` | Add slot filling, frame state tracking, and repair handling as an appendix to complement chapter 25 |

## Bottom Line

- Classical NLP, parsing, sequence labeling, and the probabilistic appendices can become highly textbook-faithful in NumPy
- Modern chapters involving large models, retrieval systems, ASR, and TTS can be made faithful only in miniature
- The repo can become much more textbook-faithful than it is now, but only if "faithful" is defined at the method and pedagogy level rather than at modern system scale
