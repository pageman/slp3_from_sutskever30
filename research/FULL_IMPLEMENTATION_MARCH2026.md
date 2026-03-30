# Full Implementation Guide, March 2026

This document answers a narrower question than the runnable codebase:

- not "what toy NumPy demo already exists?"
- but "what must be added to promote each `SCAFFOLDED` SLP3 chapter from lite mode to a full March 2026 implementation?"

The recommendations below are grounded in the local source repos:

- `/Users/hifi/sutskever-30-implementations`
- `/Users/hifi/sutskever-30-beyond-numpy`

The cross-repo principle taken from `sutskever-30-beyond-numpy` is the most important one:

- `NumPy` should define the executable mathematical object
- `SymPy` should expose the derivation humans will otherwise hand-wave
- `tinygrad` should force a minimal autodiff view when gradients matter
- `PyTorch` should be the first practical training implementation
- `JAX` should be the parity backend that catches state/function drift
- `Cubical Agda` should only formalize the invariants worth stating, not pretend every chapter needs full formal proof

## Upgrade Rule

A `SCAFFOLDED` chapter becomes `FULL` only when all of the following are true:

1. There is a nontrivial NumPy reference implementation for the core chapter object.
2. There is at least one real evaluation protocol, not only synthetic shape checks.
3. The repo states which claims are covered and which are intentionally omitted.
4. The implementation exposes a chapter-specific failure mode, not only success cases.
5. At least one of `SymPy`, `tinygrad`, `PyTorch`, `JAX`, or `Cubical Agda` is used where it adds real explanatory leverage.

## Chapter 2

Full implementation target:
- Unicode-aware tokenization with deterministic normalization policy.
- Multiple tokenizers: whitespace, regex, subword BPE, unigram LM.
- Corpus statistics: type-token growth, OOV curves, boundary ambiguity examples.

Use these inspirations:
- Paper 23 MDL: tokenization is model selection, not preprocessing.
- Paper 25 Kolmogorov complexity: the best tokenization compresses downstream regularities, not strings alone.

Counterintuitive insight:
- The right full implementation should optimize for *error locality*, not token count. Humans over-optimize vocabulary size; models fail more often because bad tokenizers smear one linguistic mistake across many positions.

What is missing right now:
- Learned subword induction.
- Evaluation on compression, OOV handling, and downstream reconstruction stability.

## Chapter 3

Full implementation target:
- Count-based n-gram family with Laplace, absolute discounting, Katz, and Kneser-Ney.
- Streaming corpus ingestion and held-out perplexity.
- Backoff-trace introspection for each prediction.

Use these inspirations:
- Paper 22 Scaling laws: n-gram behavior must be plotted across data size, not presented at one point estimate.
- Paper 30 Lost in the Middle: long-context failures should be measured by position, not just aggregate perplexity.

Counterintuitive insight:
- Full n-gram implementations should treat continuation diversity as the first-class signal and raw counts as secondary. Humans default to counts; models care more about *how many distinct contexts survive compression*.

What is missing right now:
- Held-out perplexity curves.
- Competing smoothers.
- Backoff diagnostics.

## Chapter 4

Full implementation target:
- Sparse bag-of-words and hashed-feature logistic regression.
- Calibrated probability outputs, not just top-1 labels.
- Class-imbalance handling and feature attribution reports.

Use these inspirations:
- Paper 5 Pruning and MDL: feature pruning and description-length penalties should be native, not bolt-ons.
- Paper 26 CS231n fundamentals: train/val diagnostics matter as much as the classifier.

Counterintuitive insight:
- The decisive upgrade is not adding more features; it is adding *calibration and ablation*. Humans think linear models fail from underfitting; in practice, they fail because nobody measures whether confidence tracks reality.

What is missing right now:
- Regularization sweeps.
- Probability calibration.
- Sparse or hashed features.

## Chapter 5

Full implementation target:
- PPMI, GloVe-style factorization, and SGNS-style negative sampling in NumPy.
- Intrinsic geometry diagnostics: analogy tests, isotropy, neighborhood purity.
- Context-window and frequency-threshold sweeps.

Use these inspirations:
- Paper 17 VAE: latent geometry matters more than pretty nearest neighbors.
- Paper 23 MDL: embedding dimensionality should be justified by compression or reconstruction tradeoff.

Counterintuitive insight:
- Full embeddings should optimize *neighborhood stability under corpus perturbation*, not analogy score alone. Humans overrate semantic demos; models need representations whose geometry does not collapse when the corpus shifts slightly.

What is missing right now:
- Learned embeddings beyond truncated SVD.
- Geometry and stability diagnostics.
- Frequency-sensitive evaluation.

## Chapter 6

Full implementation target:
- MLP training stack with initialization sweeps, loss curves, calibration, and failure cases.
- Batch normalization, dropout, and optimizer comparisons.
- A clear bridge from chapter 4 linear baselines to nonlinear gains.

Use these inspirations:
- Paper 3 LSTM and paper 4 RNN regularization: the repo should expose gradient flow and regularization mechanics, not just accuracy.
- Paper 26 CS231n: debugging instrumentation is part of the implementation, not ancillary.

Counterintuitive insight:
- The full chapter should center *what the network forgets*, not what it fits. Humans focus on expressivity; neural systems are better understood by examining which simple linear separations they accidentally destroy.

What is missing right now:
- Optimizer and regularization comparisons.
- Internal activation diagnostics.
- Representation-collapse tests.

## Chapter 9

Full implementation target:
- Supervised fine-tuning, pairwise preference optimization, rejection sampling, reranking, and budgeted decoding in one coherent pipeline.
- Separation between policy quality and verifier quality.
- Explicit test-time compute experiments.

Use these inspirations:
- Paper 27 Multi-token prediction: train-time and decode-time objectives should be co-designed.
- Paper 29 RAG and paper 30 Lost in the Middle: post-training is really retrieval-and-selection engineering under context constraints.

Counterintuitive insight:
- The full implementation should allocate more code to *candidate generation and evaluation traces* than to the policy loss. Humans think alignment is a loss-function chapter; in 2026 it is more accurately a *search control systems* chapter.

What is missing right now:
- Preference datasets and objective variants.
- Verifier/reranker loop.
- Budget-sensitive decoding benchmarks.

## Chapter 10

Full implementation target:
- True masked-token training with dynamic masking, span masking, and corruption policies.
- Bidirectional encoder with residuals, layer norm, and position handling.
- Probing tasks that justify why masked pretraining helps.

Use these inspirations:
- Paper 13 Transformer: attention block fidelity matters.
- Paper 15 Identity mappings: residual pathway quality matters more than depth for stable encoder training.

Counterintuitive insight:
- The missing full feature is not mask prediction itself; it is *corruption diversity*. Humans implement a mask token and stop. Models become useful only when corruption policies teach invariance rather than a single special-token trick.

What is missing right now:
- Multi-layer encoder.
- Dynamic corruption policies.
- Representation probes.

## Chapter 14

Full implementation target:
- Real DSP front end: pre-emphasis, framing, windowing, FFT, mel filterbanks, deltas, CMVN.
- Phone-level or speaker-level probe tasks.
- Comparative ablations between raw, spectral, and mel representations.

Use these inspirations:
- Paper 11 Dilated convolutions: temporal receptive field design matters in speech just as in vision.
- Paper 21 CTC: downstream alignment behavior should validate the feature pipeline.

Counterintuitive insight:
- The full feature extractor should be optimized for *alignment entropy*, not only perceptual plausibility. Humans judge spectrograms by appearance; speech models care more about whether successive frames make alignments easy to disambiguate.

What is missing right now:
- Proper DSP stages.
- Delta and normalization features.
- Probe tasks and ablations.

## Chapter 16

Full implementation target:
- Text normalization, grapheme-to-phoneme, duration modeling, acoustic modeling, and vocoder boundary.
- Teacher-forced and non-teacher-forced generation evaluation.
- Attention failure diagnostics and stop-token behavior.

Use these inspirations:
- Paper 14 Bahdanau attention: alignment is central.
- Paper 20 NTM: externalized alignment memory is a better mental model than "just another seq2seq."

Counterintuitive insight:
- The full TTS chapter should treat durations as the main latent variable and acoustics as conditional rendering. Humans focus on spectrogram generation; models are mostly bottlenecked by *timing correctness*.

What is missing right now:
- Front-end normalization and G2P.
- Duration/attention modeling.
- Teacher-forcing mismatch evaluation.

## Chapter 17

Full implementation target:
- Token encoder, character encoder, CRF or constrained decoder, and corpus-level BIO evaluation.
- Separate analyses for segmentation errors versus label confusions.
- Span-level calibration, not only token accuracy.

Use these inspirations:
- Paper 6 Pointer networks: sequence labeling becomes clearer when viewed as constrained index selection.
- Paper 18 Relational RNN: token decisions should condition on inter-token memory, not only local features.

Counterintuitive insight:
- A full sequence-labeling implementation should optimize *boundary certainty* more than tag accuracy. Humans over-focus on labels; models lose more utility when they misplace the span edges than when they confuse two semantically similar tags.

What is missing right now:
- Constrained decoding.
- Boundary-sensitive metrics.
- Char/token fusion beyond averaging.

## Chapter 18

Full implementation target:
- Grammar induction or manually specified CFGs, CKY parsing, forest extraction, and tree evaluation.
- Lexicalization choices and unary-chain handling.
- Ambiguity visualization and chart sparsification.

Use these inspirations:
- Paper 8 Seq2Seq for sets: parse forests should be treated as structured sets before final ordering.
- Paper 23 MDL: grammar quality should be judged by compression of treebank structure, not only F1.

Counterintuitive insight:
- The real full implementation should spend more effort on *ambiguity accounting* than on single-best trees. Humans ask for the parse; models benefit more from knowing *where the chart stays uncertain*.

What is missing right now:
- Real grammar machinery.
- Backpointer forests.
- Parse evaluation and ambiguity reporting.

## Chapter 19

Full implementation target:
- Arc-factored and transition-based dependency parsing baselines.
- MST decoding for nonprojective cases and constrained projective decoding.
- Error taxonomy by relation type, length, and crossing complexity.

Use these inspirations:
- Paper 12 GNNs: dependency parsing is naturally graph-structured, not merely sequential.
- Paper 16 Relational reasoning: arc decisions should be implemented as relation scoring, not token classification.

Counterintuitive insight:
- Full dependency parsing should model *wrong but self-consistent trees* explicitly. Humans count attachment accuracy; models fail most dangerously when they build globally coherent but semantically wrong graph structure.

What is missing right now:
- Real decoding algorithms.
- Relation labeling.
- Structural diagnostics.

## Chapter 20

Full implementation target:
- Span proposal, entity typing, relation classification, event trigger detection, argument linking, and temporal links.
- Joint decoding constraints across subtasks.
- Evaluation that punishes cascading errors instead of isolated heads.

Use these inspirations:
- Paper 16 Relational reasoning and paper 12 GNNs: IE is better implemented as graph construction than as a bag of classifiers.
- Paper 29 RAG: retrieval over candidate events and schemas should be native.

Counterintuitive insight:
- A full IE system should prioritize *consistency propagation* over local classifier strength. Humans build separate heads; models gain more from enforcing that impossible event graphs never become legal outputs.

What is missing right now:
- Joint span/event graph.
- Schema constraints.
- Cascading evaluation.

## Chapter 21

Full implementation target:
- Predicate detection, argument span proposal, role labeling, and constraint-aware decoding.
- Distinct handling for core roles, adjuncts, and null arguments.
- Predicate-conditioned contextualization.

Use these inspirations:
- Paper 14 Bahdanau attention: each predicate should induce a different alignment view over the sentence.
- Paper 6 Pointer networks: argument extraction is often better framed as selecting spans than tagging tokens.

Counterintuitive insight:
- The full SRL implementation should model *predicate-specific worldviews*, not one sentence representation reused for every predicate. Humans reuse encodings; models need the sentence to be re-read differently for each predicate.

What is missing right now:
- Predicate-conditioned encoding.
- Span proposal and constrained decoding.
- Role-type diagnostics.

## Chapter 22

Full implementation target:
- Lexicon induction, domain adaptation, compositional polarity rules, and sentence/document affect aggregation.
- Calibration against domain shift.
- Explicit distinction between valence, arousal, dominance, and connotation.

Use these inspirations:
- Paper 5 Pruning and paper 23 MDL: lexicons should stay small and interpretable unless complexity measurably pays off.
- Paper 22 Scaling laws: lexicon utility should be reported as a function of corpus and supervision scale.

Counterintuitive insight:
- Full sentiment lexicon systems should optimize *portability under domain shift*, not in-domain sentiment accuracy. Humans celebrate benchmark wins; the real value of lexicons in 2026 is that they degrade more gracefully than giant classifiers.

What is missing right now:
- Lexicon induction.
- Composition rules.
- Cross-domain evaluation.

## Chapter 23

Full implementation target:
- Mention detection, pair scoring, clustering, and candidate entity linking.
- Cross-document entity memory and deferred disambiguation.
- Joint evaluation of coreference and linking, not separate islands.

Use these inspirations:
- Paper 20 NTM: external memory is the right abstraction for entity state.
- Paper 28 DPR and paper 29 RAG: entity linking should retrieve candidate KB entries as part of inference, not after the fact.

Counterintuitive insight:
- The full chapter should delay commitment longer than humans prefer. People want early clustering; models perform better when *linking evidence can rewrite coreference hypotheses late*.

What is missing right now:
- Mention proposal.
- Clustering algorithm.
- Retrieval-backed linking.

## Chapter 24

Full implementation target:
- Entity-grid, sentence-ordering, discourse relation scoring, and coherence perturbation tests.
- Hierarchical document encoders.
- Contrastive objectives over minimally corrupted discourse variants.

Use these inspirations:
- Paper 30 Lost in the Middle: discourse models must measure positional salience failures explicitly.
- Paper 18 Relational RNN: coherence depends on persistent multi-sentence memory, not only pairwise transitions.

Counterintuitive insight:
- Full discourse coherence should be implemented as *robustness to near-correct documents*, not as binary well-formedness. Humans notice obvious incoherence; models need training on tiny perturbations that feel almost acceptable.

What is missing right now:
- Perturbation suite.
- Hierarchical discourse state.
- Entity plus relation modeling together.

## Chapter 25

Full implementation target:
- Dialogue act classification, turn-taking prediction, grounding state, repair detection, and long-range conversation memory.
- Multi-speaker state tracking.
- Retrieval or memory for persistent entities and commitments.

Use these inspirations:
- Paper 18 Relational RNN: multi-slot conversation memory is the core missing ingredient.
- Paper 29 RAG: conversations need external recall of prior commitments.
- Paper 30 Lost in the Middle: long dialogue context should be stress-tested for position bias.

Counterintuitive insight:
- The full conversation chapter should optimize *commitment consistency* before response quality. Humans judge chat by fluency; conversational systems fail harder when they forget what they already implicitly promised.

What is missing right now:
- Speaker-conditioned recurrent memory.
- Repair and grounding signals.
- Long-context commitment tests.

## Appendix A

Full implementation target:
- Forward, backward, Viterbi, posterior marginals, and EM/Baum-Welch.
- Supervised and unsupervised HMM training.
- Explicit state-aliasing diagnostics.

Use these inspirations:
- Paper 21 CTC: dynamic programming should be treated as a first-class computational object.
- SymPy layer from the broader stack: derive normalization and marginal identities explicitly.

Counterintuitive insight:
- Full HMM work should expose *state non-identifiability* as a feature, not a bug. Humans expect named hidden states; models care about equivalent latent partitions that produce the same likelihood.

What is missing right now:
- Forward-backward.
- EM training.
- Posterior diagnostics.

## Appendix B

Full implementation target:
- Bernoulli and multinomial Naive Bayes variants, calibration, smoothing sweeps, and odds-ratio explanations.
- Failure analysis on correlated features.

Use these inspirations:
- Paper 23 MDL: the appeal of Naive Bayes is partly description-length efficiency.
- Paper 26 CS231n: baseline quality depends on measurement discipline.

Counterintuitive insight:
- Full Naive Bayes should lean into *useful wrongness*. Humans dismiss the independence assumption; the model remains valuable because its errors are structurally interpretable and therefore debuggable.

What is missing right now:
- Variant comparison.
- Calibration.
- Correlation-failure diagnostics.

## Appendix C

Full implementation target:
- Full Kneser-Ney family: interpolated, modified Kneser-Ney, continuation bookkeeping, multi-order recursion, and perplexity evaluation.
- Side-by-side comparison to simpler smoothing methods.

Use these inspirations:
- Paper 22 Scaling laws: smoothing quality should be plotted over corpus size.
- Paper 23 MDL: the penalty budget hidden inside smoothing must be made explicit.

Counterintuitive insight:
- Full Kneser-Ney should be implemented as a *continuation-estimation algorithm*, not a discount heuristic. Humans remember the formula; models benefit when the continuation bookkeeping becomes the conceptual center.

What is missing right now:
- Multi-order recursion.
- Modified discount estimation.
- Perplexity comparisons.

## Appendix D

Full implementation target:
- Noisy-channel spelling correction with learned confusion matrices, candidate generation tiers, and language-model reranking.
- Real-word error handling, not only non-word correction.
- Latency-aware candidate pruning.

Use these inspirations:
- Paper 28 DPR: candidate generation and reranking are retrieval problems.
- Paper 29 RAG: correction should be retrieval plus generation under uncertainty.
- Paper 23 MDL: the best correction is the one that shortens total description length, not the one with minimal edit distance.

Counterintuitive insight:
- The full spelling system should rank candidates by *global explanation cost*, not local edit plausibility. Humans over-trust edit distance; models do better when the language model can overrule a smaller edit if it yields a much simpler sentence-level explanation.

What is missing right now:
- Learned confusion model.
- Real-word correction.
- LM-based reranking.

## March 2026 Prioritization Order

If the goal is maximum upgrade per unit work, implement in this order:

1. Appendix `C`, chapter `3`, and chapter `10`
2. Chapter `17`, chapter `18`, and chapter `19`
3. Chapter `23`, chapter `24`, and chapter `25`
4. Chapter `14`, chapter `16`, and chapter `21`
5. Chapter `20` and chapter `22`
6. Chapter `2`, chapter `4`, chapter `5`, chapter `6`, appendix `A`, appendix `B`, appendix `D`

Why this order:

- it converts the weakest probabilistic and structured-prediction scaffolds first
- it creates reusable infrastructure for later chapters
- it upgrades chapters where lite demos are most misleading if left unexpanded
