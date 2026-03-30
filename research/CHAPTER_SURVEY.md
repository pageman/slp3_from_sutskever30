# Chapter Survey

Status taxonomy used in this repository:

- `FULL`: chapter now has a richer NumPy-only chapter contract with evaluation and explicit failure modes
- `DIRECT`: chapter can map one-to-one onto an existing NumPy implementation lineage from `pageman/sutskever-30-implementations` or the NumPy slices of `sutskever-30-beyond-numpy`
- `ADAPTED`: chapter is mostly derived from those repos but required chapter-specific adaptation or multi-paper composition
- `SCAFFOLDED`: chapter does not have a clean direct source implementation in those repos, so this repository provides a NumPy-only lite implementation

Current survey:

| Key | Status | Source papers | Notes |
|---|---|---:|---|
| 2 | `FULL` | - | Full batch-1 tokenizer chapter with normalization, BPE-lite, OOV, and error-locality diagnostics |
| 3 | `FULL` | - | Full batch-1 n-gram chapter with held-out perplexity, backoff traces, and Kneser-Ney comparison |
| 4 | `FULL` | - | Full batch-1 logistic regression chapter with calibration and feature attribution |
| 5 | `FULL` | - | Full batch-1 embeddings chapter with PPMI, SGNS-style training, and geometry diagnostics |
| 6 | `FULL` | - | Full batch-1 neural network chapter with baseline comparison and representation-health metrics |
| 7 | `DIRECT` | 27 | Direct lineage from multi-token prediction |
| 8 | `DIRECT` | 13 | Direct lineage from transformer attention |
| 9 | `FULL` | - | Full batch-2 post-training chapter with SFT, pairwise preferences, verifier reranking, and budget traces |
| 10 | `FULL` | - | Full batch-2 masked-LM chapter with dynamic masking policies, encoder stack, and probe metrics |
| 11 | `ADAPTED` | 28,29 | Combined dense retrieval and RAG chapter |
| 12 | `ADAPTED` | 14 | Adapted additive-attention MT chapter |
| 13 | `ADAPTED` | 2,3 | Adapted RNN/LSTM chapter |
| 14 | `FULL` | - | Full batch-3 DSP chapter with pre-emphasis, mel, deltas, CMVN, and alignment-entropy diagnostics |
| 15 | `FULL` | 21 | Full batch-3 ASR chapter with DSP frontend hooks, CTC-like loss separation, and beam/alignment diagnostics |
| 16 | `FULL` | - | Full batch-3 TTS chapter with normalization, G2P, durations, attention alignment, and teacher-forcing gap metrics |
| 17 | `FULL` | - | Full structured-prediction batch with constrained BIO decoding and boundary-sensitive metrics |
| 18 | `FULL` | - | Full parsing batch with explicit grammar rules, CKY backpointers, tree reconstruction, and ambiguity diagnostics |
| 19 | `FULL` | - | Full parsing batch with projective vs MST decoding, relation labels, and structural diagnostics |
| 20 | `FULL` | - | Full span-graph batch with entity, relation, event, and time heads plus schema-constrained decoding |
| 21 | `FULL` | - | Full structured-prediction batch with predicate-conditioned role scoring and constrained role decoding |
| 22 | `FULL` | - | Full discourse batch with lexicon induction, composition rules, and domain-shift evaluation |
| 23 | `FULL` | - | Full span-graph batch with clustering, candidate retrieval, and late linking revision |
| 24 | `FULL` | - | Full discourse batch with entity-grid, perturbation-based coherence scoring, and near-correct document evaluation |
| 25 | `FULL` | - | Full dialogue batch with dialogue acts, repair, grounding, and commitment-consistency evaluation |
| A | `SCAFFOLDED` | - | Lite HMM appendix chapter |
| B | `SCAFFOLDED` | - | Lite Naive Bayes appendix chapter |
| C | `SCAFFOLDED` | - | Lite Kneser-Ney appendix chapter |
| D | `SCAFFOLDED` | - | Lite spelling-correction appendix chapter |

Orphan audit result:

- Expected chapter set: `2-25`, `A-D`
- Orphaned chapters: none
- Unexpected chapter entries: none
