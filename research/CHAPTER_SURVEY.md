# Chapter Survey

Status taxonomy used in this repository:

- `DIRECT`: chapter can map one-to-one onto an existing NumPy implementation lineage from `pageman/sutskever-30-implementations` or the NumPy slices of `sutskever-30-beyond-numpy`
- `ADAPTED`: chapter is mostly derived from those repos but required chapter-specific adaptation or multi-paper composition
- `SCAFFOLDED`: chapter does not have a clean direct source implementation in those repos, so this repository provides a NumPy-only lite implementation

Current survey:

| Key | Status | Source papers | Notes |
|---|---|---:|---|
| 2 | `SCAFFOLDED` | - | NumPy tokenization and edit-distance lite chapter |
| 3 | `SCAFFOLDED` | - | NumPy n-gram language model lite chapter |
| 4 | `SCAFFOLDED` | - | NumPy logistic regression text classification lite chapter |
| 5 | `SCAFFOLDED` | - | NumPy PPMI/embedding lite chapter |
| 6 | `SCAFFOLDED` | - | NumPy MLP text-classification lite chapter |
| 7 | `DIRECT` | 27 | Direct lineage from multi-token prediction |
| 8 | `DIRECT` | 13 | Direct lineage from transformer attention |
| 9 | `SCAFFOLDED` | - | Lite post-training objective demo |
| 10 | `SCAFFOLDED` | - | Lite masked-language-model demo |
| 11 | `ADAPTED` | 28,29 | Combined dense retrieval and RAG chapter |
| 12 | `ADAPTED` | 14 | Adapted additive-attention MT chapter |
| 13 | `ADAPTED` | 2,3 | Adapted RNN/LSTM chapter |
| 14 | `SCAFFOLDED` | - | Lite speech-feature chapter |
| 15 | `ADAPTED` | 21 | Adapted CTC/ASR chapter |
| 16 | `SCAFFOLDED` | - | Lite text-to-speech chapter |
| 17 | `SCAFFOLDED` | - | Lite sequence-labeling chapter |
| 18 | `SCAFFOLDED` | - | Lite CKY/chart parsing chapter |
| 19 | `SCAFFOLDED` | - | Lite dependency parsing chapter |
| 20 | `SCAFFOLDED` | - | Lite IE/relation chapter |
| 21 | `SCAFFOLDED` | - | Lite semantic-role-labeling chapter |
| 22 | `SCAFFOLDED` | - | Lite sentiment/affect lexicon chapter |
| 23 | `SCAFFOLDED` | - | Lite coreference/entity-linking chapter |
| 24 | `SCAFFOLDED` | - | Lite discourse-coherence chapter |
| 25 | `SCAFFOLDED` | - | Lite conversation-structure chapter |
| A | `SCAFFOLDED` | - | Lite HMM appendix chapter |
| B | `SCAFFOLDED` | - | Lite Naive Bayes appendix chapter |
| C | `SCAFFOLDED` | - | Lite Kneser-Ney appendix chapter |
| D | `SCAFFOLDED` | - | Lite spelling-correction appendix chapter |

Orphan audit result:

- Expected chapter set: `2-25`, `A-D`
- Orphaned chapters: none
- Unexpected chapter entries: none
