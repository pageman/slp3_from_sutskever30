# SLP3 Gap Audit 2026-03-30

This audit compares the current repository against the live Stanford SLP3 table of contents released January 6, 2026:

- https://web.stanford.edu/~jurafsky/slp3/

## Rubric

- `title match`: chapter title and broad placement match SLP3, but the repo implementation is clearly narrower than the book chapter and only partially captures the lesson
- `topic match`: chapter covers the same topic family, but the method stack is reduced, adapted, or composed from adjacent work rather than closely following the chapter’s central methods
- `method match`: the repo includes runnable code for the core methods named or strongly implied by the SLP3 chapter title, even if it remains toy-scale rather than textbook-complete
- `not yet aligned`: missing, materially off-topic, or absent from the current repo scope

## Summary

- In scope here: chapters `2-25`, appendices `A-D`
- Out of scope here: chapter `1`, appendices `E-K`
- Result by audit classification:
  - `method match`: `22`
  - `topic match`: `6`
  - `title match`: `0`
  - `not yet aligned`: `8`

## Chapter Audit

| Key | Stanford title | Repo status | Audit class | Why |
|---|---|---|---|---|
| 1 | Introduction | missing | `not yet aligned` | Chapter 1 is not implemented in this repo |
| 2 | Words and Tokens | `FULL` | `method match` | Normalization, tokenization, BPE-lite, OOV, and edit-locality diagnostics match the chapter’s core computational concerns |
| 3 | N-gram Language Models | `FULL` | `method match` | Unigram/bigram/trigram behavior, backoff, perplexity, and Kneser-Ney-style comparison are directly represented |
| 4 | Logistic Regression and Text Classification | `FULL` | `method match` | The chapter’s central classifier family is implemented directly with calibration and feature diagnostics |
| 5 | Embeddings | `FULL` | `method match` | PPMI and SGNS-style embedding construction align with the chapter’s main method family |
| 6 | Neural Networks | `FULL` | `method match` | Feed-forward neural baselines and representation diagnostics are present in code |
| 7 | Large Language Models | `DIRECT` | `topic match` | The topic matches, but the repo is a compact NumPy analog rather than a true LLM training/inference chapter implementation |
| 8 | Transformers | `DIRECT` | `method match` | Attention and transformer mechanics are directly represented, even though the implementation is toy-scale |
| 9 | Post-training: Instruction Tuning, Alignment, and Test-Time Compute | `FULL` | `method match` | SFT, preference optimization, reranking, and compute-budget traces line up with the chapter’s method core |
| 10 | Masked Language Models | `FULL` | `method match` | Masking policies plus a bidirectional encoder stack give a real method-level chapter analog |
| 11 | Information Retrieval and Retrieval-Augmented Generation | `ADAPTED` | `method match` | Dense retrieval and RAG are both explicitly implemented, though via adaptation rather than direct Stanford chapter code |
| 12 | Machine Translation | `ADAPTED` | `topic match` | MT is covered, but the implementation centers a narrower additive-attention setup than the full chapter likely spans |
| 13 | RNNs and LSTMs | `ADAPTED` | `method match` | RNN/LSTM mechanics are directly implemented and align with the chapter title |
| 14 | Phonetics and Speech Feature Extraction | `FULL` | `method match` | DSP front-end steps such as framing, mel features, deltas, and CMVN are directly represented |
| 15 | Automatic Speech Recognition | `FULL` | `method match` | Acoustic logits, CTC-like dynamic programming, and beam/alignment diagnostics match the method core |
| 16 | Text-to-Speech | `FULL` | `method match` | Normalization, G2P, durations, attention alignment, and mel rendering align with the TTS chapter theme |
| 17 | Sequence Labeling for Parts of Speech and Named Entities | `FULL` | `method match` | Constrained BIO decoding and boundary-sensitive metrics directly match the sequence-labeling lesson |
| 18 | Context-Free Grammars and Constituency Parsing | `FULL` | `method match` | CFG rules, CKY parsing, and tree reconstruction are directly represented |
| 19 | Dependency Parsing | `FULL` | `method match` | Dependency decoding, labels, and structural diagnostics align well with the chapter |
| 20 | Information Extraction: Relations, Events, and Time | `FULL` | `topic match` | IE is covered well, but the repo compresses several subproblems into one toy span-graph pipeline |
| 21 | Semantic Role Labeling and Argument Structure | `FULL` | `method match` | Predicate-conditioned role scoring and constrained decoding fit the chapter directly |
| 22 | Lexicons for Sentiment, Affect, and Connotation | `FULL` | `topic match` | Lexicon induction and composition are present, but the chapter likely carries broader lexical resources and analysis than the repo does |
| 23 | Coreference Resolution and Entity Linking | `FULL` | `topic match` | Both tasks are present, but the implementation is a compact joint clustering/linking analog rather than a broad chapter companion |
| 24 | Discourse Coherence | `FULL` | `method match` | Entity-grid-style coherence modeling and perturbation evaluation match the chapter’s central methods |
| 25 | Conversation and its Structure | `FULL` | `topic match` | Dialogue acts, repair, grounding, and commitment state are present, but the chapter scope is broader than the repo’s compact simulator |
| A | Hidden Markov Models | `FULL` | `method match` | Forward-backward, marginals, and Baum-Welch update are direct HMM appendix methods |
| B | Naive Bayes Classification | `FULL` | `method match` | Bernoulli and multinomial NB are directly implemented |
| C | Kneser-Ney Smoothing | `FULL` | `method match` | Continuation-based smoothing is explicitly implemented and compared |
| D | Spelling Correction and the Noisy Channel | `FULL` | `method match` | Noisy-channel scoring and confusion-model reranking align with the appendix topic |
| E | Statistical Constituency Parsing | missing | `not yet aligned` | Web appendix is not implemented in this repo |
| F | Context-Free Grammars | missing | `not yet aligned` | Web appendix is not implemented in this repo |
| G | Combinatory Categorial Grammar | missing | `not yet aligned` | Web appendix is not implemented in this repo |
| H | Logical Representations of Sentence Meaning | missing | `not yet aligned` | Web appendix is not implemented in this repo |
| I | Word Senses and WordNet | missing | `not yet aligned` | Web appendix is not implemented in this repo |
| J | PPMI | missing | `not yet aligned` | Web appendix is not implemented separately, although related ideas appear inside chapter 5 |
| K | Frame-based Dialogue Systems | missing | `not yet aligned` | Web appendix is not implemented in this repo |

## Bottom Line

- The repo matches the live SLP3 chapter map well for chapters `2-25` and appendices `A-D`
- The strongest alignment is in classical NLP, parsing, appendices, masked LMs, and speech front-end material
- The weakest alignment is in the broadest modern/system chapters where the repo intentionally compresses textbook scope into compact NumPy demonstrations
- The repo is not a faithful code companion to the entire online book; it is a runnable educational reimplementation set aligned to much of the SLP3 topic structure
