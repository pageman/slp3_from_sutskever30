from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from slp3_from_sutskever30 import numpy_chapters as ch


@dataclass(frozen=True)
class ChapterSpec:
    key: str
    title: str
    source_papers: tuple[int, ...]
    runner: Callable[[], dict[str, object]]


def get_chapters() -> list[ChapterSpec]:
    return [
        ChapterSpec("2", "Words and Tokens", (), ch.chapter2_words_and_tokens),
        ChapterSpec("3", "N-gram Language Models", (), ch.chapter3_ngram_language_models),
        ChapterSpec("4", "Logistic Regression and Text Classification", (), ch.chapter4_logistic_regression),
        ChapterSpec("5", "Embeddings", (), ch.chapter5_embeddings),
        ChapterSpec("6", "Neural Networks", (), ch.chapter6_neural_networks),
        ChapterSpec("7", "Large Language Models", (27,), ch.chapter7_large_language_models),
        ChapterSpec("8", "Transformers", (13,), ch.chapter8_transformers),
        ChapterSpec("9", "Post-training: Instruction Tuning, Alignment, and Test-Time Compute", (), ch.chapter9_post_training),
        ChapterSpec("10", "Masked Language Models", (), ch.chapter10_masked_language_models),
        ChapterSpec("11", "Information Retrieval and Retrieval-Augmented Generation", (28, 29), ch.chapter11_ir_and_rag),
        ChapterSpec("12", "Machine Translation", (14,), ch.chapter12_machine_translation),
        ChapterSpec("13", "RNNs and LSTMs", (2, 3), ch.chapter13_rnns_and_lstms),
        ChapterSpec("14", "Phonetics and Speech Feature Extraction", (), ch.chapter14_phonetics_and_features),
        ChapterSpec("15", "Automatic Speech Recognition", (21,), ch.chapter15_asr),
        ChapterSpec("16", "Text-to-Speech", (), ch.chapter16_tts),
        ChapterSpec("17", "Sequence Labeling for Parts of Speech and Named Entities", (), ch.chapter17_sequence_labeling),
        ChapterSpec("18", "Context-Free Grammars and Constituency Parsing", (), ch.chapter18_constituency_parsing),
        ChapterSpec("19", "Dependency Parsing", (), ch.chapter19_dependency_parsing),
        ChapterSpec("20", "Information Extraction: Relations, Events, and Time", (), ch.chapter20_information_extraction),
        ChapterSpec("21", "Semantic Role Labeling and Argument Structure", (), ch.chapter21_semantic_role_labeling),
        ChapterSpec("22", "Lexicons for Sentiment, Affect, and Connotation", (), ch.chapter22_lexicons_sentiment_affect),
        ChapterSpec("23", "Coreference Resolution and Entity Linking", (), ch.chapter23_coreference_and_entity_linking),
        ChapterSpec("24", "Discourse Coherence", (), ch.chapter24_discourse_coherence),
        ChapterSpec("25", "Conversation and its Structure", (), ch.chapter25_conversation_structure),
        ChapterSpec("A", "Hidden Markov Models", (), ch.chapterA_hidden_markov_models),
        ChapterSpec("B", "Naive Bayes Classification", (), ch.chapterB_naive_bayes),
        ChapterSpec("C", "Kneser-Ney Smoothing", (), ch.chapterC_kneser_ney),
        ChapterSpec("D", "Spelling Correction and the Noisy Channel", (), ch.chapterD_spelling_correction),
    ]


def get_chapter_map() -> dict[str, ChapterSpec]:
    return {spec.key: spec for spec in get_chapters()}
