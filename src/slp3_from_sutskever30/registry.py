from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from slp3_from_sutskever30 import numpy_chapters as ch


@dataclass(frozen=True)
class ChapterSpec:
    key: str
    title: str
    implementation_status: str
    source_papers: tuple[int, ...]
    runner: Callable[[], dict[str, object]]


EXPECTED_CHAPTER_KEYS: tuple[str, ...] = tuple([str(index) for index in range(2, 26)] + ["A", "B", "C", "D"])


def get_chapters() -> list[ChapterSpec]:
    return [
        ChapterSpec("2", "Words and Tokens", "SCAFFOLDED", (), ch.chapter2_words_and_tokens),
        ChapterSpec("3", "N-gram Language Models", "SCAFFOLDED", (), ch.chapter3_ngram_language_models),
        ChapterSpec("4", "Logistic Regression and Text Classification", "SCAFFOLDED", (), ch.chapter4_logistic_regression),
        ChapterSpec("5", "Embeddings", "SCAFFOLDED", (), ch.chapter5_embeddings),
        ChapterSpec("6", "Neural Networks", "SCAFFOLDED", (), ch.chapter6_neural_networks),
        ChapterSpec("7", "Large Language Models", "DIRECT", (27,), ch.chapter7_large_language_models),
        ChapterSpec("8", "Transformers", "DIRECT", (13,), ch.chapter8_transformers),
        ChapterSpec("9", "Post-training: Instruction Tuning, Alignment, and Test-Time Compute", "SCAFFOLDED", (), ch.chapter9_post_training),
        ChapterSpec("10", "Masked Language Models", "SCAFFOLDED", (), ch.chapter10_masked_language_models),
        ChapterSpec("11", "Information Retrieval and Retrieval-Augmented Generation", "ADAPTED", (28, 29), ch.chapter11_ir_and_rag),
        ChapterSpec("12", "Machine Translation", "ADAPTED", (14,), ch.chapter12_machine_translation),
        ChapterSpec("13", "RNNs and LSTMs", "ADAPTED", (2, 3), ch.chapter13_rnns_and_lstms),
        ChapterSpec("14", "Phonetics and Speech Feature Extraction", "SCAFFOLDED", (), ch.chapter14_phonetics_and_features),
        ChapterSpec("15", "Automatic Speech Recognition", "ADAPTED", (21,), ch.chapter15_asr),
        ChapterSpec("16", "Text-to-Speech", "SCAFFOLDED", (), ch.chapter16_tts),
        ChapterSpec("17", "Sequence Labeling for Parts of Speech and Named Entities", "SCAFFOLDED", (), ch.chapter17_sequence_labeling),
        ChapterSpec("18", "Context-Free Grammars and Constituency Parsing", "SCAFFOLDED", (), ch.chapter18_constituency_parsing),
        ChapterSpec("19", "Dependency Parsing", "SCAFFOLDED", (), ch.chapter19_dependency_parsing),
        ChapterSpec("20", "Information Extraction: Relations, Events, and Time", "SCAFFOLDED", (), ch.chapter20_information_extraction),
        ChapterSpec("21", "Semantic Role Labeling and Argument Structure", "SCAFFOLDED", (), ch.chapter21_semantic_role_labeling),
        ChapterSpec("22", "Lexicons for Sentiment, Affect, and Connotation", "SCAFFOLDED", (), ch.chapter22_lexicons_sentiment_affect),
        ChapterSpec("23", "Coreference Resolution and Entity Linking", "SCAFFOLDED", (), ch.chapter23_coreference_and_entity_linking),
        ChapterSpec("24", "Discourse Coherence", "SCAFFOLDED", (), ch.chapter24_discourse_coherence),
        ChapterSpec("25", "Conversation and its Structure", "SCAFFOLDED", (), ch.chapter25_conversation_structure),
        ChapterSpec("A", "Hidden Markov Models", "SCAFFOLDED", (), ch.chapterA_hidden_markov_models),
        ChapterSpec("B", "Naive Bayes Classification", "SCAFFOLDED", (), ch.chapterB_naive_bayes),
        ChapterSpec("C", "Kneser-Ney Smoothing", "SCAFFOLDED", (), ch.chapterC_kneser_ney),
        ChapterSpec("D", "Spelling Correction and the Noisy Channel", "SCAFFOLDED", (), ch.chapterD_spelling_correction),
    ]


def get_chapter_map() -> dict[str, ChapterSpec]:
    return {spec.key: spec for spec in get_chapters()}


def get_orphaned_chapter_keys() -> list[str]:
    present = {spec.key for spec in get_chapters()}
    return [key for key in EXPECTED_CHAPTER_KEYS if key not in present]


def get_unexpected_chapter_keys() -> list[str]:
    expected = set(EXPECTED_CHAPTER_KEYS)
    return [spec.key for spec in get_chapters() if spec.key not in expected]
