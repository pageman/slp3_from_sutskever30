from __future__ import annotations

from collections import Counter
import math

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import edit_distance, tokenize_words


def _bigram_language_model(sentences: list[list[str]]) -> dict[tuple[str, str], float]:
    counts: Counter[tuple[str, str]] = Counter()
    context_totals: Counter[str] = Counter()
    for sentence in sentences:
        augmented = ["<s>"] + sentence + ["</s>"]
        for left, right in zip(augmented, augmented[1:]):
            counts[(left, right)] += 1
            context_totals[left] += 1
    return {pair: count / context_totals[pair[0]] for pair, count in counts.items()}


def _sentence_nll(tokens: list[str], model: dict[tuple[str, str], float]) -> float:
    augmented = ["<s>"] + tokens + ["</s>"]
    loss = 0.0
    for left, right in zip(augmented, augmented[1:]):
        prob = model.get((left, right), 1e-6)
        loss -= math.log(prob)
    return loss


def build_fixture() -> dict[str, object]:
    corpus = [
        "Language technology turns text into measurable structure.",
        "Tokenization changes what a model can learn.",
        "Evaluation matters because impressive demos can still fail silently.",
        "Speech and language processing connects symbols, probabilities, and decisions.",
    ]
    probes = {
        "generation_seed": "language technology",
        "classification_texts": [
            "probabilities and evaluation guide the system",
            "symbols and structure guide the parser",
        ],
        "classification_labels": [1, 0],
        "edit_probe": ("language", "langauge"),
    }
    return {"corpus": corpus, "probes": probes}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    corpus = fixture["corpus"]
    tokenized = [tokenize_words(text.lower()) for text in corpus]
    vocab = sorted({token for row in tokenized for token in row})
    lm = _bigram_language_model(tokenized)
    word_counts = Counter(token for row in tokenized for token in row)
    classification_texts = [tokenize_words(text.lower()) for text in fixture["probes"]["classification_texts"]]
    keyword = "probabilities"
    classifier_scores = [sum(token == keyword for token in row) for row in classification_texts]
    classification_predictions = [int(score > 0) for score in classifier_scores]
    generation_seed = tokenize_words(fixture["probes"]["generation_seed"].lower())
    last_token = generation_seed[-1]
    next_token_candidates = sorted(
        ((right, prob) for (left, right), prob in lm.items() if left == last_token),
        key=lambda item: item[1],
        reverse=True,
    )
    edit_source, edit_variant = fixture["probes"]["edit_probe"]
    return {
        "tokenized_corpus": tokenized,
        "vocab": vocab,
        "corpus_statistics": {
            "document_count": len(corpus),
            "token_count": sum(len(row) for row in tokenized),
            "vocab_size": len(vocab),
            "most_common_tokens": word_counts.most_common(6),
        },
        "pipeline_preview": {
            "raw_text": corpus[0],
            "tokens": tokenized[0],
            "language_model_seed": generation_seed,
            "next_token_candidates": next_token_candidates[:3],
        },
        "micro_tasks": {
            "classification_predictions": classification_predictions,
            "classification_scores": classifier_scores,
            "sentence_losses": [_sentence_nll(row, lm) for row in tokenized[:2]],
            "edit_distance_probe": {
                "source": edit_source,
                "variant": edit_variant,
                "distance": edit_distance(edit_source, edit_variant),
            },
        },
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    gold = fixture["probes"]["classification_labels"]
    preds = outputs["micro_tasks"]["classification_predictions"]
    tokenized = outputs["tokenized_corpus"]
    return {
        "classification_accuracy": float(sum(int(p == y) for p, y in zip(preds, gold)) / len(gold)),
        "avg_tokens_per_document": float(sum(len(row) for row in tokenized) / len(tokenized)),
        "avg_sentence_nll": float(sum(outputs["micro_tasks"]["sentence_losses"]) / len(outputs["micro_tasks"]["sentence_losses"])),
        "edit_distance_probe": outputs["micro_tasks"]["edit_distance_probe"]["distance"],
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "keyword_classifiers_do_not_understand_semantics",
            "example": "structure without probabilities",
            "note": "The introductory classifier only detects a single lexical cue and fails on semantically similar rewrites.",
        },
        {
            "case": "tiny_language_models_overfit_surface_form",
            "seed": outputs["pipeline_preview"]["language_model_seed"],
            "observed_next_tokens": outputs["pipeline_preview"]["next_token_candidates"],
            "note": "The introductory bigram model demonstrates probabilistic prediction, not robust text generation.",
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "introductory_overview",
        "counterintuitive_insight": "The best introduction is not a survey of tools but a runnable reduction of the whole field into data, representation, prediction, and evaluation loops.",
        "covered_claims": [
            "SLP can be framed as a pipeline from raw strings to measurable decisions.",
            "Tokenization, probabilistic modeling, classification, and evaluation already appear in miniature at the introductory level.",
            "A chapter can be introductory and still executable rather than purely narrative.",
        ],
        "omitted_claims": ["No speech waveform demo yet.", "No chapter-1 textbook narrative prose or historical survey."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    return build_chapter_payload(
        chapter="1",
        implementation_status="FULL",
        core_outputs={
            "corpus_statistics": outputs["corpus_statistics"],
            "pipeline_preview": outputs["pipeline_preview"],
            "micro_tasks": outputs["micro_tasks"],
        },
        metrics=evaluate(fixture, outputs),
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["repo-native introductory synthesis built to align with SLP3 chapter 1"]},
        lesson_objectives=[
            "Show the end-to-end SLP pipeline from raw text to measurable decisions.",
            "Connect tokenization, probabilistic modeling, classification, and evaluation in one runnable chapter.",
            "Make the introductory chapter operational so later chapters feel like refinements rather than isolated topics.",
        ],
        core_algorithms=["tokenization", "corpus statistics", "bigram language modeling", "keyword classification", "edit distance"],
        minimal_dataset={
            "document_count": len(fixture["corpus"]),
            "classification_probe_count": len(fixture["probes"]["classification_texts"]),
            "task_types": ["tokenization", "language modeling", "classification", "string similarity"],
        },
        reference_experiments=[
            {"name": "pipeline_walkthrough", "metric": "corpus_statistics", "expected_signal": "raw text becomes measurable structure"},
            {"name": "introductory_micro_task_suite", "metric": "classification_accuracy", "expected_signal": "simple baselines work but expose obvious limits"},
        ],
        book_vs_repo_gap="This chapter matches the repo's chapter style and makes the introduction executable, but it is still much narrower than the textbook's full conceptual, historical, and speech-inclusive introduction.",
    )


SPEC = {
    "key": "1",
    "title": "Introduction",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
