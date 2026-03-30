from __future__ import annotations

from collections import Counter
import unicodedata

from slp3_from_sutskever30.chapter_contract import build_chapter_payload
from slp3_from_sutskever30.classical import edit_distance, tokenize_words


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().strip()


def _whitespace_tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _train_bpe(corpus: list[str], *, vocab_size: int = 48) -> list[tuple[str, str]]:
    words = [list(token) + ["</w>"] for text in corpus for token in tokenize_words(_normalize(text))]
    merges: list[tuple[str, str]] = []
    while len({piece for word in words for piece in word}) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word in words:
            for idx in range(len(word) - 1):
                pair_counts[(word[idx], word[idx + 1])] += 1
        if not pair_counts:
            break
        pair, count = pair_counts.most_common(1)[0]
        if count < 2:
            break
        merges.append(pair)
        merged_symbol = "".join(pair)
        next_words: list[list[str]] = []
        for word in words:
            merged_word: list[str] = []
            idx = 0
            while idx < len(word):
                if idx < len(word) - 1 and (word[idx], word[idx + 1]) == pair:
                    merged_word.append(merged_symbol)
                    idx += 2
                else:
                    merged_word.append(word[idx])
                    idx += 1
            next_words.append(merged_word)
        words = next_words
    return merges


def _apply_bpe(token: str, merges: list[tuple[str, str]]) -> list[str]:
    pieces = list(token) + ["</w>"]
    for pair in merges:
        merged_symbol = "".join(pair)
        idx = 0
        next_pieces: list[str] = []
        while idx < len(pieces):
            if idx < len(pieces) - 1 and (pieces[idx], pieces[idx + 1]) == pair:
                next_pieces.append(merged_symbol)
                idx += 2
            else:
                next_pieces.append(pieces[idx])
                idx += 1
        pieces = next_pieces
    if pieces and pieces[-1] == "</w>":
        pieces = pieces[:-1]
    elif pieces and pieces[-1].endswith("</w>"):
        pieces[-1] = pieces[-1].removesuffix("</w>")
    return [piece for piece in pieces if piece]


def _tokenize_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    return [piece for token in tokenize_words(_normalize(text)) for piece in _apply_bpe(token, merges)]


def _oov_rate(train_tokens: list[list[str]], eval_tokens: list[list[str]]) -> float:
    vocab = {token for row in train_tokens for token in row}
    flat = [token for row in eval_tokens for token in row]
    if not flat:
        return 0.0
    return float(sum(token not in vocab for token in flat) / len(flat))


def _token_diff(a: list[str], b: list[str]) -> int:
    width = max(len(a), len(b))
    a_pad = a + ["<pad>"] * (width - len(a))
    b_pad = b + ["<pad>"] * (width - len(b))
    return sum(x != y for x, y in zip(a_pad, b_pad))


def build_fixture() -> dict[str, object]:
    train_texts = [
        "Can't stop, won't stop processing language.",
        "Café tokens should normalize cleanly.",
        "Email-like text: nlp@example.com should split predictably.",
        "The tokenizer should keep punctuation localized.",
    ]
    eval_texts = [
        "Cafe tokens normalize even without accents.",
        "Can't-stop errors should stay local.",
    ]
    typo_probe = ("tokenization", "tokeniztaion")
    return {"train_texts": train_texts, "eval_texts": eval_texts, "typo_probe": typo_probe}


def run_numpy(fixture: dict[str, object]) -> dict[str, object]:
    train_texts = fixture["train_texts"]
    eval_texts = fixture["eval_texts"]
    typo_source, typo_variant = fixture["typo_probe"]
    merges = _train_bpe(train_texts, vocab_size=48)
    whitespace_train = [_whitespace_tokenize(text) for text in train_texts]
    whitespace_eval = [_whitespace_tokenize(text) for text in eval_texts]
    regex_train = [tokenize_words(_normalize(text)) for text in train_texts]
    regex_eval = [tokenize_words(_normalize(text)) for text in eval_texts]
    bpe_train = [_tokenize_bpe(text, merges) for text in train_texts]
    bpe_eval = [_tokenize_bpe(text, merges) for text in eval_texts]
    return {
        "normalization_preview": [_normalize(text) for text in train_texts[:2]],
        "tokenizers": {
            "whitespace": whitespace_eval[0],
            "regex": regex_eval[0],
            "bpe": bpe_eval[0],
        },
        "bpe_merge_count": len(merges),
        "merge_preview": [list(pair) for pair in merges[:10]],
        "error_locality": {
            "regex_changed_positions": _token_diff(tokenize_words(typo_source), tokenize_words(typo_variant)),
            "bpe_changed_positions": _token_diff(_tokenize_bpe(typo_source, merges), _tokenize_bpe(typo_variant, merges)),
            "edit_distance": edit_distance(typo_source, typo_variant),
        },
        "train_tokens": {"whitespace": whitespace_train, "regex": regex_train, "bpe": bpe_train},
        "eval_tokens": {"whitespace": whitespace_eval, "regex": regex_eval, "bpe": bpe_eval},
    }


def evaluate(fixture: dict[str, object], outputs: dict[str, object]) -> dict[str, object]:
    whitespace_train = outputs["train_tokens"]["whitespace"]
    whitespace_eval = outputs["eval_tokens"]["whitespace"]
    regex_train = outputs["train_tokens"]["regex"]
    regex_eval = outputs["eval_tokens"]["regex"]
    bpe_train = outputs["train_tokens"]["bpe"]
    bpe_eval = outputs["eval_tokens"]["bpe"]
    return {
        "vocab_sizes": {
            "whitespace": len({token for row in whitespace_train for token in row}),
            "regex": len({token for row in regex_train for token in row}),
            "bpe": len({token for row in bpe_train for token in row}),
        },
        "oov_rates": {
            "whitespace": _oov_rate(whitespace_train, whitespace_eval),
            "regex": _oov_rate(regex_train, regex_eval),
            "bpe": _oov_rate(bpe_train, bpe_eval),
        },
        "avg_tokens_per_eval_text": {
            "whitespace": float(sum(len(row) for row in whitespace_eval) / len(whitespace_eval)),
            "regex": float(sum(len(row) for row in regex_eval) / len(regex_eval)),
            "bpe": float(sum(len(row) for row in bpe_eval) / len(bpe_eval)),
        },
    }


def failure_cases(fixture: dict[str, object], outputs: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case": "hyphenation_smears_boundaries",
            "example": "can't-stop",
            "observed_regex": tokenize_words(_normalize("can't-stop")),
            "observed_bpe": _tokenize_bpe(_normalize("can't-stop"), _train_bpe(fixture["train_texts"])),
        },
        {
            "case": "email_tokenization_is_not_semantic",
            "example": "nlp@example.com",
            "observed_regex": tokenize_words(_normalize("nlp@example.com")),
        },
    ]


def chapter_notes() -> dict[str, object]:
    return {
        "batch": "batch_1_classical_foundations",
        "counterintuitive_insight": "Error locality matters more than vocabulary size: a tokenizer is better when one typo corrupts fewer positions.",
        "covered_claims": [
            "Normalization policy changes token statistics.",
            "Tokenizers can be compared on OOV rate and error locality.",
            "A learned BPE-lite tokenizer gives a different compression profile than regex tokenization.",
        ],
        "omitted_claims": ["No unigram LM tokenizer yet.", "No large-corpus compression benchmark yet."],
    }


def run_chapter() -> dict[str, object]:
    fixture = build_fixture()
    outputs = run_numpy(fixture)
    metrics = evaluate(fixture, outputs)
    return build_chapter_payload(
        chapter="2",
        implementation_status="FULL",
        core_outputs={
            "normalization_preview": outputs["normalization_preview"],
            "tokenizers": outputs["tokenizers"],
            "bpe_merge_count": outputs["bpe_merge_count"],
            "merge_preview": outputs["merge_preview"],
            "error_locality": outputs["error_locality"],
        },
        metrics=metrics,
        failure_modes=failure_cases(fixture, outputs),
        chapter_notes=chapter_notes(),
        sources={"source_papers": [], "derivation_lineage": ["pageman/sutskever-30-implementations", "pageman/sutskever-30-beyond-numpy"]},
    )


SPEC = {
    "key": "2",
    "title": "Words and Tokens",
    "implementation_status": "FULL",
    "source_papers": (),
    "runner": run_chapter,
}
