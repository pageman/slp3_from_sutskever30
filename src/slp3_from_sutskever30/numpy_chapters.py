from __future__ import annotations

import numpy as np

from slp3_from_sutskever30.classical import (
    build_demo_speller,
    build_ppmi_embeddings,
    correct_token,
    edit_distance,
    kneser_ney_next_distribution,
    predict_next_distribution,
    tokenize_words,
    train_hmm_tagger,
    train_logistic_regression_text,
    train_mlp_text_classifier,
    train_naive_bayes_text,
    train_ngram_language_model,
    viterbi_decode,
)
from slp3_from_sutskever30.common import cross_entropy_from_probs, one_hot, seeded_rng, stable_softmax


def chapter2_words_and_tokens() -> dict[str, object]:
    text = "Speech and language processing with language models."
    tokens = tokenize_words(text)
    return {"chapter": "2", "tokens": tokens, "vocab_size": len(set(tokens)), "edit_distance": edit_distance("language", "linguistics")}


def chapter3_ngram_language_models() -> dict[str, object]:
    corpus = ["language models predict words", "language models predict tokens", "speech models predict words"]
    model = train_ngram_language_model(corpus, order=2)
    distribution = predict_next_distribution(model, ["language"])
    top = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:3]
    return {"chapter": "3", "top_predictions": top, "vocab_size": len(model.vocab)}


def chapter4_logistic_regression() -> dict[str, object]:
    texts = ["great movie wonderful acting", "excellent plot and great cast", "boring movie dull script", "bad acting and boring scenes"]
    labels = ["pos", "pos", "neg", "neg"]
    model, losses = train_logistic_regression_text(texts, labels)
    return {"chapter": "4", "initial_loss": losses[0], "final_loss": losses[-1], "prediction": model.predict("great acting")}


def chapter5_embeddings() -> dict[str, object]:
    vocab, ppmi, emb = build_ppmi_embeddings(["language models use context", "speech systems use context", "language systems use embeddings"])
    return {"chapter": "5", "vocab_size": len(vocab), "ppmi_shape": tuple(ppmi.shape), "embedding_shape": tuple(emb.shape)}


def chapter6_neural_networks() -> dict[str, object]:
    texts = ["great movie wonderful acting", "excellent story great dialogue", "terrible movie bad acting", "awful script terrible pacing"]
    labels = ["pos", "pos", "neg", "neg"]
    model, losses = train_mlp_text_classifier(texts, labels)
    return {"chapter": "6", "initial_loss": losses[0], "final_loss": losses[-1], "prediction": model.predict("great dialogue")}


def chapter7_large_language_models() -> dict[str, object]:
    rng = seeded_rng(27)
    contexts = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float64)
    targets = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    w_hidden = rng.normal(scale=0.18, size=(5, 3))
    b_hidden = np.zeros((5,), dtype=np.float64)
    w_out = rng.normal(scale=0.18, size=(2, 4, 5))
    b_out = np.zeros((2, 4), dtype=np.float64)
    hidden = np.tanh(contexts @ w_hidden.T + b_hidden)
    logits = np.einsum("tvh,bh->btv", w_out, hidden) + b_out[None, :, :]
    probs = stable_softmax(logits, axis=2)
    losses = [cross_entropy_from_probs(probs[:, idx, :], targets[:, idx]) for idx in range(targets.shape[1])]
    return {"chapter": "7", "loss": float(np.mean(losses)), "logits_shape": tuple(logits.shape)}


def chapter8_transformers() -> dict[str, object]:
    rng = seeded_rng(13)
    seqs = rng.normal(size=(4, 3, 4))
    targets = np.asarray([0, 1, 2, 1], dtype=np.int64)
    wq = rng.normal(scale=0.2, size=(4, 4))
    wk = rng.normal(scale=0.2, size=(4, 4))
    wv = rng.normal(scale=0.2, size=(4, 4))
    wo = rng.normal(scale=0.2, size=(3, 4))
    q = seqs @ wq.T
    k = seqs @ wk.T
    v = seqs @ wv.T
    scores = np.einsum("btd,bsd->bts", q, k) / np.sqrt(q.shape[-1])
    weights = stable_softmax(scores, axis=2)
    pooled = np.mean(weights @ v, axis=1)
    logits = pooled @ wo.T
    probs = stable_softmax(logits, axis=1)
    return {"chapter": "8", "loss": cross_entropy_from_probs(probs, targets), "logits_shape": tuple(logits.shape), "attention_shape": tuple(weights.shape)}


def chapter9_post_training() -> dict[str, object]:
    rng = seeded_rng(9)
    prompts = rng.normal(size=(4, 8))
    chosen = rng.normal(size=(4, 10))
    rejected = rng.normal(size=(4, 10))
    w = rng.normal(scale=0.15, size=(10, 8))
    logits = prompts @ w.T
    sft_probs = stable_softmax(logits, axis=1)
    sft_targets = np.asarray([0, 1, 2, 3], dtype=np.int64)
    sft_loss = cross_entropy_from_probs(sft_probs, sft_targets)
    chosen_scores = np.sum(chosen * logits, axis=1)
    rejected_scores = np.sum(rejected * logits, axis=1)
    pref_margin = chosen_scores - rejected_scores
    preference_loss = float(np.mean(np.log1p(np.exp(-pref_margin))))
    reranked = np.argsort(-(chosen_scores[:, None] + np.asarray([0.2, 0.1, 0.0])), axis=1)
    return {"chapter": "9", "sft_loss": sft_loss, "preference_loss": preference_loss, "search_trace_shape": tuple(reranked.shape)}


def chapter10_masked_language_models() -> dict[str, object]:
    rng = seeded_rng(10)
    vocab_size = 7
    token_ids = np.asarray([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1], [3, 4, 5, 6, 1, 2], [4, 5, 6, 1, 2, 3]], dtype=np.int64)
    mask_positions = np.asarray([[1, 4], [0, 3], [2, 5], [1, 2]], dtype=np.int64)
    embeddings = rng.normal(scale=0.2, size=(vocab_size, 5))
    out = rng.normal(scale=0.2, size=(vocab_size, 5))
    logits = []
    targets = []
    for row, positions in zip(token_ids, mask_positions):
        row_emb = embeddings[row]
        for pos in positions:
            context = np.mean(np.delete(row_emb, pos, axis=0), axis=0)
            logits.append(out @ context)
            targets.append(int(row[pos]))
    logits_arr = np.stack(logits)
    probs = stable_softmax(logits_arr, axis=1)
    return {"chapter": "10", "loss": cross_entropy_from_probs(probs, np.asarray(targets, dtype=np.int64)), "masked_logits_shape": tuple(logits_arr.shape)}


def chapter11_ir_and_rag() -> dict[str, object]:
    rng = seeded_rng(11)
    queries = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=np.float64)
    passages = np.asarray([[1.0, 0.0, 0.1], [0.0, 1.0, 0.1], [0.8, 0.2, 0.1], [0.2, 0.8, 0.1]], dtype=np.float64)
    targets = np.arange(4, dtype=np.int64)
    wq = rng.normal(scale=0.18, size=(4, 3))
    wp = rng.normal(scale=0.18, size=(4, 3))
    q = np.tanh(queries @ wq.T)
    p = np.tanh(passages @ wp.T)
    scores = q @ p.T
    dpr_loss = cross_entropy_from_probs(stable_softmax(scores, axis=1), targets)
    docs = np.stack([passages[[0, 1, 2]], passages[[1, 0, 2]], passages[[0, 2, 1]], passages[[1, 2, 0]]])
    p_docs = np.tanh(np.einsum("bdm,hm->bdh", docs, wp))
    retr = stable_softmax(np.einsum("bh,bdh->bd", q, p_docs), axis=1)
    wout = rng.normal(scale=0.18, size=(3, 8))
    combined = np.concatenate([np.repeat(q[:, None, :], 3, axis=1), p_docs], axis=2)
    doc_logits = np.einsum("vh,bdh->bdv", wout, combined)
    mixture = np.sum(retr[:, :, None] * stable_softmax(doc_logits, axis=2), axis=1)
    rag_targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return {"chapter": "11", "dpr_loss": dpr_loss, "rag_loss": cross_entropy_from_probs(mixture, rag_targets), "rag_probs_shape": tuple(mixture.shape)}


def chapter12_machine_translation() -> dict[str, object]:
    rng = seeded_rng(12)
    src = np.asarray([[0, 1, 2, 1], [1, 2, 0, 2], [2, 1, 1, 0], [0, 2, 1, 2]], dtype=np.int64)
    decoder = np.asarray([1, 0, 2, 1], dtype=np.int64)
    targets = np.asarray([2, 1, 0, 2], dtype=np.int64)
    emb = rng.normal(scale=0.2, size=(3, 5))
    dec_emb = emb[decoder]
    enc = emb[src]
    wa = rng.normal(scale=0.2, size=(5, 5))
    attention_scores = np.einsum("bth,bh->bt", enc @ wa.T, dec_emb)
    attention = stable_softmax(attention_scores, axis=1)
    context = np.einsum("bt,bth->bh", attention, enc)
    wout = rng.normal(scale=0.2, size=(3, 10))
    logits = np.concatenate([context, dec_emb], axis=1) @ wout.T
    probs = stable_softmax(logits, axis=1)
    return {"chapter": "12", "loss": cross_entropy_from_probs(probs, targets), "attention_shape": tuple(attention.shape)}


def chapter13_rnns_and_lstms() -> dict[str, object]:
    rng = seeded_rng(13)
    vocab = 5
    inputs = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    targets = np.asarray([1, 2, 1, 0, 1], dtype=np.int64)
    xs = one_hot(inputs, vocab)
    wxh = rng.normal(scale=0.2, size=(6, vocab))
    whh = rng.normal(scale=0.2, size=(6, 6))
    why = rng.normal(scale=0.2, size=(vocab, 6))
    h = np.zeros((6,), dtype=np.float64)
    rnn_logits = []
    for x_t in xs:
        h = np.tanh(wxh @ x_t + whh @ h)
        rnn_logits.append(why @ h)
    rnn_probs = stable_softmax(np.stack(rnn_logits), axis=1)
    rnn_loss = cross_entropy_from_probs(rnn_probs, targets)
    wf = rng.normal(scale=0.2, size=(6, vocab + 6))
    wi = rng.normal(scale=0.2, size=(6, vocab + 6))
    wo = rng.normal(scale=0.2, size=(6, vocab + 6))
    wc = rng.normal(scale=0.2, size=(6, vocab + 6))
    h = np.zeros((6,), dtype=np.float64)
    c = np.zeros((6,), dtype=np.float64)
    lstm_logits = []
    for x_t in xs:
        concat = np.concatenate([x_t, h])
        f = 1.0 / (1.0 + np.exp(-(wf @ concat)))
        i = 1.0 / (1.0 + np.exp(-(wi @ concat)))
        o = 1.0 / (1.0 + np.exp(-(wo @ concat)))
        g = np.tanh(wc @ concat)
        c = f * c + i * g
        h = o * np.tanh(c)
        lstm_logits.append(why @ h)
    lstm_probs = stable_softmax(np.stack(lstm_logits), axis=1)
    return {"chapter": "13", "rnn_loss": rnn_loss, "lstm_loss": cross_entropy_from_probs(lstm_probs, targets)}


def chapter14_phonetics_and_features() -> dict[str, object]:
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    waves = np.stack([np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * (freq * 2.0) * t) for freq in [120, 180, 240, 300]])
    frames = waves[:, :15680].reshape(4, 98, 160)
    spectrum = np.abs(np.fft.rfft(frames, axis=2))
    mel = np.log1p(spectrum[:, :, :80])
    phonetic_labels = np.argmax(np.mean(mel, axis=1)[:, :8], axis=1)
    return {"chapter": "14", "log_mel_shape": tuple(mel.shape), "phonetic_labels": phonetic_labels.tolist()}


def chapter15_asr() -> dict[str, object]:
    rng = seeded_rng(15)
    features = np.asarray([[[1.0, 0.0], [0.3, 0.9]], [[0.8, 0.2], [0.1, 1.0]]], dtype=np.float64)
    targets = np.asarray([1, 1], dtype=np.int64)
    w = rng.normal(scale=0.25, size=(3, 2))
    logits = features @ w.T
    probs = stable_softmax(logits, axis=2)
    losses = []
    for sample in probs:
        prob = sample[0, 0] * sample[1, 1] + sample[0, 1] * sample[1, 0] + sample[0, 1] * sample[1, 1]
        losses.append(-np.log(prob + 1e-12))
    return {"chapter": "15", "loss": float(np.mean(losses)), "logits_shape": tuple(logits.shape), "targets": targets.tolist()}


def chapter16_tts() -> dict[str, object]:
    rng = seeded_rng(16)
    phoneme_ids = np.asarray([[1, 2, 3, 2], [2, 3, 1, 3], [3, 2, 2, 1], [1, 3, 2, 3]], dtype=np.int64)
    durations = np.asarray([[3, 2, 4, 1], [2, 3, 2, 3], [4, 2, 2, 2], [2, 4, 2, 2]], dtype=np.int64)
    emb = rng.normal(scale=0.2, size=(4, 6))
    mel_bank = rng.normal(scale=0.2, size=(80, 6))
    mel_frames = []
    stop_logits = []
    for seq, dur in zip(phoneme_ids, durations):
        expanded = np.repeat(emb[seq], dur, axis=0)
        mel = expanded @ mel_bank.T
        mel_frames.append(mel)
        stop_logits.append(np.linspace(-1.0, 1.0, mel.shape[0]))
    max_len = max(frame.shape[0] for frame in mel_frames)
    padded = np.zeros((4, max_len, 80), dtype=np.float64)
    stop = np.full((4, max_len), -10.0, dtype=np.float64)
    for idx, frame in enumerate(mel_frames):
        padded[idx, : frame.shape[0], :] = frame
        stop[idx, : frame.shape[0]] = stop_logits[idx]
    return {"chapter": "16", "mel_frames_shape": tuple(padded.shape), "stop_logits_shape": tuple(stop.shape)}


def chapter17_sequence_labeling() -> dict[str, object]:
    rng = seeded_rng(17)
    token_ids = np.asarray([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]], dtype=np.int64)
    char_ids = rng.integers(0, 6, size=(4, 4, 5))
    token_emb = rng.normal(scale=0.2, size=(5, 6))
    char_emb = rng.normal(scale=0.2, size=(6, 3))
    classifier = rng.normal(scale=0.2, size=(7, 9))
    features = np.concatenate([token_emb[token_ids], np.mean(char_emb[char_ids], axis=2)], axis=2)
    logits = np.einsum("cf,btf->btc", classifier, features)
    decoded = np.argmax(logits, axis=2)
    return {"chapter": "17", "tag_logits_shape": tuple(logits.shape), "decoded_tags_shape": tuple(decoded.shape)}


def chapter18_constituency_parsing() -> dict[str, object]:
    rng = seeded_rng(18)
    chart = np.zeros((4, 8, 8, 6), dtype=np.float64)
    chart[:, np.arange(8), np.arange(8), :] = rng.normal(scale=0.1, size=(4, 8, 6))
    for span in range(2, 9):
        for left in range(0, 9 - span):
            right = left + span - 1
            candidates = [
                np.max(chart[:, left, split, :], axis=1) + np.max(chart[:, split + 1, right, :], axis=1)
                for split in range(left, right)
            ]
            best_split_score = np.max(np.stack(candidates, axis=1), axis=1)
            chart[:, left, right, 0] = best_split_score
    return {"chapter": "18", "chart_scores_shape": tuple(chart.shape), "root_scores": chart[:, 0, 7, 0].tolist()}


def chapter19_dependency_parsing() -> dict[str, object]:
    rng = seeded_rng(19)
    token_emb = rng.normal(scale=0.2, size=(4, 10, 6))
    wh = rng.normal(scale=0.2, size=(6, 6))
    wd = rng.normal(scale=0.2, size=(6, 6))
    heads = token_emb @ wh.T
    deps = token_emb @ wd.T
    scores = np.einsum("bih,bjh->bij", heads, deps)
    np.fill_diagonal(scores[0], -1e9)
    predicted_heads = np.argmax(scores, axis=1)
    return {"chapter": "19", "arc_scores_shape": tuple(scores.shape), "head_indices_shape": tuple(predicted_heads.shape)}


def chapter20_information_extraction() -> dict[str, object]:
    rng = seeded_rng(20)
    token_emb = rng.normal(scale=0.2, size=(4, 14, 8))
    spans = np.asarray([[[0, 1], [2, 3], [4, 5]], [[1, 2], [3, 4], [5, 6]], [[0, 2], [4, 6], [7, 8]], [[2, 4], [5, 7], [8, 10]]], dtype=np.int64)
    span_repr = []
    for batch_idx in range(spans.shape[0]):
        rows = []
        for start, end in spans[batch_idx]:
            rows.append(np.mean(token_emb[batch_idx, start : end + 1], axis=0))
        span_repr.append(rows)
    span_repr_arr = np.asarray(span_repr)
    entity_head = rng.normal(scale=0.2, size=(4, 8))
    relation_head = rng.normal(scale=0.2, size=(5, 16))
    entity_logits = np.einsum("cf,bsf->bsc", entity_head, span_repr_arr)
    pair_repr = np.concatenate([span_repr_arr[:, :, None, :].repeat(3, axis=2), span_repr_arr[:, None, :, :].repeat(3, axis=1)], axis=3)
    relation_logits = np.einsum("cf,bsrf->bsrc", relation_head, pair_repr)
    return {"chapter": "20", "entity_logits_shape": tuple(entity_logits.shape), "relation_logits_shape": tuple(relation_logits.shape)}


def chapter21_semantic_role_labeling() -> dict[str, object]:
    rng = seeded_rng(21)
    token_emb = rng.normal(scale=0.2, size=(4, 12, 8))
    predicate_indices = np.asarray([[1, 5], [2, 7], [3, 8], [4, 9]], dtype=np.int64)
    predicate_repr = token_emb[np.arange(4)[:, None], predicate_indices]
    span_repr = np.stack([token_emb[:, idx : idx + 2, :].mean(axis=1) for idx in range(8)], axis=1)
    features = np.concatenate([predicate_repr[:, :, None, :].repeat(8, axis=2), span_repr[:, None, :, :].repeat(2, axis=1)], axis=3)
    role_head = rng.normal(scale=0.2, size=(6, 16))
    role_logits = np.einsum("cf,bpsf->bpsc", role_head, features)
    return {"chapter": "21", "role_logits_shape": tuple(role_logits.shape)}


def chapter22_lexicons_sentiment_affect() -> dict[str, object]:
    lexicon = {"good": np.asarray([1.0, 0.2, 0.1]), "bad": np.asarray([-1.0, 0.7, 0.8]), "calm": np.asarray([0.2, -0.5, -0.4]), "angry": np.asarray([-0.8, 0.9, 0.9])}
    docs = ["good calm movie", "bad angry script", "good good calm", "bad bad angry"]
    token_scores = []
    doc_scores = []
    for doc in docs:
        scores = np.stack([lexicon.get(token, np.zeros((3,), dtype=np.float64)) for token in tokenize_words(doc)])
        token_scores.append(scores)
        doc_scores.append(np.mean(scores, axis=0))
    padded = np.zeros((4, max(score.shape[0] for score in token_scores), 3), dtype=np.float64)
    for idx, score in enumerate(token_scores):
        padded[idx, : score.shape[0], :] = score
    return {"chapter": "22", "token_scores_shape": tuple(padded.shape), "document_scores_shape": tuple(np.asarray(doc_scores).shape)}


def chapter23_coreference_and_entity_linking() -> dict[str, object]:
    rng = seeded_rng(23)
    mention_emb = rng.normal(scale=0.2, size=(4, 6, 16))
    coref_scores = np.einsum("bmd,bnd->bmn", mention_emb, mention_emb)
    kb = rng.normal(scale=0.2, size=(5, 16))
    link_scores = np.einsum("bmd,kd->bmk", mention_emb, kb)
    cluster_ids = np.argmax(coref_scores, axis=2)
    return {"chapter": "23", "coref_scores_shape": tuple(coref_scores.shape), "link_scores_shape": tuple(link_scores.shape), "cluster_ids_shape": tuple(cluster_ids.shape)}


def chapter24_discourse_coherence() -> dict[str, object]:
    rng = seeded_rng(24)
    sent_emb = rng.normal(scale=0.2, size=(4, 6, 16))
    entity_grid = rng.integers(0, 3, size=(4, 6, 8))
    pair_scores = np.einsum("bid,bjd->bij", sent_emb, sent_emb)
    transition_bonus = np.mean(entity_grid[:, 1:, :] == entity_grid[:, :-1, :], axis=(1, 2))
    coherence = np.mean(pair_scores, axis=(1, 2)) + transition_bonus
    ordering_logits = np.stack([coherence, -coherence], axis=1)
    return {"chapter": "24", "coherence_scores_shape": tuple(coherence.shape), "ordering_logits_shape": tuple(ordering_logits.shape)}


def chapter25_conversation_structure() -> dict[str, object]:
    rng = seeded_rng(25)
    turns = rng.integers(0, 10, size=(4, 5, 12))
    speakers = np.asarray([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [1, 1, 0, 0, 1]], dtype=np.int64)
    emb = rng.normal(scale=0.2, size=(10, 6))
    speaker_emb = rng.normal(scale=0.2, size=(2, 2))
    features = np.concatenate([np.mean(emb[turns], axis=2), speaker_emb[speakers]], axis=2)
    head = rng.normal(scale=0.2, size=(8, 8))
    logits = np.einsum("cf,btf->btc", head, features)
    state_summary = np.mean(features, axis=1)
    return {"chapter": "25", "dialogue_act_logits_shape": tuple(logits.shape), "state_summary_shape": tuple(state_summary.shape)}


def chapterA_hidden_markov_models() -> dict[str, object]:
    tagged = [[("time", "NOUN"), ("flies", "VERB")], [("fruit", "NOUN"), ("flies", "NOUN")], [("time", "NOUN"), ("runs", "VERB")]]
    model = train_hmm_tagger(tagged)
    return {"chapter": "A", "states": list(model.states), "tags": viterbi_decode(model, ["time", "runs"])}


def chapterB_naive_bayes() -> dict[str, object]:
    texts = ["good fun comedy", "great fun movie", "bad dull drama", "boring slow movie"]
    labels = ["pos", "pos", "neg", "neg"]
    model = train_naive_bayes_text(texts, labels)
    return {"chapter": "B", "prediction": model.predict("good movie"), "vocab_size": len(model.vocab)}


def chapterC_kneser_ney() -> dict[str, object]:
    corpus = ["language models predict words", "language models predict tokens", "speech models predict words"]
    distribution = kneser_ney_next_distribution(corpus, ["language"])
    top = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:3]
    return {"chapter": "C", "top_predictions": top, "support_size": len(distribution)}


def chapterD_spelling_correction() -> dict[str, object]:
    corpus = ["language models process speech", "language systems process text", "speech recognition uses language models"]
    speller = build_demo_speller(corpus)
    return {"chapter": "D", "correction": correct_token("langauge", speller)}
