import numpy as np

from common.metrics import score_set, timer

from .decode import decode_batch, norm_vocab

def seq_tokens(sequence, index_to_word, start_token="<start>", end_token="<end>", pad_token="<pad>"):
    idx_to_word = norm_vocab(index_to_word)
    tokens = []

    for index in np.asarray(sequence).astype(int).tolist():
        token = idx_to_word.get(index, "<unk>")
        if token in (start_token, pad_token):
            continue
        if token == end_token:
            break
        tokens.append(token)

    return tokens

def eval_keras(model, image_features, ref_sequences, word_to_index, index_to_word, max_length):
    references = []
    for sequence in np.asarray(ref_sequences):
        references.append([seq_tokens(sequence, index_to_word)])

    with timer() as elapsed:
        predictions = decode_batch(
            model,
            image_features,
            word_to_index,
            index_to_word,
            max_length,
        )

    scores = score_set(references, predictions)
    scores["runtime_seconds"] = float(elapsed["elapsed_seconds"])
    scores["samples"] = int(len(predictions))
    return scores

def hist_sum(history):
    raw_history = getattr(history, "history", history)
    result = {}
    for key, values in raw_history.items():
        result[key] = [float(value) for value in values]
    return result
