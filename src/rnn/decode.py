import numpy as np


def norm_vocab(index_to_word):
    return {int(key): value for key, value in index_to_word.items()}

def greedy_decode(
    model,
    image_feature,
    word_to_index,
    index_to_word,
    max_length,
    start_token="<start>",
    end_token="<end>",
    pad_token="<pad>",
):
    idx_to_word = norm_vocab(index_to_word)
    start_id = int(word_to_index[start_token])
    end_id = int(word_to_index[end_token])
    pad_id = int(word_to_index.get(pad_token, 0))

    feature = np.asarray(image_feature, dtype="float32")
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)

    sequence = np.full((1, max_length), pad_id, dtype="int32")
    sequence[0, 0] = start_id
    tokens = []

    for step in range(max_length):
        probs = model.predict([feature, sequence], verbose=0)
        next_id = int(np.argmax(probs[0, step]))

        if next_id == end_id:
            break

        tokens.append(idx_to_word.get(next_id, "<unk>"))

        if step + 1 < max_length:
            sequence[0, step + 1] = next_id

    return tokens

def decode_batch(model, image_features, word_to_index, index_to_word, max_length):
    predictions = []
    for feature in np.asarray(image_features):
        tokens = greedy_decode(model, feature, word_to_index, index_to_word, max_length)
        predictions.append(tokens)
    return predictions
