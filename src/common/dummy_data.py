from pathlib import Path

import numpy as np

from .io import save_json, save_npy


TOKENS = [
    "<pad>",
    "<start>",
    "<end>",
    "a",
    "man",
    "woman",
    "child",
    "dog",
    "cat",
    "runs",
    "sits",
    "plays",
    "on",
    "with",
    "grass",
    "street",
]


def build_vocab(tokens=None):
    token_list = list(tokens or TOKENS)
    word_to_index = {}
    index_to_word = {}

    for index, word in enumerate(token_list):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word


def dummy_feats(num_samples=32, feature_dim=2048, seed=42):
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(num_samples, feature_dim)).astype("float32")
    image_ids = np.array(["dummy_image_{0:04d}.jpg".format(i) for i in range(num_samples)])
    return features, image_ids


def dummy_caps(num_samples=32, max_length=12, vocab_size=None, seed=42):
    if max_length < 3:
        raise ValueError("max_length must be at least 3.")

    token_count = int(vocab_size or len(TOKENS))
    if token_count <= 3:
        raise ValueError("vocab_size must include at least one content token.")

    rng = np.random.default_rng(seed)
    sequences = np.zeros((num_samples, max_length), dtype="int32")
    start_id = 1
    end_id = 2

    for row in range(num_samples):
        content_len = int(rng.integers(1, max_length - 1))
        content = rng.integers(3, token_count, size=content_len, dtype="int32")
        sequence = np.concatenate(([start_id], content, [end_id])).astype("int32")
        limit = min(sequence.shape[0], max_length)
        sequences[row, :limit] = sequence[:limit]
        if sequence.shape[0] > max_length:
            sequences[row, max_length - 1] = end_id

    return sequences


def teacher_pairs(caption_sequences):
    sequences = np.asarray(caption_sequences, dtype="int32")
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError("caption_sequences must have shape (n, length >= 2).")
    return sequences[:, :-1], sequences[:, 1:]


def save_dummy(base_dir=".", num_samples=32, feature_dim=2048, max_length=12, seed=42):
    base_path = Path(base_dir)
    word_to_index, index_to_word = build_vocab()
    features, image_ids = dummy_feats(num_samples, feature_dim, seed)
    sequences = dummy_caps(num_samples, max_length, len(word_to_index), seed)

    return {
        "features": save_npy(features, base_path / "data" / "features" / "dummy_flickr8k_features.npy"),
        "image_ids": save_npy(image_ids, base_path / "data" / "features" / "dummy_image_ids.npy"),
        "caption_sequences": save_npy(sequences, base_path / "data" / "vocab" / "dummy_caption_sequences.npy"),
        "word_to_index": save_json(word_to_index, base_path / "data" / "vocab" / "dummy_word_to_index.json"),
        "index_to_word": save_json(
            {str(key): value for key, value in index_to_word.items()},
            base_path / "data" / "vocab" / "dummy_index_to_word.json",
        ),
    }
