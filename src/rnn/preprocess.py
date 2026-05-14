import re
from pathlib import Path

import numpy as np

from common.io import ensure_dir, load_json, save_json, save_npy

PUNCT = re.compile(r"[^a-z0-9\s]")

def clean_text(text):
    text = str(text).lower()
    text = PUNCT.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    text = clean_text(text)
    if not text:
        return []
    return text.split()

def make_vocab(captions, special=None, min_freq=1):
    special = list(special or ["<pad>", "<start>", "<end>", "<unk>"])
    freq = {}

    for caption in captions:
        for token in tokenize(caption):
            freq[token] = freq.get(token, 0) + 1

    words = [word for word, count in sorted(freq.items()) if count >= min_freq]
    word_to_index = {}
    index_to_word = {}

    index = 0
    for token in special:
        if token not in word_to_index:
            word_to_index[token] = index
            index_to_word[index] = token
            index += 1

    for word in words:
        if word not in word_to_index:
            word_to_index[word] = index
            index_to_word[index] = word
            index += 1

    return word_to_index, index_to_word

def pad_seq(sequence, max_length, pad_id=0):
    arr = np.asarray(sequence, dtype="int32")
    if arr.shape[0] >= max_length:
        return arr[:max_length]
    pad_width = max_length - arr.shape[0]
    return np.pad(arr, (0, pad_width), mode="constant", constant_values=pad_id)

def encode_caps(captions, max_length, special=None, min_freq=1):
    word_to_index, index_to_word = make_vocab(captions, special=special, min_freq=min_freq)
    pad_id = int(word_to_index.get("<pad>", 0))
    start_id = int(word_to_index.get("<start>", 1))
    end_id = int(word_to_index.get("<end>", 2))
    unk_id = int(word_to_index.get("<unk>", pad_id))

    sequences = []
    for caption in captions:
        tokens = tokenize(caption)
        ids = [start_id]
        for token in tokens:
            ids.append(int(word_to_index.get(token, unk_id)))
        ids.append(end_id)
        sequences.append(pad_seq(ids, max_length, pad_id=pad_id))

    return np.asarray(sequences, dtype="int32"), word_to_index, index_to_word

def make_index(word_to_index):
    return {int(index): word for word, index in word_to_index.items()}

def save_vocab(word_to_index, index_to_word, out_dir):
    out_path = Path(out_dir)
    ensure_dir(out_path)
    save_json(word_to_index, out_path / "vocab.json")
    return out_path

def load_vocab(out_dir):
    out_path = Path(out_dir)
    word_to_index = load_json(out_path / "vocab.json")
    index_to_word = make_index(word_to_index)
    return word_to_index, index_to_word

def prep_data(captions, max_length, out_dir=None, special=None, min_freq=1):
    sequences, word_to_index, index_to_word = encode_caps(
        captions,
        max_length=max_length,
        special=special,
        min_freq=min_freq,
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        ensure_dir(out_path)
        save_npy(sequences, out_path / "caption_sequences.npy")
        save_vocab(word_to_index, index_to_word, out_path)

    return sequences, word_to_index, index_to_word
