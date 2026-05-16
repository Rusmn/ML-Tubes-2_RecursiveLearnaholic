import re

import numpy as np

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
