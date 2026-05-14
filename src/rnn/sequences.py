import numpy as np


def teacher_pairs(caption_sequences):
    sequences = np.asarray(caption_sequences, dtype="int32")
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError("caption_sequences must have shape (n, length >= 2).")
    return sequences[:, :-1], sequences[:, 1:]
