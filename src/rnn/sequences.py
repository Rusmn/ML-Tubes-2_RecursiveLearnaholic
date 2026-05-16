import numpy as np

def teacher_pairs(caption_sequences):
    sequences = np.asarray(caption_sequences, dtype="int32")
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError("caption_sequences must have shape (n, length >= 2).")
    return sequences[:, :-1], sequences[:, 1:]

def align_features_to_captions(features, feature_ids, caption_sequences, caption_ids):
    features = np.asarray(features, dtype="float32")
    sequences = np.asarray(caption_sequences, dtype="int32")
    feature_lookup = {
        str(image_id): features[index]
        for index, image_id in enumerate(feature_ids)
    }

    aligned_features = []
    aligned_sequences = []
    aligned_ids = []
    missing_ids = []

    for image_id, sequence in zip(caption_ids, sequences):
        key = str(image_id)
        if key not in feature_lookup:
            missing_ids.append(key)
            continue
        aligned_features.append(feature_lookup[key])
        aligned_sequences.append(sequence)
        aligned_ids.append(key)

    if not aligned_features:
        raise ValueError("No caption rows could be aligned to extracted image features.")

    return (
        np.asarray(aligned_features, dtype="float32"),
        np.asarray(aligned_sequences, dtype="int32"),
        aligned_ids,
        missing_ids,
    )
