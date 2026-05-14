import time
from contextlib import contextmanager

import numpy as np

def _as_tokens(sentence):
    if isinstance(sentence, str):
        return sentence.split()
    return list(sentence)

def bleu4_score(references, prediction):
    reference_tokens = [_as_tokens(reference) for reference in references]
    prediction_tokens = _as_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        smoother = SmoothingFunction().method1
        return float(sentence_bleu(reference_tokens, prediction_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother))
    except Exception:
        reference_vocab = set(token for reference in reference_tokens for token in reference)
        matches = sum(token in reference_vocab for token in prediction_tokens)
        return float(matches / max(len(prediction_tokens), 1))

def meteor_safe(references, prediction):
    reference_tokens = [_as_tokens(reference) for reference in references]
    prediction_tokens = _as_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    try:
        from nltk.translate.meteor_score import meteor_score

        return float(meteor_score(reference_tokens, prediction_tokens))
    except Exception:
        reference_vocab = set(token for reference in reference_tokens for token in reference)
        prediction_vocab = set(prediction_tokens)
        overlap = len(reference_vocab & prediction_vocab)
        precision = overlap / max(len(prediction_vocab), 1)
        recall = overlap / max(len(reference_vocab), 1)
        if precision + recall == 0:
            return 0.0
        return float((2 * precision * recall) / (precision + recall))

def score_set(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")

    bleu_scores = [bleu4_score(refs, pred) for refs, pred in zip(references, predictions)]
    meteor_scores = [meteor_safe(refs, pred) for refs, pred in zip(references, predictions)]
    return {
        "bleu4": float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        "meteor": float(np.mean(meteor_scores)) if meteor_scores else 0.0,
    }

@contextmanager
def timer():
    start = time.perf_counter()
    result = {"elapsed_seconds": 0.0}
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
