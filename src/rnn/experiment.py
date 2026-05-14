from pathlib import Path

import numpy as np

from common.io import ensure_dir, load_json, load_npy, save_json

from .evaluate import eval_keras, hist_sum
from .keras_models import need_tf
from .train import grid_cfg, train_run
from .weights import export_weights

def make_index(word_to_index):
    return {int(index): word for word, index in word_to_index.items()}

def load_vocab(vocab_dir):
    vocab_path = Path(vocab_dir)
    word_to_index = load_json(vocab_path / "vocab.json")
    index_to_word = make_index(word_to_index)
    return word_to_index, index_to_word

def load_data(feature_path, caption_path, vocab_dir):
    features = load_npy(feature_path).astype("float32")
    captions = load_npy(caption_path).astype("int32")
    word_to_index, index_to_word = load_vocab(vocab_dir)

    if features.shape[0] != captions.shape[0]:
        raise ValueError("feature dan caption harus memiliki jumlah sample yang sama.")

    return features, captions, word_to_index, index_to_word

def split_data(features, captions, val_ratio=0.2, seed=42):
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError("val_ratio harus di antara 0 dan 1.")

    rng = np.random.default_rng(seed)
    order = np.arange(features.shape[0])
    rng.shuffle(order)

    val_count = int(round(features.shape[0] * val_ratio))
    val_index = order[:val_count]
    train_index = order[val_count:]

    train_data = (features[train_index], captions[train_index])
    val_data = (features[val_index], captions[val_index])
    return train_data, val_data

def save_hist(history, path):
    return save_json(hist_sum(history), path)

def save_score(score, path):
    clean_score = {}
    for key, value in score.items():
        if isinstance(value, (np.integer, np.floating)):
            clean_score[key] = value.item()
        else:
            clean_score[key] = value
    return save_json(clean_score, path)

def train_grid(
    features,
    captions,
    base_config,
    model_dir="models/rnn",
    report_dir="reports/tables",
    val_ratio=0.2,
    seed=42,
    verbose=1,
):
    train_data, val_data = split_data(features, captions, val_ratio=val_ratio, seed=seed)
    train_features, train_captions = train_data
    configs = grid_cfg(base_config)
    records = []
    report_path = ensure_dir(report_dir)

    for config in configs:
        model, history, model_path = train_run(
            train_features,
            train_captions,
            config,
            model_dir=model_dir,
            val_data=val_data,
            verbose=verbose,
        )
        name = Path(model_path).stem
        hist_path = report_path / "{0}_history.json".format(name)
        weight_path = Path(model_dir) / "{0}.npz".format(name)

        save_hist(history, hist_path)
        export_weights(model, weight_path)

        records.append(
            {
                "config": config,
                "model_path": str(model_path),
                "weight_path": str(weight_path),
                "history_path": str(hist_path),
            }
        )

    save_json(records, report_path / "train_records.json")
    return records

def load_model(model_path):
    tf = need_tf()
    return tf.keras.models.load_model(model_path)

def eval_model(record, features, captions, word_to_index, index_to_word, max_length, report_dir="reports/tables"):
    model = load_model(record["model_path"])
    score = eval_keras(model, features, captions, word_to_index, index_to_word, max_length)

    config = record.get("config", {})
    score["recur_type"] = config.get("recur_type")
    score["recur_layers"] = config.get("recur_layers")
    score["hidden_size"] = config.get("hidden_size")
    score["model_path"] = record["model_path"]

    name = Path(record["model_path"]).stem
    score_path = Path(report_dir) / "{0}_score.json".format(name)
    save_score(score, score_path)
    score["score_path"] = str(score_path)
    return score

def eval_grid(records, features, captions, word_to_index, index_to_word, max_length, report_dir="reports/tables"):
    ensure_dir(report_dir)
    scores = []
    for record in records:
        score = eval_model(
            record,
            features,
            captions,
            word_to_index,
            index_to_word,
            max_length,
            report_dir=report_dir,
        )
        scores.append(score)

    save_json(scores, Path(report_dir) / "rnn_scores.json")
    return scores

def best_score(scores, key="bleu4"):
    if not scores:
        return None
    return max(scores, key=lambda score: score.get(key, 0.0))

def eval_lengths(model_path, features, captions, word_to_index, index_to_word, lengths, report_dir="reports/tables"):
    model = load_model(model_path)
    scores = []

    for length in lengths:
        score = eval_keras(model, features, captions, word_to_index, index_to_word, int(length))
        score["max_length"] = int(length)
        score["model_path"] = str(model_path)
        scores.append(score)

    save_json(scores, Path(report_dir) / "length_scores.json")
    return scores
