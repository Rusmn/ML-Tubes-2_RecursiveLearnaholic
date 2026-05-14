from pathlib import Path

from common.dummy_data import teacher_pairs
from common.io import ensure_dir

from .keras_models import build_preinject, compile_model


def cap_len(config):
    if "caption_length" in config:
        return int(config["caption_length"])
    return max(int(config["max_length"]) - 1, 1)


def build_cfg(config):
    caption_length = cap_len(config)
    model = build_preinject(
        vocab_size=int(config["vocab_size"]),
        feature_dim=int(config["feature_dim"]),
        max_length=caption_length,
        embed_dim=int(config.get("embed_dim", 256)),
        hidden_size=int(config.get("hidden_size", 256)),
        recur_layers=int(config.get("recur_layers", config.get("recurrent_layers", 1))),
        recur_type=str(config.get("recur_type", config.get("recurrent_type", "lstm"))),
        dropout=float(config.get("dropout", 0.0)),
    )
    return compile_model(model, learn_rate=float(config.get("learn_rate", config.get("learning_rate", 0.001))))


def train_decoder(
    model,
    image_features,
    caption_sequences,
    val_data=None,
    batch_size=32,
    epochs=10,
    model_path=None,
    verbose=1,
):
    caption_inputs, caption_targets = teacher_pairs(caption_sequences)
    if image_features.shape[0] != caption_inputs.shape[0]:
        raise ValueError("feature dan caption harus memiliki jumlah sample yang sama.")

    fit_val = None

    if val_data is not None:
        val_features, val_sequences = val_data
        val_inputs, val_targets = teacher_pairs(val_sequences)
        if val_features.shape[0] != val_inputs.shape[0]:
            raise ValueError("feature validasi dan caption validasi harus memiliki jumlah sample yang sama.")
        fit_val = ([val_features, val_inputs], val_targets)

    history = model.fit(
        [image_features, caption_inputs],
        caption_targets,
        validation_data=fit_val,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    if model_path is not None:
        output_path = Path(model_path)
        ensure_dir(output_path.parent)
        model.save(output_path)

    return history


def train_run(image_features, caption_sequences, config, model_dir="models/rnn", val_data=None, verbose=1):
    model = build_cfg(config)
    kind = config.get("recur_type", config.get("recurrent_type", "lstm"))
    layers = config.get("recur_layers", config.get("recurrent_layers", 1))
    hidden = config.get("hidden_size", 256)
    caption_length = cap_len(config)
    model_name = config.get("model_name")

    if model_name is None:
        model_name = "{0}_layers{1}_hidden{2}_len{3}.keras".format(kind, layers, hidden, caption_length)

    model_path = Path(model_dir) / str(model_name)
    history = train_decoder(
        model,
        image_features,
        caption_sequences,
        val_data=val_data,
        batch_size=int(config.get("batch_size", 32)),
        epochs=int(config.get("epochs", 10)),
        model_path=model_path,
        verbose=verbose,
    )
    return model, history, model_path


def grid_cfg(base_config, recur_types=("rnn", "lstm"), layer_opts=(1, 2, 3), hidden_opts=(128, 512)):
    configs = []
    for kind in recur_types:
        for layers in layer_opts:
            for hidden in hidden_opts:
                config = dict(base_config)
                config["recur_type"] = kind
                config["recur_layers"] = layers
                config["hidden_size"] = hidden
                caption_length = cap_len(config)
                config["model_name"] = "{0}_layers{1}_hidden{2}_len{3}.keras".format(kind, layers, hidden, caption_length)
                configs.append(config)
    return configs


def grid_train(image_features, caption_sequences, configs, model_dir="models/rnn", val_data=None, verbose=1):
    results = []
    for config in configs:
        model, history, model_path = train_run(
            image_features,
            caption_sequences,
            config,
            model_dir=model_dir,
            val_data=val_data,
            verbose=verbose,
        )
        results.append(
            {
                "config": config,
                "model": model,
                "history": history,
                "model_path": model_path,
            }
        )
    return results
