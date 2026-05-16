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
