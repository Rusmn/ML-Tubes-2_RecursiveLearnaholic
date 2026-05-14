def need_tf():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError
    return tf


def build_preinject(
    vocab_size,
    feature_dim,
    max_length,
    embed_dim=256,
    hidden_size=256,
    recur_layers=1,
    recur_type="lstm",
    dropout=0.0,
):
    tf = need_tf()
    kind = recur_type.lower()

    if recur_layers <= 0:
        raise ValueError("recur_layers harus lebih dari 0.")
    if kind not in ("rnn", "simple_rnn", "lstm"):
        raise ValueError("recur_type harus berisi 'rnn', 'simple_rnn', atau 'lstm'.")

    image_features = tf.keras.Input(shape=(feature_dim,), name="image_features")
    caption_tokens = tf.keras.Input(shape=(None,), dtype="int32", name="caption_tokens")

    feature_token = tf.keras.layers.Dense(
        embed_dim,
        activation="relu",
        name="feature_projection",
    )(image_features)
    feature_token = tf.keras.layers.Reshape((1, embed_dim), name="feature_token")(feature_token)

    token_embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_tokens)

    x = tf.keras.layers.Concatenate(axis=1, name="preinject_concat")(
        [feature_token, token_embeddings]
    )

    layer_class = tf.keras.layers.LSTM if kind == "lstm" else tf.keras.layers.SimpleRNN
    layer_prefix = "lstm" if kind == "lstm" else "rnn"

    for index in range(recur_layers):
        x = layer_class(
            hidden_size,
            return_sequences=True,
            dropout=dropout,
            name="{0}_{1}".format(layer_prefix, index + 1),
        )(x)

    x = tf.keras.layers.Lambda(lambda tensor: tensor[:, 1:, :], name="drop_feature")(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="token_dist")(x)

    return tf.keras.Model(
        inputs=[image_features, caption_tokens],
        outputs=outputs,
        name="preinject_{0}".format(layer_prefix),
    )


def compile_model(model, learn_rate=0.001):
    tf = need_tf()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model
