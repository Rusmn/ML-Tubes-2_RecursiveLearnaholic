import numpy as np

from common.autograd import Value

from .dense import DenseOutputLayer, DenseProjectionLayer
from .embedding import EmbeddingLayer
from .lstm import LSTM
from .rnn import RNN

def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def layer_weights(arrays, name):
    weights = []
    index = 0
    while "{0}_{1}".format(name, index) in arrays:
        weights.append(arrays["{0}_{1}".format(name, index)])
        index += 1
    if not weights:
        raise ValueError("bobot layer tidak ditemukan: {0}".format(name))
    return weights

class CaptionDecoder:
    def __init__(
        self,
        vocab_size,
        feature_dim,
        embed_dim,
        hidden_size,
        recur_layers=1,
        recur_type="lstm",
    ):
        kind = recur_type.lower()
        if kind not in ("rnn", "simple_rnn", "lstm"):
            raise ValueError("recur_type harus berisi rnn atau lstm.")

        self.kind = "lstm" if kind == "lstm" else "rnn"
        self.recur_layers = int(recur_layers)
        self.feature_projection = DenseProjectionLayer(feature_dim, embed_dim)
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        self.recurrent = (
            LSTM(embed_dim, hidden_size, self.recur_layers)
            if self.kind == "lstm"
            else RNN(embed_dim, hidden_size, self.recur_layers)
        )
        self.output = DenseOutputLayer(hidden_size, vocab_size)

    def load_arrays(self, arrays):
        self.feature_projection.load_keras(layer_weights(arrays, "feature_projection"))
        self.embedding.load_keras(layer_weights(arrays, "token_embedding"))
        self.output.load_keras(layer_weights(arrays, "token_dist"))

        prefix = "lstm" if self.kind == "lstm" else "rnn"
        weights = []
        for index in range(self.recur_layers):
            weights.append(layer_weights(arrays, "{0}_{1}".format(prefix, index + 1)))
        self.recurrent.load_keras(weights)
        return self

    def load_npz(self, path):
        return self.load_arrays(load_npz(path))

    def predict(self, image_feature, caption_tokens):
        feature = np.asarray(image_feature, dtype="float32")
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)

        tokens = np.asarray(caption_tokens, dtype="int32")
        if tokens.ndim == 1:
            tokens = tokens.reshape(1, -1)

        feature_token = self.feature_projection(Value(feature)).relu().data.reshape(feature.shape[0], 1, -1)
        token_embeddings = self.embedding(tokens).data
        sequence = np.concatenate([feature_token, token_embeddings], axis=1)
        hidden_seq, _ = self.recurrent(sequence)
        hidden_seq = hidden_seq.data[:, 1:, :]

        batch_size, seq_len, hidden_size = hidden_seq.shape
        flat_hidden = hidden_seq.reshape(batch_size * seq_len, hidden_size)
        probs = self.output(Value(flat_hidden)).data
        return probs.reshape(batch_size, seq_len, -1)

def build_decoder(config, weight_path):
    decoder = CaptionDecoder(
        vocab_size=int(config["vocab_size"]),
        feature_dim=int(config["feature_dim"]),
        embed_dim=int(config.get("embed_dim", 256)),
        hidden_size=int(config.get("hidden_size", 256)),
        recur_layers=int(config.get("recur_layers", config.get("recurrent_layers", 1))),
        recur_type=str(config.get("recur_type", config.get("recurrent_type", "lstm"))),
    )
    return decoder.load_npz(weight_path)
