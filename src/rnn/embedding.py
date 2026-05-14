import numpy as np

from common.autograd import Value
from model_implementation.layer.module import Module

class Embedding(Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = Value(
            np.random.randn(vocab_size, embedding_dim) * 0.01,
            label="embedding_weight"
        )

    def forward(self, indices):
        out = Value(
            self.weight.data[indices],
            (self.weight,),
            'embedding'
        )

        def _backward():
            grad = np.zeros_like(self.weight.data)
            np.add.at(grad, indices, out.grad)
            self.weight.grad += grad
        out._backward = _backward

        return out

    def parameters(self):
        return [self.weight]

    def state_dict(self):
        return {
            "weight": np.array(self.weight.data, copy=True),
        }

    def load_state(self, state):
        weight = np.array(state["weight"], dtype=float, copy=True)
        if weight.shape != self.weight.data.shape:
            raise ValueError

        self.weight.data = weight
        self.weight.grad = np.zeros_like(self.weight.data)

    def load_keras(self, weights):
        self.load_state({"weight": weights[0]})


EmbeddingLayer = Embedding
