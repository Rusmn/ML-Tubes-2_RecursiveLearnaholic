import numpy as np

from common.autograd.autograd import Value
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