import numpy as np

from common.autograd.autograd import Value
from model_implementation.layer.module import Module

class SimpleRNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = Value(
            np.random.randn(input_size, hidden_size)
            / np.sqrt(input_size),
            label="W_xh"
        )

        self.W_hh = Value(
            np.random.randn(hidden_size, hidden_size)
            / np.sqrt(hidden_size),
            label="W_hh"
        )

        self.b_h = Value(
            np.zeros((1, hidden_size)),
            label="b_h"
        )

    def forward(self, x_t, h_prev):
        h_t = (
            x_t @ self.W_xh
            + h_prev @ self.W_hh
            + self.b_h
        ).tanh()

        return h_t

    def parameters(self):
        return [
            self.W_xh,
            self.W_hh,
            self.b_h
        ]

class RNN(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = SimpleRNNCell(
            input_size,
            hidden_size
        )

    def children(self):
        return [self.cell]

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.data.shape

        if h0 is None:
            h_t = Value(
                np.zeros((batch_size, self.hidden_size))
            )
        else:
            h_t = h0

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_t)
            outputs.append(h_t)
        output = Value.stack(outputs, axis=1)

        return output, h_t

    def parameters(self):
        return self.cell.parameters()