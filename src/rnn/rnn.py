import numpy as np

from common.autograd import Value
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

    def state_dict(self):
        return {
            "W_xh": np.array(self.W_xh.data, copy=True),
            "W_hh": np.array(self.W_hh.data, copy=True),
            "b_h": np.array(self.b_h.data, copy=True),
        }

    def load_state(self, state):
        for name in ("W_xh", "W_hh", "b_h"):
            if name not in state:
                raise ValueError

        self.W_xh.data = np.array(state["W_xh"], dtype=float, copy=True)
        self.W_hh.data = np.array(state["W_hh"], dtype=float, copy=True)
        self.b_h.data = np.array(state["b_h"], dtype=float, copy=True)

        self.W_xh.grad = np.zeros_like(self.W_xh.data)
        self.W_hh.grad = np.zeros_like(self.W_hh.data)
        self.b_h.grad = np.zeros_like(self.b_h.data)

    def load_keras(self, weights):
        if len(weights) < 3:
            raise ValueError
        self.load_state({
            "W_xh": weights[0],
            "W_hh": weights[1],
            "b_h": weights[2],
        })

class RNN(Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.layers = int(layers)
        if self.layers <= 0:
            raise ValueError

        self.cells = []
        in_size = input_size
        for _ in range(self.layers):
            cell = SimpleRNNCell(in_size, hidden_size)
            self.cells.append(cell)
            in_size = hidden_size

        self.cell = self.cells[0]

    def children(self):
        return list(self.cells)

    def forward(self, x, h0=None):
        x = x if isinstance(x, Value) else Value(x)
        batch_size, seq_len, _ = x.data.shape
        layer_input = x
        last_hidden = None

        for index, cell in enumerate(self.cells):
            if h0 is None:
                h_t = Value(np.zeros((batch_size, self.hidden_size)))
            elif isinstance(h0, (list, tuple)):
                h_t = h0[index]
            else:
                h_t = h0

            outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                h_t = cell(x_t, h_t)
                outputs.append(h_t)

            layer_input = Value.stack(outputs, axis=1)
            last_hidden = h_t

        return layer_input, last_hidden

    def parameters(self):
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params

    def state_dict(self):
        return [cell.state_dict() for cell in self.cells]

    def load_state(self, states):
        if len(states) != len(self.cells):
            raise ValueError
        for cell, state in zip(self.cells, states):
            cell.load_state(state)

    def load_keras(self, weights):
        if len(weights) != self.layers:
            raise ValueError
        for cell, cell_weights in zip(self.cells, weights):
            cell.load_keras(cell_weights)
