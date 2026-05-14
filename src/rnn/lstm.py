import numpy as np

from common.autograd import Value
from model_implementation.layer.module import Module

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        def init_gate(in_dim):
            return Value(
                np.random.randn(in_dim, hidden_size)
                / np.sqrt(in_dim)
            )

        self.W_f = init_gate(input_size)
        self.U_f = init_gate(hidden_size)
        self.b_f = Value(np.zeros((1, hidden_size)))

        self.W_i = init_gate(input_size)
        self.U_i = init_gate(hidden_size)
        self.b_i = Value(np.zeros((1, hidden_size)))

        self.W_c = init_gate(input_size)
        self.U_c = init_gate(hidden_size)
        self.b_c = Value(np.zeros((1, hidden_size)))

        self.W_o = init_gate(input_size)
        self.U_o = init_gate(hidden_size)
        self.b_o = Value(np.zeros((1, hidden_size)))

    def forward(self, x_t, h_prev, c_prev):
        f_t = (
            x_t @ self.W_f
            + h_prev @ self.U_f
            + self.b_f
        ).sigmoid()

        i_t = (
            x_t @ self.W_i
            + h_prev @ self.U_i
            + self.b_i
        ).sigmoid()

        c_tilde = (
            x_t @ self.W_c
            + h_prev @ self.U_c
            + self.b_c
        ).tanh()

        o_t = (
            x_t @ self.W_o
            + h_prev @ self.U_o
            + self.b_o
        ).sigmoid()

        c_t = f_t * c_prev + i_t * c_tilde

        h_t = o_t * c_t.tanh()

        return h_t, c_t

    def parameters(self):
        return [
            self.W_f, self.U_f, self.b_f,
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o,
        ]

    def state_dict(self):
        return {
            "W_f": np.array(self.W_f.data, copy=True),
            "U_f": np.array(self.U_f.data, copy=True),
            "b_f": np.array(self.b_f.data, copy=True),
            "W_i": np.array(self.W_i.data, copy=True),
            "U_i": np.array(self.U_i.data, copy=True),
            "b_i": np.array(self.b_i.data, copy=True),
            "W_c": np.array(self.W_c.data, copy=True),
            "U_c": np.array(self.U_c.data, copy=True),
            "b_c": np.array(self.b_c.data, copy=True),
            "W_o": np.array(self.W_o.data, copy=True),
            "U_o": np.array(self.U_o.data, copy=True),
            "b_o": np.array(self.b_o.data, copy=True),
        }

    def load_state(self, state):
        for name in self.state_dict():
            if name not in state:
                raise ValueError

        for name in self.state_dict():
            param = getattr(self, name)
            param.data = np.array(state[name], dtype=float, copy=True)
            param.grad = np.zeros_like(param.data)

    def load_keras(self, weights):
        if len(weights) < 3:
            raise ValueError

        kernel = np.array(weights[0], dtype=float, copy=True)
        recurrent = np.array(weights[1], dtype=float, copy=True)
        bias = np.array(weights[2], dtype=float, copy=True).reshape(-1)
        hidden = self.hidden_size

        if kernel.shape[1] != 4 * hidden:
            raise ValueError

        self.load_state({
            "W_i": kernel[:, 0:hidden],
            "W_f": kernel[:, hidden:2 * hidden],
            "W_c": kernel[:, 2 * hidden:3 * hidden],
            "W_o": kernel[:, 3 * hidden:4 * hidden],
            "U_i": recurrent[:, 0:hidden],
            "U_f": recurrent[:, hidden:2 * hidden],
            "U_c": recurrent[:, 2 * hidden:3 * hidden],
            "U_o": recurrent[:, 3 * hidden:4 * hidden],
            "b_i": bias[0:hidden].reshape(1, hidden),
            "b_f": bias[hidden:2 * hidden].reshape(1, hidden),
            "b_c": bias[2 * hidden:3 * hidden].reshape(1, hidden),
            "b_o": bias[3 * hidden:4 * hidden].reshape(1, hidden),
        })

class LSTM(Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.layers = int(layers)
        if self.layers <= 0:
            raise ValueError

        self.cells = []
        in_size = input_size
        for _ in range(self.layers):
            cell = LSTMCell(in_size, hidden_size)
            self.cells.append(cell)
            in_size = hidden_size

        self.cell = self.cells[0]

    def children(self):
        return list(self.cells)

    def forward(self, x, h0=None, c0=None):
        x = x if isinstance(x, Value) else Value(x)
        batch_size, seq_len, _ = x.data.shape
        layer_input = x
        last_h = None
        last_c = None

        for index, cell in enumerate(self.cells):
            if h0 is None:
                h_t = Value(np.zeros((batch_size, self.hidden_size)))
            elif isinstance(h0, (list, tuple)):
                h_t = h0[index]
            else:
                h_t = h0

            if c0 is None:
                c_t = Value(np.zeros((batch_size, self.hidden_size)))
            elif isinstance(c0, (list, tuple)):
                c_t = c0[index]
            else:
                c_t = c0

            outputs = []
            for t in range(seq_len):
                x_t = layer_input[:, t, :]
                h_t, c_t = cell(
                    x_t,
                    h_t,
                    c_t
                )
                outputs.append(h_t)

            layer_input = Value.stack(outputs, axis=1)
            last_h = h_t
            last_c = c_t

        return layer_input, (last_h, last_c)

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
