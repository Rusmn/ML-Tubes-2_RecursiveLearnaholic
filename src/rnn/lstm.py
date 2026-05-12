import numpy as np

from common.autograd.autograd import Value
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

class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = LSTMCell(
            input_size,
            hidden_size
        )

    def children(self):
        return [self.cell]

    def forward(self, x, h0=None, c0=None):
        batch_size, seq_len, _ = x.data.shape

        if h0 is None:
            h_t = Value(
                np.zeros((batch_size, self.hidden_size))
            )
        else:
            h_t = h0

        if c0 is None:
            c_t = Value(
                np.zeros((batch_size, self.hidden_size))
            )
        else:
            c_t = c0

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(
                x_t,
                h_t,
                c_t
            )
            outputs.append(h_t)

        output = Value.stack(outputs, axis=1)
        return output, (h_t, c_t)

    def parameters(self):
        return self.cell.parameters()