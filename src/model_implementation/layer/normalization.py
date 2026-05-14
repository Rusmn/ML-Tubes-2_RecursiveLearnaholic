import numpy as np

from common.autograd import Value
from model_implementation.layer.module import Module

class RMSNorm(Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma = Value(np.ones((1, self.dim)), label="gamma")

    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)

        data = x.data
        rms = np.sqrt(np.mean(data ** 2, axis=1, keepdims=True) + self.eps)
        normalized = data / rms
        out = Value(normalized * self.gamma.data, (x, self.gamma), "rmsnorm")

        def _backward():
            grad_out = out.grad
            gamma_data = self.gamma.data
            dim = data.shape[1]

            self.gamma.grad += np.sum(grad_out * normalized, axis=0, keepdims=True)

            scaled_grad = grad_out * gamma_data
            dot = np.sum(scaled_grad * data, axis=1, keepdims=True)
            dx = scaled_grad / rms - data * dot / (dim * (rms ** 3))
            x.grad += dx

        out._backward = _backward
        return out

    def parameters(self):
        return [self.gamma]

    def state_dict(self):
        return {
            "gamma": np.array(self.gamma.data, copy=True),
        }

    def load_state(self, state):
        self.gamma.data = np.array(state["gamma"], dtype=float, copy=True)
        self.gamma.grad = np.zeros_like(self.gamma.data)
