from common.autograd.autograd import Value
import numpy as np

# parent dari semua yang di layer FFNN
class Module:
    def __init__(self):
        self.training = True
    
    def parameters(self):
        raise NotImplementedError

    def children(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def train(self):
        self.training = True
        for child in self.children():
            child.train()
        return self
    
    def eval(self):
        self.training = False
        for child in self.children():
            child.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *inputs):
        raise NotImplementedError

class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        initialization="xavier_uniform",
        init_params=None,
        seed=None,
        rng=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.initialization = initialization
        self.init_params = dict(init_params or {})
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(seed)

        self.reset_parameters()

    def _sample(self, shape, is_bias=False):
        init_name = str(self.initialization).lower()

        if init_name == "zero":
            return np.zeros(shape, dtype=float)

        if init_name == "uniform":
            lower = self.init_params.get("lower_bound", -0.5)
            upper = self.init_params.get("upper_bound", 0.5)
            return self.rng.uniform(lower, upper, size=shape)

        if init_name == "normal":
            mean = self.init_params.get("mean", 0.0)
            variance = self.init_params.get("variance", 1.0)
            if variance < 0:
                raise ValueError
            return self.rng.normal(mean, np.sqrt(variance), size=shape)

        if init_name in {"xavier_uniform", "xavier"}:
            if is_bias:
                return np.zeros(shape, dtype=float)
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            return self.rng.uniform(-limit, limit, size=shape)

        if init_name in {"he_normal", "he"}:
            if is_bias:
                return np.zeros(shape, dtype=float)
            std = np.sqrt(2.0 / self.in_features)
            return self.rng.normal(0.0, std, size=shape)

        raise ValueError

    def reset_parameters(self):
        self.weight = Value(
            self._sample((self.in_features, self.out_features)),
            label="weight",
        )
        if self.use_bias:
            self.bias = Value(
                self._sample((1, self.out_features), is_bias=True),
                label="bias",
            )
        else:
            self.bias = None
    
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        out = x @ self.weight
        if self.use_bias:
            out = out + self.bias
        return out
    
    def parameters(self):
        params = [self.weight]

        if self.use_bias:
            params.append(self.bias)
        return params

    def state_dict(self):
        return {
            "weight": np.array(self.weight.data, copy=True),
            "bias": None if self.bias is None else np.array(self.bias.data, copy=True),
        }

    def load_state(self, state):
        self.weight.data = np.array(state["weight"], dtype=float, copy=True)
        self.weight.grad = np.zeros_like(self.weight.data)

        if self.use_bias:
            if state.get("bias") is None:
                raise ValueError
            self.bias.data = np.array(state["bias"], dtype=float, copy=True)
            self.bias.grad = np.zeros_like(self.bias.data)
        elif state.get("bias") is not None:
            raise ValueError

# biar bisa Sequential(Linear(10,32), ReLU(), Linear(64, 3))
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = tuple(layers)

    def children(self):
        return list(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
