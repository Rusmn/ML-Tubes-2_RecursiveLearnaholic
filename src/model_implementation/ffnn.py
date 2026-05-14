import numpy as np

from model_implementation.layer import activation as activation_layer
from model_implementation.layer import normalization as normalization_layer
from model_implementation.layer.module import Linear, Module, Sequential
from model_implementation.serialization import load_net, save_net
from model_implementation.trainer import fit_net, predict, proba
from model_implementation.visualization import (
    plot_grads,
    plot_weights,
    plot_weights_and_grads,
)

class FFNN(Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=None,
        output_activation=None,
        bias=True,
        initialization="xavier_uniform",
        init_params=None,
        seed=None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif isinstance(hidden_dims, (int, np.integer)):
            hidden_dims = [hidden_dims]
        else:
            hidden_dims = list(hidden_dims)

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.output_dim = int(output_dim)
        self.bias = bool(bias)
        self.initialization = initialization
        self.init_params = dict(init_params or {})
        self.seed = seed

        self.activation_spec, self.output_activation_spec = self._get_acts(
            activation,
            output_activation,
        )

        self.history_ = {"training_loss": [], "validation_loss": []}
        self.training_loss_history = []
        self.validation_loss_history = []
        self.classes_ = None
        self.task_ = None
        self.loss_name_ = None

        self.network = self._make_net()

    def _norm_spec(self, spec):
        if spec is None:
            return None

        if isinstance(spec, dict):
            normalized = dict(spec)
            if "name" not in normalized:
                raise ValueError
            normalized["name"] = normalized["name"].lower()
            return normalized

        if isinstance(spec, str):
            return {"name": spec.lower()}

        if isinstance(spec, Module):
            normalized = {"name": spec.__class__.__name__.lower()}
            if hasattr(spec, "eps"):
                normalized["eps"] = float(spec.eps)
            return normalized

        if isinstance(spec, type) and issubclass(spec, Module):
            return {"name": spec.__name__.lower()}

        raise TypeError

    def _get_acts(self, activation, output_activation):
        hidden_count = len(self.hidden_dims)

        if isinstance(activation, (list, tuple)):
            activation_specs = [self._norm_spec(spec) for spec in activation]

            if output_activation is None and len(activation_specs) == hidden_count + 1:
                return activation_specs[:-1], activation_specs[-1]

            if len(activation_specs) != hidden_count:
                raise ValueError
            return activation_specs, self._norm_spec(output_activation)

        hidden_spec = self._norm_spec(activation)
        return [hidden_spec] * hidden_count, self._norm_spec(output_activation)

    def _make_mod(self, module_spec, out_features):
        if module_spec is None:
            return None

        name = module_spec["name"]
        if hasattr(activation_layer, name):
            module_cls = getattr(activation_layer, name)
            return module_cls()

        if name in {"rmsnorm", "rms_norm"}:
            eps = module_spec.get("eps", 1e-8)
            return normalization_layer.RMSNorm(out_features, eps=eps)

        if hasattr(normalization_layer, name):
            module_cls = getattr(normalization_layer, name)
            return module_cls(out_features)

        raise ValueError

    def _make_net(self):
        layer_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        rng = np.random.default_rng(self.seed)
        layers = []

        for index, (in_features, out_features) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(
                Linear(
                    in_features,
                    out_features,
                    bias=self.bias,
                    initialization=self.initialization,
                    init_params=self.init_params,
                    rng=rng,
                )
            )

            is_output_layer = index == len(layer_dims) - 2
            module_spec = self.output_activation_spec if is_output_layer else self.activation_spec[index]
            module = self._make_mod(module_spec, out_features)
            if module is not None:
                layers.append(module)

        return Sequential(*layers)

    def children(self):
        return [self.network]

    def forward(self, x):
        return self.network(x)

    def parameters(self):
        return self.network.parameters()

    def reset_parameters(self):
        for layer in self.network.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def trainable(self):
        return [layer for layer in self.network.layers if len(layer.parameters()) > 0]

    def regularization_l1(self, lambda_):
        lambda_ = float(lambda_)
        total = 0.0
        for layer in self.trainable():
            for parameter in layer.parameters():
                total += np.sum(np.abs(parameter.data))
        return lambda_ * total

    def regularization_l2(self, lambda_):
        lambda_ = float(lambda_)
        total = 0.0
        for layer in self.trainable():
            for parameter in layer.parameters():
                total += np.sum(parameter.data ** 2)
        return lambda_ * total

    def add_l1(self, lambda_):
        lambda_ = float(lambda_)
        for layer in self.trainable():
            for parameter in layer.parameters():
                parameter.grad += lambda_ * np.sign(parameter.data)

    def add_l2(self, lambda_):
        lambda_ = float(lambda_)
        for layer in self.trainable():
            for parameter in layer.parameters():
                parameter.grad += 2.0 * lambda_ * parameter.data

    def config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "output_dim": self.output_dim,
            "activation": self.activation_spec,
            "output_activation": self.output_activation_spec,
            "bias": self.bias,
            "initialization": self.initialization,
            "init_params": dict(self.init_params),
            "seed": self.seed,
        }

    def state_dict(self):
        states = []
        for layer in self.network.layers:
            if hasattr(layer, "state_dict"):
                states.append(layer.state_dict())
            else:
                states.append(None)
        return states

    def load_state(self, states):
        if len(states) != len(self.network.layers):
            raise ValueError

        for layer, state in zip(self.network.layers, states):
            if state is not None and hasattr(layer, "load_state"):
                layer.load_state(state)

    def fit(self, *args, **kwargs):
        return fit_net(self, *args, **kwargs)

    def train(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return super().train()
        return self.fit(*args, **kwargs)

    def predict_proba(self, x):
        return proba(self, x)

    def predict(self, x):
        return predict(self, x)

    def save(self, path):
        save_net(self, path)

    @classmethod
    def load(cls, path):
        return load_net(path, cls)

    def plot_weights(self, layers=None, bins=30):
        return plot_weights(self, layers=layers, bins=bins)

    def plot_grads(self, layers=None, bins=30):
        return plot_grads(self, layers=layers, bins=bins)
    
    def plot_weights_and_grads(self, layers=None, bins=30):
        return plot_weights_and_grads(self, layers=layers, bins=bins)
