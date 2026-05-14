from model_implementation.layer.module import Linear


def dense_state(weights):
    bias = weights[1].reshape(1, -1) if len(weights) > 1 and weights[1] is not None else None
    return {
        "weight": weights[0],
        "bias": bias,
    }


class DenseProjectionLayer(Linear):
    def forward(self, x):
        return super().forward(x)

    def load_keras(self, weights):
        self.load_state(dense_state(weights))

class DenseOutputLayer(Linear):
    def forward(self, x):
        return super().forward(x).softmax(axis=-1)

    def load_keras(self, weights):
        self.load_state(dense_state(weights))
