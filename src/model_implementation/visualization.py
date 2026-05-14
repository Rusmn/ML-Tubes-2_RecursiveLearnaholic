import numpy as np

def _layers(model):
    return model.trainable()

def _pick_layers(model, layers):
    trainable_layers = _layers(model)
    if layers is None:
        indices = list(range(len(trainable_layers)))
    else:
        indices = list(layers)

    for index in indices:
        if index < 0 or index >= len(trainable_layers):
            raise ValueError

    return indices, trainable_layers

def _values(layer, attr):
    values = []
    for parameter in layer.parameters():
        value = getattr(parameter, attr, None)
        if value is not None:
            values.append(np.ravel(value))
    if not values:
        return np.array([])
    return np.concatenate(values)

def plot_weights(model, layers=None, bins=30):
    import matplotlib.pyplot as plt

    indices, trainable_layers = _pick_layers(model, layers)
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for axis, layer_index in zip(axes, indices):
        values = _values(trainable_layers[layer_index], "data")
        axis.hist(values, bins=bins)
        axis.set_title(f"Weight Distribution - Trainable Layer {layer_index}")
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")

    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_grads(model, layers=None, bins=30):
    import matplotlib.pyplot as plt

    indices, trainable_layers = _pick_layers(model, layers)
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for axis, layer_index in zip(axes, indices):
        values = _values(trainable_layers[layer_index], "grad")
        axis.hist(values, bins=bins)
        axis.set_title(f"Gradient Distribution - Trainable Layer {layer_index}")
        axis.set_xlabel("Gradient")
        axis.set_ylabel("Frequency")

    fig.tight_layout()
    plt.show()
    return fig, axes

def plot_weights_and_grads(model, layers=None, bins=30):
    import matplotlib.pyplot as plt

    indices, trainable_layers = _pick_layers(model, layers)

    fig, axes = plt.subplots(len(indices), 2, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    fig.suptitle(f"Weight & Gradient Distribution", fontsize=14)

    for row, layer_index in zip(axes, indices):
        weight_values = _values(trainable_layers[layer_index], "data")
        grad_values = _values(trainable_layers[layer_index], "grad")

        # Weights
        row[0].hist(weight_values, bins=bins)
        row[0].set_title(f"Weights - Layer {layer_index}")
        row[0].set_xlabel("Value")
        row[0].set_ylabel("Frequency")

        # Gradients
        row[1].hist(grad_values, bins=bins)
        row[1].set_title(f"Gradients - Layer {layer_index}")
        row[1].set_xlabel("Gradient")
        row[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return fig, axes
