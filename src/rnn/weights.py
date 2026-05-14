from pathlib import Path

import numpy as np

from common.io import ensure_dir


def export_weights(model, path):
    output_path = Path(path)
    ensure_dir(output_path.parent)

    arrays = {}
    for layer in model.layers:
        weights = layer.get_weights()
        for index, weight in enumerate(weights):
            arrays["{0}_{1}".format(layer.name, index)] = np.asarray(weight)

    np.savez(output_path, **arrays)
    return output_path
