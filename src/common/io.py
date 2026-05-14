import json
from pathlib import Path

import numpy as np


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data, path):
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    return output_path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_npy(array, path):
    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.save(output_path, array)
    return output_path


def load_npy(path):
    return np.load(Path(path), allow_pickle=False)


def save_npz(path, **arrays):
    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.savez(output_path, **arrays)
    return output_path
