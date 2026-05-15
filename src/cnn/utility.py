import numpy as np
from PIL import Image
import pathlib

def image_loader(path, target_size=(224, 224)):
    """Loads a single image, resizes, and normalizes pixel values."""
    image = Image.open(path).convert("RGB")
    image = image.resize(tuple(target_size))
    return np.asarray(image, dtype="float32") / 255.0

def batch_loader(paths, target_size=(224, 224)):
    """Loads a batch of images into a (N, H, W, C) array."""
    return np.stack([image_loader(path, target_size) for path in paths]).astype("float32")

def image_paths(root_dir, extensions=(".jpg", ".jpeg", ".png")):
    root_path = pathlib.Path(root_dir)
    paths = []
    for extension in extensions:
        paths.extend(root_path.rglob(f"*{extension}"))
    return sorted(path for path in paths if path.is_file())

def feature_extractor(paths, keras_encoder, save_path, target_size=(224, 224), batch_size=32, image_id_path=None, preprocess_fn=None):
    ordered_paths = [pathlib.Path(path) for path in paths]
    features = []

    for start in range(0, len(ordered_paths), int(batch_size)):
        batch_paths = ordered_paths[start:start + int(batch_size)]
        images = batch_loader(batch_paths, target_size=target_size)
        if preprocess_fn is not None:
            images = preprocess_fn(images * 255.0)
        batch_features = keras_encoder.predict(images, verbose=0)
        features.append(np.asarray(batch_features))

    if features:
        feature_array = np.concatenate(features, axis=0)
    else:
        feature_array = np.empty((0, 0), dtype="float32")

    output_path = pathlib.Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature_array)

    if image_id_path is not None:
        id_path = pathlib.Path(image_id_path)
        id_path.parent.mkdir(parents=True, exist_ok=True)
        ids = [path.stem for path in ordered_paths]
        id_path.write_text("\n".join(ids), encoding="utf-8")

    return feature_array
