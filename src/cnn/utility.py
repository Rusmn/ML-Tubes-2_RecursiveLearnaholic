import numpy as np
from PIL import Image

def image_loader(path, target_size=(224, 224)):
    """Loads a single image, resizes, and normalizes pixel values."""
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size)
    return np.array(img) / 255.0

def batch_loader(paths, target_size=(224, 224)):
    """Loads a batch of images into a (N, H, W, C) array."""
    return np.array([image_loader(p, target_size) for p in paths])

def feature_extractor(paths, keras_model, save_path):
    """
    Extracts features using a frozen Keras CNN and saves as .npy.
    """
    images = batch_loader(paths)
    # Ensure images are in the right shape for the model
    features = keras_model.predict(images, verbose=1)
    np.save(save_path, features)
    print(f"Features saved to {save_path}")