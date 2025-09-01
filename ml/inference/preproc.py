import numpy as np
from PIL import Image

def preprocess(img: Image.Image, size=224):
    """
    Preprocesses a PIL image for model inference.
    - Resizes to a square image.
    - Converts to a numpy array and scales to [0, 1].
    - Normalizes using ImageNet stats.
    - Transposes from HWC to CHW format.
    - Adds a batch dimension.
    """
    img = img.resize((size, size))
    arr = np.asarray(img).astype("float32") / 255.0

    # Normalize with ImageNet's mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)

    # Add batch dimension -> [1, C, H, W]
    return np.expand_dims(arr, 0).astype("float32")
