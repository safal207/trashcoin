import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes softmax probabilities from unnormalized log probabilities (logits).
    This is numerically stable.
    """
    # Subtracting the max value for numerical stability
    x = x - np.max(x)

    e = np.exp(x)
    return e / np.sum(e)
