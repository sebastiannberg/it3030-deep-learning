import numpy as np


def softmax(x):
    # Subtract maximum value to help numerical stability
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum(axis=0)
