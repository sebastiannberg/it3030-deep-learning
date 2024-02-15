import numpy as np


def softmax(X):
    # Subtract maximum value to help numerical stability
    return np.exp(X - np.max(X)) / np.exp(X - np.max(X)).sum(axis=0)
