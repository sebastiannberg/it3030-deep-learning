import numpy as np


def relu(X):
    return np.maximum(0, X)


def relu_derivative(X):
    # Derivative of ReLU is 1 for positive values, 0 otherwise
    return np.where(X > 0, 1, 0)
