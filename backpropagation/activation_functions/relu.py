import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    # Derivative of ReLU is 1 for positive values, 0 otherwise
    return np.where(x > 0, 1, 0)
