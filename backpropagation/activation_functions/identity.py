import numpy as np


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)  # Derivative of identity function is always 1
