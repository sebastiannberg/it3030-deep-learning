import numpy as np


def tanh(X):
    return np.tanh(X)

def tanh_derivative(X):
    return 1 - np.tanh(X)**2
